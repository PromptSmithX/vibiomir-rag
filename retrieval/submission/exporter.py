"""Official ViBioMIR Submission Exporter.

Converts retrieval and RRF fusion artifacts into official submission JSON and ZIP
formats strictly complying with competition benchmark requirements.
"""

from __future__ import annotations

import json
import logging
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from retrieval.dense.config import SubmissionConfig
from retrieval.searcher import BM25Searcher
from retrieval.submission.chunk_selector import (
    ChunkSelectionConfig,
    ChunkSelector,
    strip_synthetic_prefixes,
)
from retrieval.submission.chunk_store import ChunkStore
from retrieval.submission.passage_diagnostics import (
    analyze_expanded_submission,
    export_diagnostics_json,
    format_passage_diagnostics_report,
)

logger = logging.getLogger(__name__)


@dataclass
class ExportSummary:
    """Summary statistics for submission export execution."""

    total_official_queries: int = 0
    total_queries_with_results: int = 0
    total_empty_queries: int = 0
    total_relevant_docs: int = 0
    total_relevant_chunks: int = 0
    missing_chunks_count: int = 0
    queries_under_target_chunks: int = 0
    total_chunk_deficit: int = 0
    chunks_skipped_cap_count: int = 0
    json_path: Path | None = None
    zip_path: Path | None = None
    metadata_path: Path | None = None
    diagnostics_path: Path | None = None
    passages_expanded_count: int = 0
    expansion_max_tokens: int | None = None
    top_docs_limit: int = 50
    top_chunks_limit: int = 50
    max_chunks_per_doc: int = 2


class SubmissionExporter:
    """Exports retrieval and fusion results to official ViBioMIR submission format.

    Strict Guarantees:
    1. Complete query coverage from official query.parquet (all 1,200 queries).
    2. Official schema: document IDs strictly formatted as integers (NO strings).
    3. Pure source chunk text: no synthetic prefixes, instructions, or hallucinations.
    4. Configurable retrieval depth (top 10, 30, 50, 100 docs/query).
    5. Intelligent chunk selection: quota per document, boilerplate filtering, deduplication.
    6. Optional passage expansion within source document bounds.
    7. Single JSON at root of ZIP archive.
    8. Preserves baseline artifacts without overwriting.
    """

    def __init__(
        self,
        config: SubmissionConfig | None = None,
        chunk_selection_config: ChunkSelectionConfig | None = None,
        chunk_selector: ChunkSelector | None = None,
    ) -> None:
        self.config = config or SubmissionConfig()
        self.chunk_selection_config = chunk_selection_config or ChunkSelectionConfig()
        self.chunk_selector = chunk_selector or ChunkSelector(self.chunk_selection_config)

    def export(
        self,
        retrieval_file: Path | str,
        query_path: Path | str,
        chunks_dir: Path | str,
        output_json_path: Path | str,
        zip_output_path: Path | str | None = None,
        corpus_path: Path | str | None = None,
        top_docs: int | None = None,
        top_chunks: int | None = None,
        max_chunks_per_doc: int | None = None,
        include_chunk_order: bool | None = None,
        doc_id_type: str | None = None,
        chunk_hits_file: Path | str | None = None,
        metadata_extra: dict[str, Any] | None = None,
        baseline_submission_path: Path | str | None = None,
        expand_passages: bool | None = None,
        expansion_max_tokens: int | None = None,
    ) -> ExportSummary:
        """Export official submission JSON and optional ZIP from pipeline artifacts.

        Args:
            retrieval_file: Path to rrf_fusion_top_*.parquet or dense_retrieval_top_*.parquet.
            query_path: Path to official query.parquet.
            chunks_dir: Path to directory containing chunk parquet shards.
            output_json_path: Destination path for submission.json.
            zip_output_path: Destination path for submission.zip (optional).
            corpus_path: Optional fallback corpus parquet path.
            top_docs: Maximum docs per query (overrides config).
            top_chunks: Maximum chunks per query (overrides config).
            max_chunks_per_doc: Maximum chunks per document (overrides config).
            include_chunk_order: Whether to include optional chunk_order field.
            doc_id_type: Must be 'int' for official evaluation.
            chunk_hits_file: Optional path to dense chunk-level hits parquet.
            metadata_extra: Additional experiment metadata to log in submission_metadata.json.
            baseline_submission_path: Optional path to baseline submission to preserve exact candidates.
            expand_passages: Whether to enable source passage expansion.
            expansion_max_tokens: Maximum token limit for passage expansion (e.g. 768 or 1024).

        Returns:
            ExportSummary containing metrics and file locations.
        """
        r_file = Path(retrieval_file)
        q_file = Path(query_path)
        c_dir = Path(chunks_dir)
        out_json = Path(output_json_path)

        if expand_passages is not None:
            self.chunk_selector.config.passage_expansion.enabled = expand_passages
            self.chunk_selector.passage_expander.config.enabled = expand_passages
        if expansion_max_tokens is not None:
            self.chunk_selector.config.passage_expansion.max_tokens = expansion_max_tokens
            self.chunk_selector.passage_expander.config.max_tokens = expansion_max_tokens

        limit_docs = top_docs if top_docs is not None else self.config.top_docs
        limit_chunks = top_chunks if top_chunks is not None else self.config.top_chunks
        with_order = (
            include_chunk_order
            if include_chunk_order is not None
            else self.config.include_chunk_order
        )

        # Enforce integer document IDs
        if max_chunks_per_doc is not None:
            self.chunk_selector.config.max_chunks_per_document = max_chunks_per_doc

        logger.info("=== EXPORTING OFFICIAL VIBIOMIR SUBMISSION ===")
        logger.info("Retrieval results file : %s", r_file)
        logger.info("Official query file     : %s", q_file)
        logger.info("Chunks directory        : %s", c_dir)
        logger.info("Output JSON destination : %s", out_json)
        logger.info(
            "Limits: top_docs=%d, top_chunks=%d, max_chunks_per_doc=%d, doc_id_type=int",
            limit_docs,
            limit_chunks,
            self.chunk_selector.config.max_chunks_per_document,
        )

        if not q_file.exists():
            raise FileNotFoundError(f"Official query file not found: {q_file}")
        if not r_file.exists():
            raise FileNotFoundError(f"Retrieval results file not found: {r_file}")

        # 1. Load official queries to guarantee 100% query coverage (1,200 queries)
        official_queries = BM25Searcher.load_queries_from_parquet(q_file)
        official_query_ids = [q.query_id for q in official_queries]
        logger.info("Loaded %d official queries from %s", len(official_query_ids), q_file.name)

        # 2. Load retrieval hits
        hits_table = pq.read_table(r_file)
        has_final_rank = "final_rank" in hits_table.column_names
        rank_col = "final_rank" if has_final_rank else "rank"
        has_dense_chunk = "dense_best_chunk_id" in hits_table.column_names
        has_best_chunk = "best_chunk_id" in hits_table.column_names

        qids = hits_table.column("query_id").to_pylist()
        dids = hits_table.column("doc_id").to_pylist()
        ranks = hits_table.column(rank_col).to_pylist()

        chunk_ids: list[str] = []
        if has_dense_chunk:
            chunk_ids = hits_table.column("dense_best_chunk_id").to_pylist()
        elif has_best_chunk:
            chunk_ids = hits_table.column("best_chunk_id").to_pylist()
        else:
            chunk_ids = [""] * len(qids)

        # Group hits by query_id, preserving rank order and deduplicating doc_ids
        hits_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
        dense_best_by_query: dict[int, dict[int, str]] = defaultdict(dict)

        for q, d, r, cid in zip(qids, dids, ranks, chunk_ids, strict=False):
            int_q = int(q)
            int_d = int(d)
            str_cid = str(cid or "")
            hits_by_query[int_q].append(
                {"doc_id": int_d, "rank": int(r), "best_chunk_id": str_cid}
            )
            if str_cid:
                dense_best_by_query[int_q][int_d] = str_cid

        # Sort each query's hits by rank ascending and eliminate any duplicate doc_id
        for q in hits_by_query:
            hits_by_query[q].sort(key=lambda x: x["rank"])
            seen_dids_in_q: set[int] = set()
            deduped_hits: list[dict[str, Any]] = []
            for h in hits_by_query[q]:
                did = h["doc_id"]
                if did not in seen_dids_in_q:
                    seen_dids_in_q.add(did)
                    deduped_hits.append(h)
            hits_by_query[q] = deduped_hits

        # 3. Load optional dense chunk hits if available
        dense_chunk_hits_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
        if chunk_hits_file and Path(chunk_hits_file).exists():
            try:
                ch_tbl = pq.read_table(chunk_hits_file)
                c_qids = ch_tbl.column("query_id").to_pylist()
                c_dids = ch_tbl.column("doc_id").to_pylist()
                c_cids = ch_tbl.column("chunk_id").to_pylist()
                c_scores = (
                    ch_tbl.column("score").to_pylist()
                    if "score" in ch_tbl.column_names
                    else [0.0] * len(c_qids)
                )
                for q, d, c, s in zip(c_qids, c_dids, c_cids, c_scores, strict=False):
                    dense_chunk_hits_by_query[int(q)].append(
                        {"query_id": int(q), "doc_id": int(d), "chunk_id": str(c), "score": float(s)}
                    )
                logger.info("Loaded dense chunk hits from %s", chunk_hits_file)
            except Exception as e:
                logger.warning("Could not load dense chunk hits from %s: %s", chunk_hits_file, e)

        # 4. Collect needed chunk IDs and doc IDs for preloading into ChunkStore
        needed_chunk_ids: set[str] = set()
        needed_doc_ids: set[int] = set()

        for q_id in official_query_ids:
            q_hits = hits_by_query.get(q_id, [])[:limit_docs]
            for h in q_hits:
                did = h["doc_id"]
                cid = h["best_chunk_id"]
                needed_doc_ids.add(did)
                if cid:
                    needed_chunk_ids.add(cid)

            # Also add dense chunk hits for top docs if present
            for ch in dense_chunk_hits_by_query.get(q_id, []):
                if ch["doc_id"] in needed_doc_ids:
                    needed_chunk_ids.add(ch["chunk_id"])

        # 5. Preload source chunk texts from chunk shards
        chunk_store = ChunkStore(chunks_dir=c_dir, corpus_path=corpus_path)
        chunk_store.preload_for_retrieval(
            needed_chunk_ids=needed_chunk_ids,
            needed_doc_ids=needed_doc_ids,
        )
        if self.chunk_selector.config.passage_expansion.enabled:
            logger.info("Preloading authentic source corpus text for passage expansion...")
            chunk_store.preload_corpus(needed_doc_ids=needed_doc_ids)

        # Optional: load baseline submission to preserve exact candidates
        base_by_qid: dict[int, dict[str, Any]] = {}
        if baseline_submission_path:
            base_p = Path(baseline_submission_path)
            if base_p.exists():
                logger.info("Reusing exact ranking and chunk candidates from baseline: %s", base_p)
                with open(base_p, encoding="utf-8") as bf:
                    base_data = json.load(bf)
                    base_by_qid = {int(x["id"]): x for x in base_data}

        # 6. Build official submission JSON structure
        self.chunk_selector.reset_diagnostics()

        submission_records: list[dict[str, Any]] = []
        total_docs_exported = 0
        total_chunks_exported = 0
        missing_chunks_count = 0
        empty_queries_count = 0
        queries_under_target: list[int] = []
        total_chunk_deficit = 0

        for q_id in official_query_ids:
            if base_by_qid and q_id in base_by_qid:
                base_item = base_by_qid[q_id]
                rel_docs = [int(d) for d in base_item.get("relevant_docs", [])][:limit_docs]
                selected_chunks: list[dict[str, Any]] = []
                for c_idx, c_obj in enumerate(base_item.get("relevant_chunks", [])[:limit_chunks]):
                    c_did = int(c_obj["doc_id"])
                    c_txt = c_obj.get("chunk_text", "")
                    if self.chunk_selector.config.passage_expansion.enabled:
                        exp_res = self.chunk_selector.passage_expander.expand_chunk(
                            doc_id=c_did,
                            chunk_id=f"{c_did}_anchor_{c_idx}",
                            chunk_text=c_txt,
                            chunk_store=chunk_store,
                        )
                        final_text = exp_res.expanded_text
                    else:
                        final_text = strip_synthetic_prefixes(c_txt)

                    selected_chunks.append(
                        {
                            "doc_id": c_did,
                            "chunk_text": final_text,
                            "chunk_id": f"{c_did}_anchor_{c_idx}",
                        }
                    )
            else:
                q_hits = hits_by_query.get(q_id, [])[:limit_docs]
                rel_docs = [int(h["doc_id"]) for h in q_hits]
                selected_chunks = self.chunk_selector.select_chunks(
                    query_id=q_id,
                    relevant_docs=rel_docs,
                    chunk_store=chunk_store,
                    dense_chunk_hits=dense_chunk_hits_by_query.get(q_id),
                    dense_best_chunk_map=dense_best_by_query.get(q_id),
                    top_k=limit_chunks,
                )

            rel_chunks: list[dict[str, Any]] = []
            for c_idx, c_item in enumerate(selected_chunks):
                clean_text = strip_synthetic_prefixes(c_item["chunk_text"])
                chunk_entry: dict[str, Any] = {
                    "doc_id": int(c_item["doc_id"]),  # STRICT INTEGER
                    "chunk_text": clean_text,
                }
                if with_order:
                    chunk_entry["chunk_order"] = c_idx
                rel_chunks.append(chunk_entry)

            if len(rel_docs) == 0:
                empty_queries_count += 1
            if len(rel_docs) > 0 and len(rel_chunks) == 0:
                missing_chunks_count += len(rel_docs)
            if len(rel_docs) > 0 and len(rel_chunks) < limit_chunks:
                queries_under_target.append(q_id)
                total_chunk_deficit += (limit_chunks - len(rel_chunks))

            total_docs_exported += len(rel_docs)
            total_chunks_exported += len(rel_chunks)

            # Strict official JSON schema:
            # - id: integer
            # - relevant_docs: list[int]
            # - relevant_chunks: list[{"doc_id": int, "chunk_text": str}]
            submission_records.append(
                {
                    "id": int(q_id),
                    "relevant_docs": rel_docs,
                    "relevant_chunks": rel_chunks,
                }
            )

        selector_diag = self.chunk_selector.get_diagnostics_summary()
        if queries_under_target:
            logger.warning(
                "Chunk pool capacity notice: %d queries have fewer than %d valid chunks (total shortfall: %d chunks).",
                len(queries_under_target),
                limit_chunks,
                total_chunk_deficit,
            )
        else:
            logger.info("All %d queries have exactly %d valid chunks.", len(official_query_ids), limit_chunks)

        # 7. Write submission.json
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(submission_records, f, ensure_ascii=False, indent=2)

        logger.info(
            "Wrote official submission to %s (%d queries, %d docs, %d chunks)",
            out_json.resolve(),
            len(submission_records),
            total_docs_exported,
            total_chunks_exported,
        )

        # 8. Package ZIP if requested
        out_zip_path: Path | None = None
        if zip_output_path:
            out_zip_path = Path(zip_output_path)
            self._create_submission_zip(out_json, out_zip_path)

        # 9. Compute and export passage expansion diagnostics if enabled
        diag_file: Path | None = None
        diag_summary = None
        if self.chunk_selector.config.passage_expansion.enabled:
            diag_file = out_json.parent / "diagnostics.json"
            base_cmp_path = Path(baseline_submission_path) if baseline_submission_path else None
            if not base_cmp_path:
                default_base = Path("data/experiments/a1_2_max2_chunks/submission/submission.json")
                if default_base.exists():
                    base_cmp_path = default_base

            diag_summary = analyze_expanded_submission(
                submission_path=out_json,
                baseline_path=base_cmp_path,
                max_tokens=self.chunk_selector.config.passage_expansion.max_tokens,
                token_counter=self.chunk_selector.token_counter,
                target_chunks=limit_chunks,
            )
            export_diagnostics_json(diag_summary.to_dict(), diag_file)
            logger.info("Saved passage expansion diagnostics to %s", diag_file.resolve())

        # 10. Write experiment metadata alongside submission for reproducibility
        metadata_file = out_json.parent / "submission_metadata.json"
        metadata_payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "retrieval_file": str(r_file.resolve()),
            "query_path": str(q_file.resolve()),
            "chunks_dir": str(c_dir.resolve()),
            "submission_json": str(out_json.resolve()),
            "submission_zip": str(out_zip_path.resolve()) if out_zip_path else None,
            "top_docs_limit": limit_docs,
            "top_chunks_limit": limit_chunks,
            "chunk_selection": {
                "max_chunks_per_document": self.chunk_selector.config.max_chunks_per_document,
                "deduplicate": self.chunk_selector.config.deduplicate,
                "dedup_threshold": self.chunk_selector.config.dedup_threshold,
                "filter_boilerplate": self.chunk_selector.config.filter_boilerplate,
                "min_chunk_chars": self.chunk_selector.config.min_chunk_chars,
                "passage_expansion": {
                    "enabled": self.chunk_selector.config.passage_expansion.enabled,
                    "max_tokens": self.chunk_selector.config.passage_expansion.max_tokens,
                    "anchor": self.chunk_selector.config.passage_expansion.anchor,
                    "tokenizer": self.chunk_selector.token_counter.__class__.__name__,
                },
            },
            "summary_stats": {
                "total_official_queries": len(official_query_ids),
                "total_queries_with_results": len(official_query_ids) - empty_queries_count,
                "total_empty_queries": empty_queries_count,
                "total_relevant_docs": total_docs_exported,
                "avg_docs_per_query": round(total_docs_exported / len(official_query_ids), 2),
                "total_relevant_chunks": total_chunks_exported,
                "avg_chunks_per_query": round(total_chunks_exported / len(official_query_ids), 2),
                "missing_chunks_count": missing_chunks_count,
                "queries_under_target_chunks": len(queries_under_target),
                "total_chunk_deficit": total_chunk_deficit,
            },
            "diagnostics": {
                "chunks_skipped_cap": selector_diag.get("total_chunks_skipped_cap", 0),
                "chunks_skipped_boilerplate": selector_diag.get("total_chunks_skipped_boilerplate", 0),
                "chunks_skipped_dedup": selector_diag.get("total_chunks_skipped_dedup", 0),
                "queries_under_target_count": len(queries_under_target),
                "queries_under_target_ids": queries_under_target[:50],
                "total_chunk_deficit": total_chunk_deficit,
                "docs_with_1_chunk": selector_diag.get("docs_with_1_chunk", 0),
                "docs_with_2_chunks": selector_diag.get("docs_with_2_chunks", 0),
                "docs_with_3_chunks": selector_diag.get("docs_with_3_chunks", 0),
                "docs_with_gt_2_chunks": selector_diag.get("docs_with_gt_2_chunks", 0),
                "avg_unique_docs_in_chunks": selector_diag.get("avg_unique_docs", 0.0),
                "median_unique_docs_in_chunks": selector_diag.get("median_unique_docs", 0.0),
            },
        }
        if diag_summary:
            metadata_payload["passage_expansion_diagnostics"] = diag_summary.to_dict()
        if metadata_extra:
            metadata_payload["experiment_context"] = metadata_extra

        with open(metadata_file, "w", encoding="utf-8") as mf:
            json.dump(metadata_payload, mf, indent=2, ensure_ascii=False)

        summary = ExportSummary(
            total_official_queries=len(official_query_ids),
            total_queries_with_results=len(official_query_ids) - empty_queries_count,
            total_empty_queries=empty_queries_count,
            total_relevant_docs=total_docs_exported,
            total_relevant_chunks=total_chunks_exported,
            missing_chunks_count=missing_chunks_count,
            queries_under_target_chunks=len(queries_under_target),
            total_chunk_deficit=total_chunk_deficit,
            chunks_skipped_cap_count=selector_diag.get("total_chunks_skipped_cap", 0),
            json_path=out_json.resolve(),
            zip_path=out_zip_path.resolve() if out_zip_path else None,
            metadata_path=metadata_file.resolve(),
            diagnostics_path=diag_file.resolve() if diag_file else None,
            passages_expanded_count=diag_summary.expanded_chunks_count if diag_summary else 0,
            expansion_max_tokens=self.chunk_selector.config.passage_expansion.max_tokens
            if self.chunk_selector.config.passage_expansion.enabled
            else None,
            top_docs_limit=limit_docs,
            top_chunks_limit=limit_chunks,
            max_chunks_per_doc=self.chunk_selector.config.max_chunks_per_document,
        )

        return summary

    @staticmethod
    def _create_submission_zip(json_path: Path, zip_path: Path) -> Path:
        """Package submission.json into submission.zip containing strictly
        ONE JSON file at the root.
        """
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_path.name)

        logger.info("Packaged submission ZIP: %s (contains: %s)", zip_path.resolve(), json_path.name)
        return zip_path
