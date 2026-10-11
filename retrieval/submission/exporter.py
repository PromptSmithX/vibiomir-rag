"""Official ViBioMIR Submission Exporter.

Converts existing retrieval and RRF fusion artifacts into the official
submission JSON and ZIP formats matching official benchmark specifications.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from retrieval.dense.config import SubmissionConfig
from retrieval.searcher import BM25Searcher
from retrieval.submission.chunk_store import ChunkStore

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
    json_path: Path | None = None
    zip_path: Path | None = None
    top_docs_limit: int = 10
    top_chunks_limit: int = 10


class SubmissionExporter:
    """Exports retrieval and fusion results to official ViBioMIR submission format.

    Guarantees:
    1. Complete query coverage from official query.parquet.
    2. Exact schema compliance (doc_id as string, id as integer).
    3. Pure source chunk text without embedding prefixes or synthetic hallucinations.
    4. Deterministic formatting and clean ZIP packaging.
    """

    def __init__(self, config: SubmissionConfig | None = None) -> None:
        self.config = config or SubmissionConfig()

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
        include_chunk_order: bool | None = None,
        doc_id_type: str | None = None,
    ) -> ExportSummary:
        """Export submission JSON and optional ZIP from artifacts.

        Args:
            retrieval_file: Path to rrf_fusion_top_*.parquet or dense_retrieval_top_*.parquet.
            query_path: Path to official query.parquet.
            chunks_dir: Path to directory containing chunk parquet shards.
            output_json_path: Target path for submission.json.
            zip_output_path: Target path for submission.zip (optional).
            corpus_path: Optional fallback corpus parquet path.
            top_docs: Maximum docs per query (overrides config).
            top_chunks: Maximum chunks per query (overrides config).
            include_chunk_order: Whether to include optional chunk_order field.
            doc_id_type: Data type for doc_id ('int' or 'str', overrides config).

        Returns:
            ExportSummary with counts and file paths.
        """
        r_file = Path(retrieval_file)
        q_file = Path(query_path)
        c_dir = Path(chunks_dir)
        out_json = Path(output_json_path)

        limit_docs = top_docs if top_docs is not None else self.config.top_docs
        limit_chunks = top_chunks if top_chunks is not None else self.config.top_chunks
        with_order = (
            include_chunk_order
            if include_chunk_order is not None
            else self.config.include_chunk_order
        )
        doc_id_mode = str(
            doc_id_type
            if doc_id_type is not None
            else getattr(self.config, "doc_id_type", "int")
        ).lower()

        logger.info("=== EXPORTING OFFICIAL VIBIOMIR SUBMISSION ===")
        logger.info("Retrieval results file : %s", r_file)
        logger.info("Official query file     : %s", q_file)
        logger.info("Chunks directory        : %s", c_dir)
        logger.info("Output JSON destination : %s", out_json)
        logger.info(
            "Limits: top_docs=%d, top_chunks=%d, chunk_order=%s, doc_id_type=%s",
            limit_docs,
            limit_chunks,
            with_order,
            doc_id_mode,
        )

        if not q_file.exists():
            raise FileNotFoundError(f"Official query file not found: {q_file}")
        if not r_file.exists():
            raise FileNotFoundError(f"Retrieval results file not found: {r_file}")

        # 1. Load official queries to guarantee 100% query coverage
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

        # Group hits by query_id, preserving rank order
        hits_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for q, d, r, cid in zip(qids, dids, ranks, chunk_ids, strict=False):
            hits_by_query[int(q)].append(
                {"doc_id": int(d), "rank": int(r), "best_chunk_id": str(cid or "")}
            )

        # Sort each query's hits by rank ascending
        for q in hits_by_query:
            hits_by_query[q].sort(key=lambda x: x["rank"])

        # 3. Collect needed chunk IDs and doc IDs for preloading
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

        # 4. Preload source chunk texts from chunk shards
        chunk_store = ChunkStore(chunks_dir=c_dir, corpus_path=corpus_path)
        chunk_store.preload_for_retrieval(
            needed_chunk_ids=needed_chunk_ids,
            needed_doc_ids=needed_doc_ids,
        )

        # 5. Build official submission JSON structure
        submission_records: list[dict[str, Any]] = []
        total_docs_exported = 0
        total_chunks_exported = 0
        missing_chunks_count = 0
        empty_queries_count = 0

        for q_id in official_query_ids:
            q_hits = hits_by_query.get(q_id, [])[:limit_docs]

            rel_docs: list[Any] = []
            rel_chunks: list[dict[str, Any]] = []

            chunk_order_counter = 0
            for h in q_hits:
                doc_id_int = h["doc_id"]
                doc_id_val: int | str = int(doc_id_int) if doc_id_mode == "int" else str(doc_id_int)
                rel_docs.append(doc_id_val)

                # Resolve chunk text if we haven't reached top_chunks limit
                if len(rel_chunks) < limit_chunks:
                    best_cid = h["best_chunk_id"]
                    chunk_text = None

                    # A. Try best chunk ID from dense retrieval
                    if best_cid:
                        chunk_text = chunk_store.get_chunk_text(best_cid)

                    # B. Fallback to first available chunk for doc_id
                    if not chunk_text:
                        fb = chunk_store.get_fallback_chunk_for_doc(doc_id_int)
                        if fb:
                            chunk_text = fb[1]

                    # C. Fallback to corpus raw text if not in candidate shards
                    if not chunk_text and corpus_path:
                        chunk_text = chunk_store.get_text_from_corpus(doc_id_int)

                    if chunk_text:
                        # Ensure pure source text without any title/heading prefix
                        clean_text = self._strip_any_synthetic_prefix(chunk_text)
                        chunk_item: dict[str, Any] = {
                            "doc_id": doc_id_val,
                            "chunk_text": clean_text,
                        }
                        if with_order:
                            chunk_item["chunk_order"] = chunk_order_counter
                            chunk_order_counter += 1

                        rel_chunks.append(chunk_item)
                    else:
                        missing_chunks_count += 1
                        logger.debug("Could not resolve chunk text for doc_id %d (query %d)", doc_id_int, q_id)

            if not rel_docs:
                empty_queries_count += 1

            total_docs_exported += len(rel_docs)
            total_chunks_exported += len(rel_chunks)

            # Strict schema: id (int), relevant_docs (list[str]), relevant_chunks (list[dict])
            submission_records.append(
                {
                    "id": int(q_id),
                    "relevant_docs": rel_docs,
                    "relevant_chunks": rel_chunks,
                }
            )

        # 6. Save submission.json
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

        # 7. Package ZIP if requested
        out_zip_path: Path | None = None
        if zip_output_path:
            out_zip_path = Path(zip_output_path)
            self._create_submission_zip(out_json, out_zip_path)

        summary = ExportSummary(
            total_official_queries=len(official_query_ids),
            total_queries_with_results=len(official_query_ids) - empty_queries_count,
            total_empty_queries=empty_queries_count,
            total_relevant_docs=total_docs_exported,
            total_relevant_chunks=total_chunks_exported,
            missing_chunks_count=missing_chunks_count,
            json_path=out_json.resolve(),
            zip_path=out_zip_path.resolve() if out_zip_path else None,
            top_docs_limit=limit_docs,
            top_chunks_limit=limit_chunks,
        )

        return summary

    @staticmethod
    def _strip_any_synthetic_prefix(text: str) -> str:
        """Defensive guard to ensure no embedding instruction or metadata prefix
        leaks into official submission chunk_text.
        """
        s = text.strip()
        # If starts with model instruction
        if s.startswith("Instruct:"):
            # Strip prompt
            idx = s.find("\nQuery: ")
            if idx != -1:
                s = s[idx + len("\nQuery: ") :].strip()

        # If starts with [Tiêu đề: ...] [Mục: ...]
        while s.startswith("[Tiêu đề:") or s.startswith("[Mục:"):
            end_bracket = s.find("]")
            if end_bracket != -1:
                s = s[end_bracket + 1 :].strip()
            else:
                break

        return s

    @staticmethod
    def _create_submission_zip(json_path: Path, zip_path: Path) -> Path:
        """Package submission.json into submission.zip containing strictly
        one JSON file at the root.
        """
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            zip_path.unlink()

        # Write single file at root: arcname matches filename (e.g. submission.json)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_path.name)

        logger.info("Packaged submission ZIP: %s (contains: %s)", zip_path.resolve(), json_path.name)
        return zip_path
