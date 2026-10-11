"""BM25 + Dense Document-level Reciprocal Rank Fusion (RRF) module."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.dense.config import FusionConfig
from retrieval.models import DenseDocHit, FusedRetrievalHit, FusionStats, RetrievalHit

logger = logging.getLogger(__name__)

SCHEMA_FUSED = pa.schema(
    [
        ("query_id", pa.int64()),
        ("doc_id", pa.int64()),
        ("final_rank", pa.int32()),
        ("fused_score", pa.float64()),
        ("bm25_rank", pa.int32()),
        ("bm25_score", pa.float32()),
        ("dense_rank", pa.int32()),
        ("dense_score", pa.float32()),
        ("dense_best_chunk_id", pa.string()),
    ]
)


class ReciprocalRankFusion:
    """Document-level Reciprocal Rank Fusion combining BM25 and Dense Retrieval rankings."""

    def __init__(self, config: FusionConfig) -> None:
        self.config = config

    def fuse(
        self,
        bm25_hits: list[RetrievalHit] | None = None,
        dense_hits: list[DenseDocHit] | None = None,
        bm25_parquet_path: Path | str | None = None,
        dense_parquet_path: Path | str | None = None,
    ) -> tuple[list[FusedRetrievalHit], FusionStats]:
        """Perform RRF merging from memory lists or parquet file paths."""
        start_time = time.time()

        # 1. Load BM25 hits
        if bm25_hits is None and bm25_parquet_path is not None:
            bm25_hits = self._load_bm25_parquet(bm25_parquet_path)
        bm25_hits = bm25_hits or []

        # 2. Load Dense hits
        if dense_hits is None and dense_parquet_path is not None:
            dense_hits = self._load_dense_parquet(dense_parquet_path)
        dense_hits = dense_hits or []

        logger.info(
            "Merging BM25 (%d hits) and Dense (%d hits) with RRF k=%d (bm25_w=%.2f, dense_w=%.2f)...",
            len(bm25_hits),
            len(dense_hits),
            self.config.rrf_k,
            self.config.bm25_weight,
            self.config.dense_weight,
        )

        # 3. Organize hits by query_id -> doc_id
        # query_id -> {doc_id: (rank, score)}
        bm25_by_query: dict[int, dict[int, tuple[int, float]]] = defaultdict(dict)
        for h in bm25_hits:
            bm25_by_query[h.query_id][h.doc_id] = (h.rank, h.bm25_score)

        # query_id -> {doc_id: (rank, score, best_chunk_id)}
        dense_by_query: dict[int, dict[int, tuple[int, float, str]]] = defaultdict(dict)
        for dh in dense_hits:
            dense_by_query[dh.query_id][dh.doc_id] = (dh.rank, dh.score, dh.best_chunk_id)

        all_query_ids = sorted(set(bm25_by_query.keys()) | set(dense_by_query.keys()))

        fused_hits: list[FusedRetrievalHit] = []
        unique_docs_set: set[int] = set()
        bm25_only_count = 0
        dense_only_count = 0
        both_count = 0

        k_rrf = self.config.rrf_k
        w_bm25 = self.config.bm25_weight
        w_dense = self.config.dense_weight

        for q_id in all_query_ids:
            q_bm25 = bm25_by_query.get(q_id, {})
            q_dense = dense_by_query.get(q_id, {})

            all_docs = set(q_bm25.keys()) | set(q_dense.keys())
            doc_scores: list[dict[str, Any]] = []

            for doc_id in all_docs:
                unique_docs_set.add(doc_id)
                score = 0.0
                bm25_r = None
                bm25_s = None
                dense_r = None
                dense_s = None
                best_chunk = None

                in_bm25 = doc_id in q_bm25
                in_dense = doc_id in q_dense

                if in_bm25 and in_dense:
                    both_count += 1
                elif in_bm25:
                    bm25_only_count += 1
                else:
                    dense_only_count += 1

                if in_bm25:
                    bm25_r, bm25_s = q_bm25[doc_id]
                    score += w_bm25 / (k_rrf + bm25_r)

                if in_dense:
                    dense_r, dense_s, best_chunk = q_dense[doc_id]
                    score += w_dense / (k_rrf + dense_r)

                doc_scores.append(
                    {
                        "doc_id": doc_id,
                        "fused_score": score,
                        "bm25_rank": bm25_r,
                        "bm25_score": bm25_s,
                        "dense_rank": dense_r,
                        "dense_score": dense_s,
                        "dense_best_chunk_id": best_chunk,
                    }
                )

            # Sort descending by fused_score
            doc_scores.sort(
                key=lambda x: (x["fused_score"], -(x["bm25_rank"] or 999999)),
                reverse=True,
            )

            if self.config.top_k:
                doc_scores = doc_scores[: self.config.top_k]

            for final_r0, item in enumerate(doc_scores):
                fused_hits.append(
                    FusedRetrievalHit(
                        query_id=q_id,
                        doc_id=item["doc_id"],
                        final_rank=final_r0 + 1,
                        fused_score=item["fused_score"],
                        bm25_rank=item["bm25_rank"],
                        bm25_score=item["bm25_score"],
                        dense_rank=item["dense_rank"],
                        dense_score=item["dense_score"],
                        dense_best_chunk_id=item["dense_best_chunk_id"],
                    )
                )

        stats = FusionStats(
            num_queries=len(all_query_ids),
            rrf_k=self.config.rrf_k,
            total_fused_hits=len(fused_hits),
            unique_docs=len(unique_docs_set),
            bm25_only_docs=bm25_only_count,
            dense_only_docs=dense_only_count,
            both_systems_docs=both_count,
        )

        logger.info(
            "Fusion completed in %.2fs: %d queries, %d total fused hits, %d unique docs",
            time.time() - start_time,
            len(all_query_ids),
            len(fused_hits),
            len(unique_docs_set),
        )
        return fused_hits, stats

    @staticmethod
    def _load_bm25_parquet(path: Path | str) -> list[RetrievalHit]:
        tbl = pq.read_table(path)
        qids = tbl.column("query_id").to_pylist()
        dids = tbl.column("doc_id").to_pylist()
        ranks = tbl.column("rank").to_pylist()
        scores = (
            tbl.column("bm25_score").to_pylist()
            if "bm25_score" in tbl.column_names
            else [0.0] * len(qids)
        )
        return [
            RetrievalHit(query_id=q, doc_id=d, rank=r, bm25_score=s)
            for q, d, r, s in zip(qids, dids, ranks, scores, strict=False)
        ]

    @staticmethod
    def _load_dense_parquet(path: Path | str) -> list[DenseDocHit]:
        tbl = pq.read_table(path)
        qids = tbl.column("query_id").to_pylist()
        dids = tbl.column("doc_id").to_pylist()
        ranks = tbl.column("rank").to_pylist()
        scores = tbl.column("score").to_pylist()
        b_chunks = (
            tbl.column("best_chunk_id").to_pylist()
            if "best_chunk_id" in tbl.column_names
            else [""] * len(qids)
        )
        b_scores = (
            tbl.column("best_chunk_score").to_pylist()
            if "best_chunk_score" in tbl.column_names
            else [0.0] * len(qids)
        )
        return [
            DenseDocHit(
                query_id=q,
                doc_id=d,
                rank=r,
                score=s,
                best_chunk_id=bc,
                best_chunk_score=bs,
            )
            for q, d, r, s, bc, bs in zip(
                qids, dids, ranks, scores, b_chunks, b_scores, strict=False
            )
        ]

    @staticmethod
    def export_fused_results(
        hits: list[FusedRetrievalHit],
        output_dir: Path | str,
        filename_prefix: str = "rrf_fusion_top_1000",
    ) -> tuple[Path, Path]:
        """Export fused hits to both Parquet and CSV for verification and submission."""
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        parquet_path = out_d / f"{filename_prefix}.parquet"
        csv_path = out_d / f"{filename_prefix}.csv"

        # 1. Parquet
        tbl = pa.Table.from_pydict(
            {
                "query_id": [h.query_id for h in hits],
                "doc_id": [h.doc_id for h in hits],
                "final_rank": [h.final_rank for h in hits],
                "fused_score": [h.fused_score for h in hits],
                "bm25_rank": [h.bm25_rank if h.bm25_rank is not None else -1 for h in hits],
                "bm25_score": [
                    h.bm25_score if h.bm25_score is not None else 0.0 for h in hits
                ],
                "dense_rank": [h.dense_rank if h.dense_rank is not None else -1 for h in hits],
                "dense_score": [
                    h.dense_score if h.dense_score is not None else 0.0 for h in hits
                ],
                "dense_best_chunk_id": [h.dense_best_chunk_id or "" for h in hits],
            },
            schema=SCHEMA_FUSED,
        )
        pq.write_table(tbl, parquet_path, compression="zstd")

        # 2. CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "query_id",
                    "doc_id",
                    "final_rank",
                    "fused_score",
                    "bm25_rank",
                    "dense_rank",
                    "dense_best_chunk_id",
                ]
            )
            for h in hits:
                writer.writerow(
                    [
                        h.query_id,
                        h.doc_id,
                        h.final_rank,
                        f"{h.fused_score:.6f}",
                        h.bm25_rank or "",
                        h.dense_rank or "",
                        h.dense_best_chunk_id or "",
                    ]
                )

        logger.info(
            "Exported %d fused hits to:\n  - Parquet: %s\n  - CSV: %s",
            len(hits),
            parquet_path,
            csv_path,
        )
        return parquet_path, csv_path
