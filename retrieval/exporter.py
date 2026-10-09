"""Candidate exporter and union generation module for downstream RAG pipeline."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.models import CandidateLevelStats, RetrievalHit

logger = logging.getLogger(__name__)


class CandidateExporter:
    """Exports candidate subsets, generates union sets with query attribution,
    and produces stats."""

    SCHEMA_HITS = pa.schema(
        [
            ("query_id", pa.int64()),
            ("doc_id", pa.int64()),
            ("rank", pa.int32()),
            ("bm25_score", pa.float32()),
        ]
    )

    SCHEMA_UNION = pa.schema(
        [
            ("doc_id", pa.int64()),
            ("match_count", pa.int32()),
            ("best_rank", pa.int32()),
            ("best_score", pa.float32()),
            ("query_ids", pa.list_(pa.int64())),
            (
                "query_matches",
                pa.list_(
                    pa.struct(
                        [
                            ("query_id", pa.int64()),
                            ("rank", pa.int32()),
                            ("score", pa.float32()),
                        ]
                    )
                ),
            ),
        ]
    )

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_hits(
        self,
        hits: list[RetrievalHit],
        filename: str = "retrieval_top1000.parquet",
    ) -> Path:
        """Export raw retrieval hits to Parquet."""
        target_path = self.output_dir / filename
        data = {
            "query_id": [h.query_id for h in hits],
            "doc_id": [h.doc_id for h in hits],
            "rank": [h.rank for h in hits],
            "bm25_score": [h.bm25_score for h in hits],
        }
        tbl = pa.Table.from_pydict(data, schema=self.SCHEMA_HITS)
        pq.write_table(tbl, target_path, compression="zstd")
        logger.info("Exported %d hits to %s", len(hits), target_path)
        return target_path

    def compute_level_stats(
        self,
        hits: list[RetrievalHit],
        k_levels: list[int] | None = None,
    ) -> list[CandidateLevelStats]:
        """Compute candidate size and unique doc_id count for multiple K values."""
        if k_levels is None:
            k_levels = [100, 200, 300, 500, 1000]

        stats_list: list[CandidateLevelStats] = []
        for k in sorted(k_levels):
            k_hits = [h for h in hits if h.rank <= k]
            unique_docs = {h.doc_id for h in k_hits}
            tot = len(k_hits)
            uniq = len(unique_docs)
            ratio = uniq / tot if tot > 0 else 0.0
            avg_q_per_doc = tot / uniq if uniq > 0 else 0.0

            stats_list.append(
                CandidateLevelStats(
                    k=k,
                    total_candidates=tot,
                    unique_docs=uniq,
                    unique_ratio=ratio,
                    avg_queries_per_doc=avg_q_per_doc,
                )
            )

        return stats_list

    def export_candidate_levels(
        self,
        hits: list[RetrievalHit],
        k_levels: list[int] | None = None,
    ) -> dict[int, Path]:
        """Export candidates at each cutoff level."""
        if k_levels is None:
            k_levels = [100, 200, 300, 500, 1000]

        exported_paths: dict[int, Path] = {}
        for k in sorted(k_levels):
            k_hits = [h for h in hits if h.rank <= k]
            filename = f"candidates_top_{k}.parquet"
            path = self.export_hits(k_hits, filename=filename)
            exported_paths[k] = path

        return exported_paths

    def generate_union_candidates(
        self,
        hits: list[RetrievalHit],
        k: int = 200,
        filename: str = "candidate_doc_ids.parquet",
    ) -> tuple[Path, int]:
        """Generate deduplicated union candidate set with full query attribution
        for downstream steps.

        Args:
            hits: All retrieval hits.
            k: Top-K cutoff per query to include in the union.
            filename: Target output file name.

        Returns:
            (Path, int): Target path and total unique doc count in union.
        """
        filtered_hits = [h for h in hits if h.rank <= k]

        # Aggregate by doc_id
        grouped: dict[int, list[RetrievalHit]] = defaultdict(list)
        for h in filtered_hits:
            grouped[h.doc_id].append(h)

        doc_ids: list[int] = []
        match_counts: list[int] = []
        best_ranks: list[int] = []
        best_scores: list[float] = []
        query_ids_list: list[list[int]] = []
        matches_list: list[list[dict[str, Any]]] = []

        for doc_id, doc_hits in sorted(grouped.items()):
            doc_ids.append(doc_id)
            match_counts.append(len(doc_hits))
            best_ranks.append(min(h.rank for h in doc_hits))
            best_scores.append(max(h.bm25_score for h in doc_hits))

            q_ids = sorted(list({h.query_id for h in doc_hits}))
            query_ids_list.append(q_ids)

            matches = [
                {"query_id": h.query_id, "rank": h.rank, "score": h.bm25_score}
                for h in sorted(doc_hits, key=lambda x: x.rank)
            ]
            matches_list.append(matches)

        data = {
            "doc_id": doc_ids,
            "match_count": match_counts,
            "best_rank": best_ranks,
            "best_score": best_scores,
            "query_ids": query_ids_list,
            "query_matches": matches_list,
        }

        tbl = pa.Table.from_pydict(data, schema=self.SCHEMA_UNION)
        target_path = self.output_dir / filename
        pq.write_table(tbl, target_path, compression="zstd")

        logger.info(
            "Generated union set at K=%d: %d unique documents saved to %s",
            k,
            len(doc_ids),
            target_path,
        )
        return target_path, len(doc_ids)
