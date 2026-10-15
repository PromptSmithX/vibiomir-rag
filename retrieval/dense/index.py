"""Dense vector index manager and chunk-to-document retrieval engine."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.dense.config import DenseRetrievalConfig, IndexConfig
from retrieval.models import ChunkHit, DenseDocHit, DenseRetrievalStats

logger = logging.getLogger(__name__)

SCHEMA_DENSE_HITS = pa.schema(
    [
        ("query_id", pa.int64()),
        ("doc_id", pa.int64()),
        ("rank", pa.int32()),
        ("score", pa.float32()),
        ("best_chunk_id", pa.string()),
        ("best_chunk_score", pa.float32()),
    ]
)

SCHEMA_CHUNK_HITS = pa.schema(
    [
        ("query_id", pa.int64()),
        ("doc_id", pa.int64()),
        ("chunk_id", pa.string()),
        ("rank", pa.int32()),
        ("score", pa.float32()),
    ]
)


class NumpyVectorIndex:
    """Exact cosine/inner-product index using pure NumPy.
    Guarantees 100% offline unit testability and CPU compatibility without FAISS dependency.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if len(self.vectors) == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute inner products and return top_k (scores, indices) for each query vector."""
        if len(self.vectors) == 0:
            return np.empty((len(query_vectors), 0)), np.empty((len(query_vectors), 0), dtype=int)

        top_k = min(top_k, len(self.vectors))
        # Matrix multiplication (Q x D) @ (D x N) -> (Q x N)
        scores = np.dot(query_vectors.astype(np.float32), self.vectors.T)

        all_top_scores = []
        all_top_indices = []

        for row in scores:
            # Use argpartition for fast top_k selection
            if top_k < len(row):
                partition_idx = np.argpartition(-row, top_k)[:top_k]
                sorted_sub_idx = partition_idx[np.argsort(-row[partition_idx])]
            else:
                sorted_sub_idx = np.argsort(-row)

            all_top_scores.append(row[sorted_sub_idx])
            all_top_indices.append(sorted_sub_idx)

        return np.array(all_top_scores, dtype=np.float32), np.array(all_top_indices, dtype=np.int64)

    def save(self, file_path: Path | str) -> None:
        np.save(file_path, self.vectors)

    @classmethod
    def load(cls, file_path: Path | str) -> NumpyVectorIndex:
        vecs = np.load(file_path)
        idx = cls(dimension=vecs.shape[1])
        idx.add(vecs)
        return idx


class FaissVectorIndex:
    """FAISS-powered Inner Product (Cosine) index for high-speed multi-core search."""

    def __init__(self, dimension: int) -> None:
        import faiss  # type: ignore

        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        # Ensure contiguous
        vectors = np.ascontiguousarray(vectors)
        self._index.add(vectors)

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = np.ascontiguousarray(query_vectors.astype(np.float32))
        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vectors, top_k)
        return scores, indices

    def save(self, file_path: Path | str) -> None:
        import faiss  # type: ignore

        faiss.write_index(self._index, str(file_path))

    @classmethod
    def load(cls, file_path: Path | str) -> FaissVectorIndex:
        import faiss  # type: ignore

        raw_index = faiss.read_index(str(file_path))
        idx = cls.__new__(cls)
        idx.dimension = raw_index.d
        idx._index = raw_index
        return idx


class DenseIndex:
    """High-level index wrapper managing vector index and chunk/doc ID mappings."""

    def __init__(
        self,
        index_backend: FaissVectorIndex | NumpyVectorIndex,
        chunk_ids: list[str],
        doc_ids: list[int],
    ) -> None:
        self.index = index_backend
        self.chunk_ids = chunk_ids
        self.doc_ids = doc_ids
        assert len(chunk_ids) == len(doc_ids), "Mismatched chunk_ids and doc_ids length"

    @classmethod
    def build_from_artifacts(
        cls,
        embeddings_dir: Path | str,
        index_config: IndexConfig,
    ) -> DenseIndex:
        """Construct dense vector index from embeddings.npy and embeddings_meta.parquet."""
        emb_dir = Path(embeddings_dir)
        npy_path = emb_dir / "embeddings.npy"
        meta_path = emb_dir / "embeddings_meta.parquet"

        if not npy_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {npy_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Embeddings metadata not found: {meta_path}")

        logger.info("Loading embeddings from %s...", npy_path)
        vectors = np.load(npy_path)
        dimension = vectors.shape[1] if len(vectors) > 0 else 1024

        tbl_meta = pq.read_table(meta_path)
        chunk_ids = tbl_meta.column("chunk_id").to_pylist()
        doc_ids = tbl_meta.column("doc_id").to_pylist()

        # Instantiate backend: FAISS if requested & available, else fallback
        backend: FaissVectorIndex | NumpyVectorIndex
        use_faiss = index_config.use_faiss
        if use_faiss:
            try:
                import faiss  # noqa: F401

                backend = FaissVectorIndex(dimension)
                logger.info("Instantiated FaissVectorIndex (dimension=%d)", dimension)
            except ImportError:
                logger.warning("FAISS not installed. Using pure NumpyVectorIndex fallback.")
                backend = NumpyVectorIndex(dimension)
        else:
            backend = NumpyVectorIndex(dimension)

        if len(vectors) > 0:
            backend.add(vectors)

        logger.info(
            "Dense index built with %d vectors (dimension: %d)",
            len(chunk_ids),
            dimension,
        )
        return cls(backend, chunk_ids, doc_ids)

    def save(self, index_dir: Path | str) -> None:
        """Persist index, ID mappings, and metadata to disk."""
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.index, FaissVectorIndex):
            self.index.save(out_dir / "index.faiss")
            backend_type = "faiss"
        else:
            self.index.save(out_dir / "index.npy")
            backend_type = "numpy"

        # Save ID mappings
        mapping_tbl = pa.Table.from_pydict(
            {
                "chunk_id": self.chunk_ids,
                "doc_id": self.doc_ids,
                "index_row": list(range(len(self.chunk_ids))),
            }
        )
        pq.write_table(mapping_tbl, out_dir / "index_id_map.parquet", compression="zstd")

        # Save manifest
        manifest = {
            "backend_type": backend_type,
            "dimension": self.index.dimension,
            "total_vectors": len(self.chunk_ids),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(out_dir / "index_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Saved dense index to %s", out_dir)

    @classmethod
    def load(cls, index_dir: Path | str) -> DenseIndex:
        """Load index and ID mappings from disk."""
        in_dir = Path(index_dir)
        map_path = in_dir / "index_id_map.parquet"
        if not map_path.exists():
            raise FileNotFoundError(f"Index ID map not found: {map_path}")

        tbl = pq.read_table(map_path)
        chunk_ids = tbl.column("chunk_id").to_pylist()
        doc_ids = tbl.column("doc_id").to_pylist()

        faiss_file = in_dir / "index.faiss"
        npy_file = in_dir / "index.npy"

        backend: FaissVectorIndex | NumpyVectorIndex
        if faiss_file.exists():
            try:
                backend = FaissVectorIndex.load(faiss_file)
            except Exception as e:
                logger.warning("Could not load FAISS index (%s). Checking numpy fallback.", e)
                if npy_file.exists():
                    backend = NumpyVectorIndex.load(npy_file)
                else:
                    raise
        elif npy_file.exists():
            backend = NumpyVectorIndex.load(npy_file)
        else:
            raise FileNotFoundError(f"No index binary found in {in_dir}")

        return cls(backend, chunk_ids, doc_ids)

    def search_chunks(self, query_vectors: np.ndarray, top_k_chunks: int) -> list[list[ChunkHit]]:
        """Search top-K chunks for each query vector."""
        scores_mat, indices_mat = self.index.search(query_vectors, top_k_chunks)
        results: list[list[ChunkHit]] = []

        for q_idx in range(len(query_vectors)):
            q_hits: list[ChunkHit] = []
            scores = scores_mat[q_idx]
            indices = indices_mat[q_idx]

            for rank_0, (score, row_idx) in enumerate(zip(scores, indices, strict=False)):
                if row_idx < 0 or row_idx >= len(self.chunk_ids):
                    continue
                q_hits.append(
                    ChunkHit(
                        chunk_id=self.chunk_ids[row_idx],
                        doc_id=self.doc_ids[row_idx],
                        rank=rank_0 + 1,
                        score=float(score),
                    )
                )
            results.append(q_hits)

        return results

    def search_and_aggregate_documents(
        self,
        query_ids: list[int],
        query_vectors: np.ndarray,
        config: DenseRetrievalConfig,
        return_chunk_hits: bool = False,
    ) -> tuple[list[DenseDocHit], DenseRetrievalStats] | tuple[list[DenseDocHit], DenseRetrievalStats, list[list[ChunkHit]]]:
        """Execute dense search at chunk level, aggregate scores to document level (Max similarity),
        and return deduplicated ranked documents per query.
        """
        start_time = time.time()
        logger.info(
            "Executing dense search for %d queries (top %d chunks per query, agg=%s)...",
            len(query_ids),
            config.dense_chunk_top_k,
            config.document_aggregation,
        )

        all_chunk_hits = self.search_chunks(query_vectors, config.dense_chunk_top_k)
        all_doc_hits: list[DenseDocHit] = []

        for q_id, chunk_hits in zip(query_ids, all_chunk_hits, strict=False):
            # Aggregate by doc_id: keep max score and best chunk
            doc_scores: dict[int, float] = defaultdict(lambda: -999999.0)
            doc_best_chunk: dict[int, str] = {}
            doc_best_chunk_score: dict[int, float] = {}

            for ch in chunk_hits:
                if ch.score > doc_scores[ch.doc_id]:
                    doc_scores[ch.doc_id] = ch.score
                    doc_best_chunk[ch.doc_id] = ch.chunk_id
                    doc_best_chunk_score[ch.doc_id] = ch.score

            # Rank unique documents by max score descending
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            if config.top_k_docs:
                sorted_docs = sorted_docs[: config.top_k_docs]

            for rank_0, (doc_id, score) in enumerate(sorted_docs):
                all_doc_hits.append(
                    DenseDocHit(
                        query_id=q_id,
                        doc_id=doc_id,
                        rank=rank_0 + 1,
                        score=score,
                        best_chunk_id=doc_best_chunk[doc_id],
                        best_chunk_score=doc_best_chunk_score[doc_id],
                    )
                )

        search_time = time.time() - start_time
        qps = len(query_ids) / search_time if search_time > 0 else 0.0

        stats = DenseRetrievalStats(
            num_queries=len(query_ids),
            top_k_chunks=config.dense_chunk_top_k,
            search_time_seconds=search_time,
            queries_per_second=qps,
        )

        logger.info(
            "Dense search completed in %.2fs (%.1f queries/s). Total doc hits: %d",
            search_time,
            qps,
            len(all_doc_hits),
        )
        if return_chunk_hits:
            return all_doc_hits, stats, all_chunk_hits
        return all_doc_hits, stats

    @staticmethod
    def export_dense_hits(
        hits: list[DenseDocHit],
        output_file: Path | str,
    ) -> Path:
        """Export dense retrieval results to Parquet."""
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        tbl = pa.Table.from_pydict(
            {
                "query_id": [h.query_id for h in hits],
                "doc_id": [h.doc_id for h in hits],
                "rank": [h.rank for h in hits],
                "score": [h.score for h in hits],
                "best_chunk_id": [h.best_chunk_id for h in hits],
                "best_chunk_score": [h.best_chunk_score for h in hits],
            },
            schema=SCHEMA_DENSE_HITS,
        )
        pq.write_table(tbl, out_p, compression="zstd")
        logger.info("Exported %d dense document hits to %s", len(hits), out_p)
        return out_p

    @staticmethod
    def export_chunk_hits(
        query_ids: list[int],
        all_chunk_hits: list[list[ChunkHit]],
        output_file: Path | str,
    ) -> Path:
        """Export dense chunk-level hits to Parquet for downstream selection."""
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        q_list: list[int] = []
        d_list: list[int] = []
        c_list: list[str] = []
        r_list: list[int] = []
        s_list: list[float] = []

        for q_id, hits in zip(query_ids, all_chunk_hits, strict=False):
            for h in hits:
                q_list.append(int(q_id))
                d_list.append(int(h.doc_id))
                c_list.append(str(h.chunk_id))
                r_list.append(int(h.rank))
                s_list.append(float(h.score))

        tbl = pa.Table.from_pydict(
            {
                "query_id": q_list,
                "doc_id": d_list,
                "chunk_id": c_list,
                "rank": r_list,
                "score": s_list,
            },
            schema=SCHEMA_CHUNK_HITS,
        )
        pq.write_table(tbl, out_p, compression="zstd")
        logger.info("Exported %d chunk hits across %d queries to %s", len(q_list), len(query_ids), out_p)
        return out_p

