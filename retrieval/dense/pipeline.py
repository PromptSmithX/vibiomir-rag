"""End-to-end experiment pipeline coordinator for Experiment A."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from retrieval.dense.candidate_loader import (
    CandidateChunkingPipeline,
    load_candidate_doc_ids,
)
from retrieval.dense.chunker import ChunkerRegistry
from retrieval.dense.config import ExperimentConfig
from retrieval.dense.embedder import DenseEmbedder
from retrieval.dense.fusion import ReciprocalRankFusion
from retrieval.dense.index import DenseIndex
from retrieval.dense.model_loader import DenseModelLoader
from retrieval.dense.tokenizer_helper import get_token_counter

logger = logging.getLogger(__name__)


class ExperimentAPipeline:
    """Orchestrates Experiment A: Sentence-aware chunking -> Qwen3 Dense Embedding ->
    Dense Retrieval -> BM25 Fusion.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.exp_dir = Path(config.paths.output_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def run_chunking(
        self,
        corpus_dirs: list[Path | str],
        max_docs: int | None = None,
    ) -> Path:
        """Step 1: Chunk candidate documents."""
        logger.info("=== STEP 1: CHUNKING CANDIDATE DOCUMENTS ===")
        # 1. Resolve token counter
        model_path = DenseModelLoader.resolve_model_path(self.config.model)
        token_counter = get_token_counter(
            model_path,
            offline_mode=self.config.model.offline_mode,
            fallback_if_missing=True,
        )

        # 2. Instantiate chunker
        chunker = ChunkerRegistry.get(self.config.chunking, token_counter=token_counter)

        # 3. Load candidate IDs
        candidate_ids = load_candidate_doc_ids(self.config.paths.candidate_doc_ids)

        # 4. Run chunking pipeline
        pipeline = CandidateChunkingPipeline(chunker=chunker, output_dir=self.exp_dir)
        chunks_dir, stats = pipeline.run(
            candidate_ids=candidate_ids,
            corpus_dirs=corpus_dirs,
            max_docs=max_docs,
        )
        return chunks_dir

    def run_embedding(
        self,
        chunks_dir: Path | str | None = None,
        resume: bool = True,
    ) -> Path:
        """Step 2: Generate dense embeddings for chunks."""
        logger.info("=== STEP 2: DENSE EMBEDDING GENERATION ===")
        if chunks_dir is not None:
            c_dir = Path(chunks_dir)
        elif (self.exp_dir / "chunks").exists():
            c_dir = self.exp_dir / "chunks"
        elif self.exp_dir.name == "embeddings" and (self.exp_dir.parent / "chunks").exists():
            c_dir = self.exp_dir.parent / "chunks"
        elif self.exp_dir.name == "chunks" and self.exp_dir.exists():
            c_dir = self.exp_dir
        else:
            c_dir = self.exp_dir / "chunks"

        if not c_dir.exists():
            raise FileNotFoundError(
                f"Chunks directory not found: {c_dir}. Run chunking first."
            )

        embedder = DenseEmbedder(config=self.config.model)
        if self.exp_dir.name == "embeddings":
            embeddings_dir = self.exp_dir
        else:
            embeddings_dir = self.exp_dir / "embeddings"

        out_dir, stats = embedder.embed_chunks_sharded(
            chunks_dir=c_dir,
            output_dir=embeddings_dir,
            resume=resume,
        )
        return out_dir

    def run_indexing(self) -> Path:
        """Step 3: Build dense vector index."""
        logger.info("=== STEP 3: BUILDING DENSE VECTOR INDEX ===")
        if (self.exp_dir / "embeddings").exists():
            embeddings_dir = self.exp_dir / "embeddings"
        elif self.exp_dir.name in ("index", "embeddings") and (self.exp_dir.parent / "embeddings").exists():
            embeddings_dir = self.exp_dir.parent / "embeddings"
        else:
            embeddings_dir = self.exp_dir / "embeddings"

        index = DenseIndex.build_from_artifacts(
            embeddings_dir=embeddings_dir,
            index_config=self.config.index,
        )
        if self.exp_dir.name in ("index", "embeddings"):
            index_dir = self.exp_dir.parent / "index"
        else:
            index_dir = self.exp_dir / "index"
        index.save(index_dir)
        return index_dir

    def run_retrieval(self, max_queries: int | None = None) -> Path:
        """Step 4: Execute dense retrieval on competition queries."""
        logger.info("=== STEP 4: DENSE RETRIEVAL ON QUERIES ===")
        index_dir = Path(self.config.paths.index_dir) if self.config.paths.index_dir else (self.exp_dir / "index")
        index = DenseIndex.load(index_dir)

        # Load queries
        query_path = Path(self.config.paths.queries)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")

        import pyarrow.parquet as pq

        tbl = pq.read_table(query_path)
        id_col = "id" if "id" in tbl.schema.names else "query_id"
        text_col = "query" if "query" in tbl.schema.names else "query_text"
        query_ids = [int(i) for i in tbl.column(id_col).to_pylist()]
        query_texts = [str(t or "") for t in tbl.column(text_col).to_pylist()]

        if max_queries:
            query_ids = query_ids[:max_queries]
            query_texts = query_texts[:max_queries]
            logger.info("Limiting dense search to %d queries", len(query_ids))

        retrieval_dir = self.exp_dir / "retrieval"
        retrieval_dir.mkdir(parents=True, exist_ok=True)

        # Check for cached query embeddings
        cached_vectors_file = (
            Path(self.config.paths.cached_query_embeddings)
            if self.config.paths.cached_query_embeddings
            else (retrieval_dir / "query_vectors.npy")
        )

        query_vectors: np.ndarray
        if cached_vectors_file.exists():
            logger.info("Loading precomputed query vectors from cache: %s", cached_vectors_file)
            query_vectors = np.load(cached_vectors_file)
            if len(query_vectors) != len(query_texts):
                logger.warning(
                    "Cached query vectors length (%d) mismatch queries count (%d). Re-encoding...",
                    len(query_vectors),
                    len(query_texts),
                )
                embedder = DenseEmbedder(config=self.config.model)
                query_vectors = embedder.encode_queries(query_texts)
                np.save(cached_vectors_file, query_vectors)
        else:
            embedder = DenseEmbedder(config=self.config.model)
            logger.info("Encoding %d queries with %s...", len(query_texts), self.config.model.model_id)
            query_vectors = embedder.encode_queries(query_texts)
            np.save(cached_vectors_file, query_vectors)
            logger.info("Saved query vectors cache to %s", cached_vectors_file)

        # Search and aggregate to document level
        res = index.search_and_aggregate_documents(
            query_ids=query_ids,
            query_vectors=query_vectors,
            config=self.config.retrieval,
            return_chunk_hits=True,
        )
        hits, stats, chunk_hits = res  # type: ignore

        out_file = retrieval_dir / f"dense_retrieval_top_{self.config.retrieval.top_k_docs}.parquet"
        DenseIndex.export_dense_hits(hits, out_file)

        # Export chunk-level hits if enabled
        if getattr(self.config.retrieval, "export_chunk_hits", True):
            chunk_hits_file = retrieval_dir / f"dense_chunk_hits_top_{self.config.retrieval.dense_chunk_top_k}.parquet"
            DenseIndex.export_chunk_hits(query_ids, chunk_hits, chunk_hits_file)

        # Save stats
        stats_data = {
            "num_queries": stats.num_queries,
            "top_k_chunks": stats.top_k_chunks,
            "search_time_seconds": round(stats.search_time_seconds, 2),
            "queries_per_second": round(stats.queries_per_second, 1),
            "total_doc_hits": len(hits),
        }
        with open(retrieval_dir / "dense_retrieval_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)

        return out_file

    def run_fusion(self) -> tuple[Path, Path]:
        """Step 5: Merge BM25 and Dense Retrieval rankings via RRF."""
        logger.info("=== STEP 5: BM25 + DENSE RECIPROCAL RANK FUSION ===")
        bm25_file = Path(self.config.paths.bm25_hits)
        if not bm25_file.exists():
            raise FileNotFoundError(
                f"BM25 hits file not found: {bm25_file}. "
                f"Ensure BM25 retrieval has been executed."
            )

        if self.config.paths.dense_hits and Path(self.config.paths.dense_hits).exists():
            dense_file = Path(self.config.paths.dense_hits)
        else:
            dense_file = (
                self.exp_dir / "retrieval" / f"dense_retrieval_top_{self.config.retrieval.top_k_docs}.parquet"
            )

        if not dense_file.exists():
            raise FileNotFoundError(f"Dense hits file not found: {dense_file}")

        fusion = ReciprocalRankFusion(config=self.config.fusion)
        fused_hits, stats = fusion.fuse(
            bm25_parquet_path=bm25_file,
            dense_parquet_path=dense_file,
        )

        fusion_dir = self.exp_dir / "fusion"
        parquet_path, csv_path = ReciprocalRankFusion.export_fused_results(
            hits=fused_hits,
            output_dir=fusion_dir,
            filename_prefix=f"rrf_fusion_top_{self.config.fusion.top_k}",
        )

        # Save stats
        stats_data = {
            "num_queries": stats.num_queries,
            "rrf_k": stats.rrf_k,
            "total_fused_hits": stats.total_fused_hits,
            "unique_docs": stats.unique_docs,
            "bm25_only_docs": stats.bm25_only_docs,
            "dense_only_docs": stats.dense_only_docs,
            "both_systems_docs": stats.both_systems_docs,
        }
        with open(fusion_dir / "fusion_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)

        return parquet_path, csv_path

    def run_export_submission(
        self,
        retrieval_file: Path | str | None = None,
        top_docs: int | None = None,
        top_chunks: int | None = None,
        max_chunks_per_doc: int | None = None,
        create_zip: bool = True,
    ) -> tuple[Path, Path | None]:
        """Step 6: Export official ViBioMIR submission JSON and ZIP."""
        logger.info("=== STEP 6: EXPORTING OFFICIAL VIBIOMIR SUBMISSION ===")
        from retrieval.submission.exporter import SubmissionExporter

        exporter = SubmissionExporter(
            config=self.config.submission,
            chunk_selection_config=self.config.chunk_selection,
        )

        if retrieval_file:
            r_file = Path(retrieval_file)
        elif self.config.paths.fusion_hits and Path(self.config.paths.fusion_hits).exists():
            r_file = Path(self.config.paths.fusion_hits)
        else:
            fusion_file = self.exp_dir / "fusion" / f"rrf_fusion_top_{self.config.fusion.top_k}.parquet"
            dense_file = self.exp_dir / "retrieval" / f"dense_retrieval_top_{self.config.retrieval.top_k_docs}.parquet"
            if fusion_file.exists():
                r_file = fusion_file
            elif dense_file.exists():
                r_file = dense_file
            else:
                r_file = fusion_file

        chunks_dir = (
            Path(self.config.paths.chunks_dir)
            if self.config.paths.chunks_dir
            else (self.exp_dir / "chunks")
        )
        query_path = Path(self.config.paths.queries)
        sub_dir = Path(self.config.submission.output_dir) if self.config.submission.output_dir else (self.exp_dir / "submission")
        json_out = sub_dir / self.config.submission.output_filename
        zip_out = (sub_dir / self.config.submission.zip_filename) if create_zip else None

        # Check for chunk hits
        chunk_hits_file = self.exp_dir / "retrieval" / f"dense_chunk_hits_top_{self.config.retrieval.dense_chunk_top_k}.parquet"
        ch_path = chunk_hits_file if chunk_hits_file.exists() else None

        summary = exporter.export(
            retrieval_file=r_file,
            query_path=query_path,
            chunks_dir=chunks_dir,
            output_json_path=json_out,
            zip_output_path=zip_out,
            corpus_path=self.config.submission.corpus_path,
            top_docs=top_docs or self.config.submission.top_docs,
            top_chunks=top_chunks or self.config.submission.top_chunks,
            max_chunks_per_doc=max_chunks_per_doc or self.config.chunk_selection.max_chunks_per_document,
            include_chunk_order=self.config.submission.include_chunk_order,
            chunk_hits_file=ch_path,
            metadata_extra={
                "experiment_id": self.config.experiment_id,
                "description": self.config.description,
            },
        )

        return summary.json_path, summary.zip_path


    def run_all(
        self,
        corpus_dirs: list[Path | str],
        max_docs: int | None = None,
        max_queries: int | None = None,
    ) -> None:
        """Run complete Experiment A pipeline end-to-end."""
        start_time = time.time()
        logger.info("Starting complete Experiment A pipeline execution...")

        # Save experiment config snapshot
        self.config.save_yaml(self.exp_dir / "experiment_config.yaml")

        self.run_chunking(corpus_dirs, max_docs=max_docs)
        self.run_embedding(resume=True)
        self.run_indexing()
        self.run_retrieval(max_queries=max_queries)
        parquet_path, csv_path = self.run_fusion()
        json_sub, zip_sub = self.run_export_submission()

        total_elapsed = time.time() - start_time
        logger.info(
            "Experiment A pipeline completed successfully in %.2fs!\n"
            "Final submissions ready:\n"
            "  - Parquet: %s\n"
            "  - CSV:     %s\n"
            "  - JSON:    %s\n"
            "  - ZIP:     %s",
            total_elapsed,
            parquet_path,
            csv_path,
            json_sub,
            zip_sub,
        )
