"""Command Line Interface for ViBioMIR BM25 Retrieval Pipeline."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from retrieval.dense.config import ExperimentConfig
from retrieval.dense.pipeline import ExperimentAPipeline
from retrieval.exporter import CandidateExporter
from retrieval.indexer import BM25Indexer
from retrieval.searcher import BM25Searcher
from retrieval.tokenizer import MultilingualMedicalTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrieval.cli")


def discover_default_corpus_dirs() -> list[Path]:
    """Discover all clean corpus directories or fallback to available shards."""
    corpus_dirs: list[Path] = []

    # Check clean runs
    clean_runs = Path("data/clean/runs")
    if clean_runs.exists():
        for d in sorted(clean_runs.iterdir()):
            corpus_path = d / "corpus"
            if corpus_path.exists() and any(corpus_path.glob("*.parquet")):
                corpus_dirs.append(corpus_path)

    # If no clean runs found, check crawl runs
    if not corpus_dirs:
        crawl_w0 = Path("data/crawl/runs/worker-0-local/corpus")
        if crawl_w0.exists():
            corpus_dirs.append(crawl_w0)
        crawl_w3 = Path("kaggle_outputs/work_3/crawl_worker_3/corpus")
        if crawl_w3.exists():
            corpus_dirs.append(crawl_w3)
        crawl_w4 = Path("kaggle_outputs/work_4/crawl_worker_4/corpus")
        if crawl_w4.exists():
            corpus_dirs.append(crawl_w4)

    return corpus_dirs


def run_index(args: argparse.Namespace) -> int:
    """Build or update BM25 Tantivy index."""
    if args.corpus_dirs:
        corpus_dirs = [Path(p) for p in args.corpus_dirs]
    else:
        corpus_dirs = discover_default_corpus_dirs()

    if not corpus_dirs:
        logger.error("No valid corpus directories specified or discovered.")
        return 1

    logger.info("Building BM25 index from %d corpus directories:", len(corpus_dirs))
    for cd in corpus_dirs:
        logger.info("  - %s", cd)

    tokenizer = MultilingualMedicalTokenizer(expand_compounds=not args.no_expand_compounds)
    indexer = BM25Indexer(
        index_dir=args.index_dir,
        tokenizer=tokenizer,
        heap_size_mb=args.heap_size_mb,
        num_threads=args.num_threads,
    )

    stats = indexer.build_index(
        corpus_dirs=corpus_dirs,
        recreate=args.recreate,
        commit_every=args.commit_every,
        only_success=not args.include_all_status,
        max_docs=args.max_docs,
    )

    print("\n" + "=" * 65)
    print("=== BM25 INDEXING SUMMARY ===")
    print(f"Total documents indexed : {stats.total_docs:,}")
    print(f"Total shards processed  : {stats.total_shards}")
    print(f"Index storage on disk   : {stats.index_size_bytes / 1024 / 1024:.2f} MB")
    print(f"Build time elapsed      : {stats.build_time_seconds:.2f}s")
    if stats.build_time_seconds > 0:
        print(f"Indexing throughput     : {stats.total_docs / stats.build_time_seconds:.0f} docs/s")
    print(f"Index location          : {Path(args.index_dir).resolve()}")
    print("=" * 65 + "\n")
    return 0


def run_retrieve(args: argparse.Namespace) -> int:
    """Execute retrieval and export candidates."""
    query_path = Path(args.query_path)
    if not query_path.exists():
        logger.error("Query file does not exist: %s", query_path)
        return 1

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        logger.error("Index directory does not exist: %s", index_dir)
        return 1

    logger.info("Loading queries from: %s", query_path)
    queries = BM25Searcher.load_queries_from_parquet(query_path)
    logger.info("Loaded %d queries from BTC dataset", len(queries))

    if args.max_queries:
        queries = queries[: args.max_queries]
        logger.info("Limiting execution to %d queries", len(queries))

    tokenizer = MultilingualMedicalTokenizer(expand_compounds=not args.no_expand_compounds)
    searcher = BM25Searcher(
        index_dir=index_dir,
        title_boost=args.title_boost,
        tokenizer=tokenizer,
    )

    start_t = time.time()
    hits = searcher.search_queries(queries, k=args.k)
    search_time = time.time() - start_t

    # Exporter
    output_dir = Path(args.output_dir)
    exporter = CandidateExporter(output_dir)

    # 1. Export raw top-K hits
    top_hits_file = exporter.export_hits(hits, filename=f"retrieval_top_{args.k}.parquet")

    # 2. Parse k-levels
    k_levels = [int(x.strip()) for x in args.k_levels.split(",") if x.strip()]
    exporter.export_candidate_levels(hits, k_levels=k_levels)
    stats_list = exporter.compute_level_stats(hits, k_levels=k_levels)

    # 3. Generate Union Top-200 (or configured union_k)
    union_file, union_unique_count = exporter.generate_union_candidates(
        hits, k=args.union_k, filename="candidate_doc_ids.parquet"
    )

    # Save summary stats JSON
    summary_data = {
        "num_queries": len(queries),
        "top_k": args.k,
        "title_boost": args.title_boost,
        "total_hits": len(hits),
        "search_time_seconds": round(search_time, 2),
        "queries_per_second": round(len(queries) / search_time, 1) if search_time > 0 else 0,
        "union_k": args.union_k,
        "union_unique_docs": union_unique_count,
        "levels": [
            {
                "k": s.k,
                "total_candidates": s.total_candidates,
                "unique_docs": s.unique_docs,
                "unique_ratio": round(s.unique_ratio, 4),
                "avg_queries_per_doc": round(s.avg_queries_per_doc, 2),
            }
            for s in stats_list
        ],
    }

    summary_file = output_dir / "retrieval_stats.json"
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary_data, fp, indent=2)

    # Print Clean Formatted Report
    print("\n" + "=" * 78)
    print("=== RETRIEVAL PERFORMANCE & CANDIDATE COVERAGE REPORT ===")
    print(f"Queries evaluated   : {len(queries):,}")
    print(f"Top-K retrieved     : {args.k:,}")
    print(f"Title boost factor  : {args.title_boost}x")
    avg_ms = search_time / len(queries) * 1000
    print(f"Search duration     : {search_time:.2f}s (avg {avg_ms:.2f} ms/query)")
    print(f"Search throughput   : {len(queries) / search_time:.1f} queries/s")
    print("-" * 78)
    print(f"{'Cutoff (K)':<12} | {'Total Candidates':<18} | {'Unique Docs (Union)':<20} | Dedup")
    print("-" * 78)
    for s in stats_list:
        ratio_pct = f"{s.unique_ratio * 100:.2f}%"
        print(f"Top {s.k:<8} | {s.total_candidates:<18,} | {s.unique_docs:<20,} | {ratio_pct}")
    print("-" * 78)
    print(f"Target Union (K={args.union_k}) : {union_unique_count:,} unique candidate docs saved")
    print(f"Union file path     : {union_file.resolve()}")
    print(f"Top-{args.k} file path   : {top_hits_file.resolve()}")
    print(f"Metrics JSON        : {summary_file.resolve()}")
    print("=" * 78 + "\n")

    return 0


def _resolve_exp_config(args: argparse.Namespace) -> ExperimentConfig:
    """Resolve and optionally override experiment configuration from CLI flags."""
    config_file = getattr(args, "config", None)
    if config_file and Path(config_file).exists():
        cfg = ExperimentConfig.load_yaml(config_file)
    elif Path("config/experiment_a.yaml").exists():
        cfg = ExperimentConfig.load_yaml("config/experiment_a.yaml")
    else:
        cfg = ExperimentConfig()

    # CLI Overrides
    if getattr(args, "output_dir", None):
        cfg.paths.output_dir = args.output_dir
    if getattr(args, "candidate_file", None):
        cfg.paths.candidate_doc_ids = args.candidate_file
    if getattr(args, "bm25_file", None):
        cfg.paths.bm25_hits = args.bm25_file
    if getattr(args, "query_path", None):
        cfg.paths.queries = args.query_path
    if getattr(args, "device", None):
        cfg.model.device = args.device
    if getattr(args, "precision", None):
        cfg.model.precision = args.precision
    if getattr(args, "batch_size", None):
        cfg.model.batch_size = args.batch_size
    if getattr(args, "offline", False):
        cfg.model.offline_mode = True
    if getattr(args, "local_model_path", None):
        cfg.model.local_model_path = args.local_model_path
    if getattr(args, "rrf_k", None):
        cfg.fusion.rrf_k = args.rrf_k

    return cfg


def run_chunk(args: argparse.Namespace) -> int:
    """Chunk candidate documents into sentence-aware units."""
    if args.corpus_dirs:
        corpus_dirs = [Path(p) for p in args.corpus_dirs]
    else:
        corpus_dirs = discover_default_corpus_dirs()

    if not corpus_dirs:
        logger.error("No valid corpus directories specified or discovered.")
        return 1

    cfg = _resolve_exp_config(args)
    pipeline = ExperimentAPipeline(cfg)
    chunks_dir = pipeline.run_chunking(corpus_dirs=corpus_dirs, max_docs=args.max_docs)
    print(f"\nChunking complete! Chunks saved to: {chunks_dir.resolve()}\n")
    return 0


def run_embed(args: argparse.Namespace) -> int:
    """Embed chunks using Qwen3-Embedding model."""
    cfg = _resolve_exp_config(args)
    pipeline = ExperimentAPipeline(cfg)
    chunks_dir = getattr(args, "chunks_dir", None)
    emb_dir = pipeline.run_embedding(chunks_dir=chunks_dir, resume=not args.no_resume)
    print(f"\nEmbedding complete! Vectors saved to: {emb_dir.resolve()}\n")
    return 0


def run_build_dense_index(args: argparse.Namespace) -> int:
    """Build dense vector index from embeddings."""
    cfg = _resolve_exp_config(args)
    pipeline = ExperimentAPipeline(cfg)
    idx_dir = pipeline.run_indexing()
    print(f"\nDense index built successfully! Saved to: {idx_dir.resolve()}\n")
    return 0


def run_dense_search(args: argparse.Namespace) -> int:
    """Search dense index and aggregate chunk similarities to document level."""
    cfg = _resolve_exp_config(args)
    pipeline = ExperimentAPipeline(cfg)
    res_file = pipeline.run_retrieval(max_queries=args.max_queries)
    print(f"\nDense retrieval complete! Results saved to: {res_file.resolve()}\n")
    return 0


def run_fuse(args: argparse.Namespace) -> int:
    """Execute Reciprocal Rank Fusion on BM25 and Dense Retrieval rankings."""
    cfg = _resolve_exp_config(args)
    pipeline = ExperimentAPipeline(cfg)
    pq_file, csv_file = pipeline.run_fusion()
    print(f"\nFusion complete!\n  - Parquet: {pq_file.resolve()}\n  - CSV:     {csv_file.resolve()}\n")
    return 0


def run_exp_a(args: argparse.Namespace) -> int:
    """Run full Experiment A pipeline end-to-end."""
    if args.corpus_dirs:
        corpus_dirs = [Path(p) for p in args.corpus_dirs]
    else:
        corpus_dirs = discover_default_corpus_dirs()

    if not corpus_dirs:
        logger.error("No valid corpus directories specified or discovered.")
        return 1

    cfg = _resolve_exp_config(args)
    pipeline = ExperimentAPipeline(cfg)
    pipeline.run_all(
        corpus_dirs=corpus_dirs,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
    )
    return 0


def run_export_submission(args: argparse.Namespace) -> int:
    """Export retrieval/fusion results to official ViBioMIR submission JSON/ZIP."""
    from retrieval.submission.exporter import SubmissionExporter

    cfg = _resolve_exp_config(args)
    exporter = SubmissionExporter(config=cfg.submission)

    exp_dir = Path(cfg.paths.output_dir)
    # Determine retrieval file
    if getattr(args, "retrieval_file", None):
        r_file = Path(args.retrieval_file)
    else:
        fusion_file = exp_dir / "fusion" / f"rrf_fusion_top_{cfg.fusion.top_k}.parquet"
        dense_file = exp_dir / "retrieval" / f"dense_retrieval_top_{cfg.retrieval.top_k_docs}.parquet"
        if fusion_file.exists():
            r_file = fusion_file
        elif dense_file.exists():
            r_file = dense_file
        else:
            r_file = fusion_file

    query_path = Path(args.query_path) if getattr(args, "query_path", None) else Path(cfg.paths.queries)
    chunks_dir = Path(args.chunks_dir) if getattr(args, "chunks_dir", None) else (exp_dir / "chunks")

    sub_dir = Path(cfg.submission.output_dir)
    out_json = Path(args.output) if getattr(args, "output", None) else (sub_dir / cfg.submission.output_filename)

    zip_path = None
    if getattr(args, "zip", None):
        if isinstance(args.zip, str) and args.zip != "true":
            zip_path = Path(args.zip)
        else:
            zip_path = sub_dir / cfg.submission.zip_filename

    summary = exporter.export(
        retrieval_file=r_file,
        query_path=query_path,
        chunks_dir=chunks_dir,
        output_json_path=out_json,
        zip_output_path=zip_path,
        corpus_path=getattr(args, "corpus_path", None) or cfg.submission.corpus_path,
        top_docs=getattr(args, "top_docs", None) or cfg.submission.top_docs,
        top_chunks=getattr(args, "top_chunks", None) or cfg.submission.top_chunks,
        include_chunk_order=(
            args.include_chunk_order if getattr(args, "include_chunk_order", False) else cfg.submission.include_chunk_order
        ),
        doc_id_type=getattr(args, "doc_id_type", None) or getattr(cfg.submission, "doc_id_type", "int"),
    )

    print("\n" + "=" * 65)
    print("=== VIBIOMIR SUBMISSION EXPORT COMPLETED ===")
    print(f"Total official queries : {summary.total_official_queries}")
    print(f"Queries with results   : {summary.total_queries_with_results}")
    print(f"Empty queries          : {summary.total_empty_queries}")
    print(f"Total relevant docs    : {summary.total_relevant_docs:,}")
    print(f"Total relevant chunks  : {summary.total_relevant_chunks:,}")
    print(f"Missing chunks count   : {summary.missing_chunks_count}")
    print(f"JSON submission file   : {summary.json_path}")
    if summary.zip_path:
        print(f"ZIP package file       : {summary.zip_path}")
    print("=" * 65 + "\n")
    return 0


def run_validate_submission(args: argparse.Namespace) -> int:
    """Validate submission JSON or ZIP against official ViBioMIR criteria."""
    from retrieval.submission.validator import SubmissionValidator

    cfg = _resolve_exp_config(args)
    q_path = getattr(args, "query_path", None) or cfg.paths.queries
    c_path = getattr(args, "corpus_path", None) or cfg.submission.corpus_path

    validator = SubmissionValidator(query_path=q_path, corpus_path=c_path)
    res = validator.validate_file(args.submission)

    print("\n" + res.to_report() + "\n")
    return 0 if res.is_valid else 1


def run_inspect_query(args: argparse.Namespace) -> int:
    """Inspect query texts and retrieval results comparison."""
    from scripts.inspect_comparison import inspect_queries

    cfg = _resolve_exp_config(args)
    exp_dir = Path(cfg.paths.output_dir)

    query_path = getattr(args, "query_path", None) or cfg.paths.queries
    fusion_path = getattr(args, "fusion_path", None) or str(
        exp_dir / "fusion" / f"rrf_fusion_top_{cfg.fusion.top_k}.parquet"
    )
    dense_path = getattr(args, "dense_path", None) or str(
        exp_dir / "retrieval" / f"dense_retrieval_top_{cfg.retrieval.top_k_docs}.parquet"
    )
    bm25_path = getattr(args, "bm25_path", None) or cfg.paths.bm25_hits
    chunks_dir = getattr(args, "chunks_dir", None) or str(exp_dir / "chunks")
    submission_path = getattr(args, "submission_path", None) or str(
        Path(cfg.submission.output_dir) / cfg.submission.output_filename
    )

    inspect_queries(
        query_ids=args.query_ids,
        query_path=query_path,
        fusion_path=fusion_path,
        dense_path=dense_path,
        bm25_path=bm25_path,
        chunks_dir=chunks_dir,
        submission_path=submission_path,
        top_n=args.top_n,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ViBioMIR BM25 & Dense Retrieval System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: index (BM25)
    p_idx = subparsers.add_parser("index", help="Build or rebuild Tantivy BM25 index")
    p_idx.add_argument(
        "--corpus-dirs",
        nargs="+",
        help="One or more directories containing parquet shards",
    )
    p_idx.add_argument(
        "--index-dir",
        default="data/retrieval/bm25_index",
        help="Target index storage directory",
    )
    p_idx.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate index from scratch if directory exists",
    )
    p_idx.add_argument(
        "--heap-size-mb",
        type=int,
        default=512,
        help="RAM buffer size for Tantivy writer in MB",
    )
    p_idx.add_argument(
        "--commit-every",
        type=int,
        default=100_000,
        help="Flush index segment every N documents",
    )
    p_idx.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Writer threads (0 for auto)",
    )
    p_idx.add_argument(
        "--max-docs",
        type=int,
        help="Optional upper limit on documents for smoke tests",
    )
    p_idx.add_argument(
        "--include-all-status",
        action="store_true",
        help="Include non-SUCCESS docs if reading raw crawl shards",
    )
    p_idx.add_argument(
        "--no-expand-compounds",
        action="store_true",
        help="Disable sub-term expansion for hyphenated/dotted terms",
    )

    # Subcommand: retrieve (BM25)
    p_ret = subparsers.add_parser("retrieve", help="Run BM25 retrieval against BTC queries")
    p_ret.add_argument(
        "--index-dir",
        default="data/retrieval/bm25_index",
        help="Path to Tantivy index directory",
    )
    p_ret.add_argument(
        "--query-path",
        default="data/source/query.parquet",
        help="Path to official query.parquet",
    )
    p_ret.add_argument(
        "--output-dir",
        default="data/retrieval/candidates",
        help="Directory to save output candidates and metrics",
    )
    p_ret.add_argument(
        "-k",
        "--k",
        type=int,
        default=1000,
        help="Top-K documents retrieved per query",
    )
    p_ret.add_argument(
        "--title-boost",
        type=float,
        default=2.0,
        help="BM25 boost factor for document titles",
    )
    p_ret.add_argument(
        "--union-k",
        type=int,
        default=200,
        help="Cutoff K for generating union candidate file",
    )
    p_ret.add_argument(
        "--k-levels",
        default="100,200,300,500,1000",
        help="Comma-separated K levels for candidate subsets and coverage stats",
    )
    p_ret.add_argument(
        "--max-queries",
        type=int,
        help="Optional upper limit on queries for smoke tests",
    )
    p_ret.add_argument(
        "--no-expand-compounds",
        action="store_true",
        help="Disable sub-term expansion for queries",
    )

    # Subcommand: chunk (Experiment A)
    p_chunk = subparsers.add_parser("chunk", help="Chunk candidate documents into sentence-aware units")
    p_chunk.add_argument("--config", help="Path to experiment YAML config")
    p_chunk.add_argument("--corpus-dirs", nargs="+", help="Corpus shard directories")
    p_chunk.add_argument("--candidate-file", help="Path to candidate_doc_ids.parquet")
    p_chunk.add_argument("--output-dir", help="Output directory for chunks")
    p_chunk.add_argument("--max-docs", type=int, help="Limit number of documents (e.g. 100 for smoke test)")
    p_chunk.add_argument("--offline", action="store_true", help="Enforce offline tokenizer loading")

    # Subcommand: embed
    p_embed = subparsers.add_parser("embed", help="Generate dense embeddings for chunks")
    p_embed.add_argument("--config", help="Path to experiment YAML config")
    p_embed.add_argument("--output-dir", help="Output directory for embeddings")
    p_embed.add_argument("--batch-size", type=int, help="Batch size for embedding")
    p_embed.add_argument("--device", choices=["cuda", "cpu", "auto"], help="Target compute device")
    p_embed.add_argument("--precision", choices=["float16", "bfloat16", "float32"], help="Floating point precision")
    p_embed.add_argument("--offline", action="store_true", help="Enforce offline model loading")
    p_embed.add_argument("--local-model-path", help="Local directory path containing model weights")
    p_embed.add_argument("--chunks-dir", help="Explicit path to directory containing chunk parquet files")
    p_embed.add_argument("--no-resume", action="store_true", help="Re-compute all chunks without resuming")

    # Subcommand: build-dense-index
    p_bidx = subparsers.add_parser("build-dense-index", help="Build dense vector index from embeddings")
    p_bidx.add_argument("--config", help="Path to experiment YAML config")
    p_bidx.add_argument("--output-dir", help="Output directory")

    # Subcommand: dense-search
    p_dsearch = subparsers.add_parser("dense-search", help="Execute dense retrieval and score aggregation")
    p_dsearch.add_argument("--config", help="Path to experiment YAML config")
    p_dsearch.add_argument("--query-path", help="Path to query.parquet")
    p_dsearch.add_argument("--max-queries", type=int, help="Limit number of queries (e.g. 10 for smoke test)")
    p_dsearch.add_argument("--device", choices=["cuda", "cpu", "auto"], help="Target compute device")
    p_dsearch.add_argument("--offline", action="store_true", help="Enforce offline model loading")
    p_dsearch.add_argument("--local-model-path", help="Local directory path containing model weights")
    p_dsearch.add_argument("--output-dir", help="Output directory")

    # Subcommand: fuse
    p_fuse = subparsers.add_parser("fuse", help="Execute Reciprocal Rank Fusion on BM25 and Dense rankings")
    p_fuse.add_argument("--config", help="Path to experiment YAML config")
    p_fuse.add_argument("--bm25-file", help="Path to BM25 retrieval hits parquet")
    p_fuse.add_argument("--rrf-k", type=int, help="RRF smoothing constant (default: 60)")
    p_fuse.add_argument("--output-dir", help="Output directory")

    # Subcommand: run-exp-a (End-to-End)
    p_expa = subparsers.add_parser("run-exp-a", help="Run full Experiment A pipeline end-to-end")
    p_expa.add_argument("--config", help="Path to experiment YAML config")
    p_expa.add_argument("--corpus-dirs", nargs="+", help="Corpus shard directories")
    p_expa.add_argument("--output-dir", help="Output directory for experiment artifacts")
    p_expa.add_argument("--max-docs", type=int, help="Limit number of docs for testing")
    p_expa.add_argument("--max-queries", type=int, help="Limit number of queries for testing")
    p_expa.add_argument("--device", choices=["cuda", "cpu", "auto"], help="Target compute device")
    p_expa.add_argument("--precision", choices=["float16", "bfloat16", "float32"], help="Precision")
    p_expa.add_argument("--batch-size", type=int, help="Batch size for embedding")
    p_expa.add_argument("--offline", action="store_true", help="Enforce offline model loading")
    p_expa.add_argument("--local-model-path", help="Local directory path containing model weights")

    # Subcommand: export-submission
    p_esub = subparsers.add_parser("export-submission", help="Export official ViBioMIR submission JSON/ZIP")
    p_esub.add_argument("--config", help="Path to experiment YAML config")
    p_esub.add_argument("--retrieval-file", help="Path to rrf_fusion_top_*.parquet or dense hits parquet")
    p_esub.add_argument("--query-path", help="Path to official query.parquet")
    p_esub.add_argument("--chunks-dir", help="Path to chunks directory")
    p_esub.add_argument("--corpus-path", help="Optional corpus parquet path for verification/fallback")
    p_esub.add_argument("--output", help="Output path for submission.json")
    p_esub.add_argument("--zip", nargs="?", const="true", help="Package into submission.zip (optional custom path)")
    p_esub.add_argument("--top-docs", type=int, help="Limit relevant documents per query")
    p_esub.add_argument("--top-chunks", type=int, help="Limit relevant chunks per query")
    p_esub.add_argument("--include-chunk-order", action="store_true", help="Include optional chunk_order field")
    p_esub.add_argument(
        "--doc-id-type",
        choices=["int", "str"],
        help="Data type for doc_id in relevant_docs and relevant_chunks ('int' or 'str', default from config)",
    )

    # Subcommand: validate-submission
    p_vsub = subparsers.add_parser("validate-submission", help="Validate official ViBioMIR submission JSON/ZIP")
    p_vsub.add_argument("--submission", required=True, help="Path to submission.json or submission.zip")
    p_vsub.add_argument("--query-path", help="Path to official query.parquet")
    p_vsub.add_argument("--corpus-path", help="Optional corpus parquet path for doc_id verification")
    p_vsub.add_argument("--config", help="Path to experiment YAML config for default paths")

    # Subcommand: inspect-query
    p_ins = subparsers.add_parser("inspect-query", help="Inspect query texts and compare retrieval results")
    p_ins.add_argument("--query-ids", nargs="+", type=int, default=[1, 4, 7, 250], help="Query IDs to inspect")
    p_ins.add_argument("--top-n", type=int, default=10, help="Number of results per query")
    p_ins.add_argument("--config", help="Path to experiment YAML config")
    p_ins.add_argument("--query-path", help="Path to query.parquet")

    args = parser.parse_args()
    if args.command == "index":
        sys.exit(run_index(args))
    elif args.command == "retrieve":
        sys.exit(run_retrieve(args))
    elif args.command == "chunk":
        sys.exit(run_chunk(args))
    elif args.command == "embed":
        sys.exit(run_embed(args))
    elif args.command == "build-dense-index":
        sys.exit(run_build_dense_index(args))
    elif args.command == "dense-search":
        sys.exit(run_dense_search(args))
    elif args.command == "fuse":
        sys.exit(run_fuse(args))
    elif args.command == "export-submission":
        sys.exit(run_export_submission(args))
    elif args.command == "validate-submission":
        sys.exit(run_validate_submission(args))
    elif args.command == "inspect-query":
        sys.exit(run_inspect_query(args))
    elif args.command == "run-exp-a":
        sys.exit(run_exp_a(args))


if __name__ == "__main__":
    main()
