"""Command Line Interface for ViBioMIR BM25 Retrieval Pipeline."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="ViBioMIR BM25 Retrieval System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: index
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

    # Subcommand: retrieve
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

    args = parser.parse_args()
    if args.command == "index":
        sys.exit(run_index(args))
    elif args.command == "retrieve":
        sys.exit(run_retrieve(args))


if __name__ == "__main__":
    main()
