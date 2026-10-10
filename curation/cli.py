import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

from curation.models import CurationStats
from curation.pipeline import CurationPipeline
from curation.reader import ParquetCorpusReader
from curation.writer import ParquetCorpusWriter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_clean_worker(args: argparse.Namespace) -> int:
    """Run curation on the corpus of a worker run."""
    run_name = args.run_name
    input_dir = Path(args.input_corpus or f"data/crawl/runs/{run_name}/corpus")
    output_dir = Path(args.output_corpus or f"data/clean/runs/{run_name}/corpus")

    if not input_dir.exists():
        print(f"Error: Input corpus directory '{input_dir}' does not exist.", file=sys.stderr)
        return 1

    reader = ParquetCorpusReader(input_dir, only_success=not args.include_all_status)
    shard_paths = reader.get_shard_paths()
    if args.shards_limit:
        shard_paths = shard_paths[: args.shards_limit]

    if not shard_paths:
        print(f"No parquet shards found in '{input_dir}'.", file=sys.stderr)
        return 1

    pipeline = CurationPipeline(min_chars=args.min_chars)
    writer = ParquetCorpusWriter(output_dir)

    print(f"=== STARTING CURATION: {run_name} ===")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Shards to process: {len(shard_paths)}")
    print(f"Min chars threshold: {args.min_chars}")
    print("=" * 60)

    stats = CurationStats()
    start_time = time.time()

    for idx, shard_path in enumerate(shard_paths, 1):
        raw_docs = reader.read_shard(shard_path)
        curated_docs = pipeline.curate_batch(raw_docs)

        # Update stats
        stats.shards_processed += 1
        stats.docs_read += len(raw_docs)

        clean_docs = [d for d in curated_docs if d.curation_status == "CLEAN"]
        low_quality_docs = [d for d in curated_docs if d.curation_status != "CLEAN"]

        stats.docs_curated += len(clean_docs)
        stats.docs_low_quality += len(low_quality_docs)

        for d in curated_docs:
            stats.total_raw_chars += d.raw_text_chars
            stats.total_clean_chars += d.text_chars
            stats.total_removed_chars += d.removed_chars

            dom = d.domain or "unknown"
            stats.domain_counts[dom] = stats.domain_counts.get(dom, 0) + 1
            stats.domain_removed_chars[dom] = (
                stats.domain_removed_chars.get(dom, 0) + d.removed_chars
            )

        # Write clean docs to output shard
        if clean_docs:
            writer.write_shard(shard_path.name, clean_docs)

        if idx % 5 == 0 or idx == len(shard_paths):
            elapsed = time.time() - start_time
            print(
                f"[{idx}/{len(shard_paths)}] Shards done | "
                f"Docs curated: {stats.docs_curated:,} | "
                f"Noise removed: {stats.total_removed_chars:,} chars | "
                f"Elapsed: {elapsed:.1f}s"
            )

    elapsed_total = time.time() - start_time
    noise_pct = (
        (stats.total_removed_chars / stats.total_raw_chars * 100)
        if stats.total_raw_chars > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("=== CURATION COMPLETED SUCCESSFULLY ===")
    print(f"Total time elapsed   : {elapsed_total:.2f}s")
    print(f"Shards processed     : {stats.shards_processed}")
    print(f"Documents processed  : {stats.docs_read:,}")
    print(f"Clean docs saved     : {stats.docs_curated:,}")
    print(f"Low quality / empty  : {stats.docs_low_quality:,}")
    print(f"Raw text volume      : {stats.total_raw_chars:,} chars")
    print(f"Clean text volume    : {stats.total_clean_chars:,} chars")
    print(f"Noise stripped       : {stats.total_removed_chars:,} chars ({noise_pct:.2f}%)")

    print("\n--- PER-DOMAIN NOISE STRIP SUMMARY ---")
    print(f"{'DOMAIN':<30} | {'DOCS':<9} | {'NOISE REMOVED':<15} | {'AVG REMOVED/DOC'}")
    print("-" * 75)
    for dom in sorted(stats.domain_counts.keys()):
        cnt = stats.domain_counts[dom]
        rem = stats.domain_removed_chars[dom]
        avg = rem / cnt if cnt > 0 else 0
        print(f"{dom:<30} | {cnt:<9,} | {rem:<15,} | {avg:.1f} chars/doc")

    return 0


def run_audit_sample(args: argparse.Namespace) -> int:
    """Sample a few documents per domain and show before/after cleaning."""
    input_dir = Path(args.input_corpus or f"data/crawl/runs/{args.run_name}/corpus")
    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' not found.", file=sys.stderr)
        return 1

    reader = ParquetCorpusReader(input_dir, only_success=True)
    pipeline = CurationPipeline()

    domain_samples = defaultdict(list)
    for _, raw_docs in reader.iter_docs():
        for doc in raw_docs:
            dom = doc.domain or "unknown"
            if len(domain_samples[dom]) < args.samples_per_domain:
                domain_samples[dom].append(doc)
        if (
            all(len(docs) >= args.samples_per_domain for docs in domain_samples.values())
            and len(domain_samples) >= 8
        ):
            break

    print("=== CURATION BEFORE & AFTER AUDIT ===")
    for dom, docs in sorted(domain_samples.items()):
        print("\n" + "=" * 80)
        print(f"DOMAIN: {dom}")
        for idx, doc in enumerate(docs, 1):
            curated = pipeline.curate_doc(doc)
            print(f"\n[Sample {idx}] Doc ID: {doc.doc_id} | Title: {doc.title}")
            print(
                f"Raw chars: {curated.raw_text_chars} -> Clean chars: {curated.text_chars} "
                f"(-{curated.removed_chars} chars)"
            )

            raw_lines = (doc.text or "").strip().splitlines()
            clean_lines = curated.text.strip().splitlines()

            print("RAW TAIL (last 3 lines):")
            for line in raw_lines[-3:]:
                print("   ", line[:100])
            print("CLEAN TAIL (last 3 lines):")
            for line in clean_lines[-3:]:
                print("   ", line[:100])

    return 0


def run_audit_html(args: argparse.Namespace) -> int:
    """Generate a self-contained interactive HTML audit report."""
    from curation.audit.presenter import AuditHtmlPresenter
    from curation.audit.sampler import StratifiedSampler

    run_name = args.run_name
    raw_dir = Path(args.raw_corpus or f"data/crawl/runs/{run_name}/corpus")
    clean_dir = Path(args.clean_corpus or f"data/clean/runs/{run_name}/corpus")
    output_html = Path(args.output_html or f"data/clean/runs/{run_name}/audit_report.html")

    if not clean_dir.exists():
        print(f"Error: Clean corpus dir '{clean_dir}' not found.", file=sys.stderr)
        return 1

    print(f"Sampling ~{args.samples_total} representative documents across domains...")
    sampler = StratifiedSampler(raw_dir, clean_dir, target_total_samples=args.samples_total)
    samples = sampler.sample(max_per_domain=args.max_per_domain)

    if not samples:
        print("No clean samples found.", file=sys.stderr)
        return 1

    print(f"Sampled {len(samples)} documents across {len({s.domain for s in samples})} domains.")
    presenter = AuditHtmlPresenter(samples)
    out_file = presenter.render_to_file(output_html)

    print(f"HTML audit report generated successfully: {out_file.resolve()}")
    print("You can double-click this file to open it in any web browser.")
    return 0


def run_audit_lookup(args: argparse.Namespace) -> int:
    """Look up any document by URL or doc_id directly from binary Parquet."""
    from curation.audit.lookup import ParquetBinaryLookup

    run_name = args.run_name
    raw_dir = Path(args.raw_corpus or f"data/crawl/runs/{run_name}/corpus")
    clean_dir = Path(args.clean_corpus or f"data/clean/runs/{run_name}/corpus")

    lookup = ParquetBinaryLookup(raw_dir, clean_dir)

    result = None
    if args.url:
        print(f"Searching for URL: {args.url}")
        result = lookup.find_by_url(args.url)
    elif args.doc_id:
        print(f"Searching for Doc ID: {args.doc_id}")
        result = lookup.find_by_doc_id(args.doc_id)
    else:
        print("Error: Specify either --url or --doc-id.", file=sys.stderr)
        return 1

    if not result:
        print("Document not found in clean corpus.", file=sys.stderr)
        return 1

    print("=" * 80)
    print(f"DOC ID : {result.doc_id}")
    print(f"DOMAIN : {result.domain}")
    print(f"URL    : {result.url}")
    print(f"TITLE  : {result.title}")
    print(
        f"STATS  : Raw: {result.raw_chars:,} | Clean: {result.clean_chars:,} | "
        f"Removed: -{result.removed_chars:,} chars (-{result.removed_pct}%)"
    )
    print("=" * 80)
    print("--- RAW TEXT TAIL (last 5 lines) ---")
    for line in result.raw_text.strip().splitlines()[-5:]:
        print("  ", line)
    print("\n--- CLEAN TEXT TAIL (last 5 lines) ---")
    for line in result.clean_text.strip().splitlines()[-5:]:
        print("  ", line)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ViBioMIR Corpus Curation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: clean-worker
    clean_parser = subparsers.add_parser("clean-worker", help="Clean worker corpus shards")
    clean_parser.add_argument("--run-name", default="worker-0-local", help="Worker run name")
    clean_parser.add_argument("--input-corpus", help="Path to input corpus dir")
    clean_parser.add_argument("--output-corpus", help="Path to output clean corpus dir")
    clean_parser.add_argument(
        "--shards-limit", type=int, help="Limit number of shards to process"
    )
    clean_parser.add_argument(
        "--min-chars", type=int, default=50, help="Min chars threshold for clean docs"
    )
    clean_parser.add_argument(
        "--include-all-status", action="store_true", help="Include non-SUCCESS docs"
    )

    # Subcommand: audit-sample
    audit_parser = subparsers.add_parser(
        "audit-sample", help="Sample and inspect cleaned documents"
    )
    audit_parser.add_argument("--run-name", default="worker-0-local", help="Worker run name")
    audit_parser.add_argument("--input-corpus", help="Path to input corpus dir")
    audit_parser.add_argument(
        "--samples-per-domain", type=int, default=2, help="Number of samples per domain"
    )

    # Subcommand: audit-html
    html_parser = subparsers.add_parser(
        "audit-html", help="Generate an interactive HTML audit report with ~300 samples"
    )
    html_parser.add_argument("--run-name", default="worker-0-local", help="Worker run name")
    html_parser.add_argument("--raw-corpus", help="Path to raw corpus dir")
    html_parser.add_argument("--clean-corpus", help="Path to clean corpus dir")
    html_parser.add_argument("--output-html", help="Path to output HTML file")
    html_parser.add_argument(
        "--samples-total", type=int, default=300, help="Total samples to include"
    )
    html_parser.add_argument(
        "--max-per-domain", type=int, default=30, help="Max samples per domain"
    )

    # Subcommand: audit-lookup
    lookup_parser = subparsers.add_parser(
        "audit-lookup", help="Look up any document by URL or doc_id from binary Parquet"
    )
    lookup_parser.add_argument("--run-name", default="worker-0-local", help="Worker run name")
    lookup_parser.add_argument("--raw-corpus", help="Path to raw corpus dir")
    lookup_parser.add_argument("--clean-corpus", help="Path to clean corpus dir")
    lookup_parser.add_argument("--url", help="Target URL to inspect")
    lookup_parser.add_argument("--doc-id", type=int, help="Target Doc ID to inspect")

    args = parser.parse_args()
    if args.command == "clean-worker":
        sys.exit(run_clean_worker(args))
    elif args.command == "audit-sample":
        sys.exit(run_audit_sample(args))
    elif args.command == "audit-html":
        sys.exit(run_audit_html(args))
    elif args.command == "audit-lookup":
        sys.exit(run_audit_lookup(args))


if __name__ == "__main__":
    main()
