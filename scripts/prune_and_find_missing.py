"""Pruning and Delta Recovery Generator for ViBioMIR.

Filters existing crawled worker shards against the updated official golden dataset
(1,220,598 URLs), discards outdated/unneeded Chinese articles, preserves valid
crawled documents into a clean pruned corpus, and exports exact missing URLs
into partitioned recovery files ready for Kaggle/Local execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prune_and_find_missing")

CORPUS_SCHEMA = pa.schema(
    [
        ("doc_id", pa.int64(), False),
        ("url", pa.string(), False),
        ("final_url", pa.string(), True),
        ("domain", pa.string(), False),
        ("content_type", pa.string(), True),
        ("title", pa.string(), True),
        ("text", pa.string(), True),
        ("language", pa.string(), True),
        ("content_hash", pa.string(), True),
        ("http_status", pa.int32(), True),
        ("crawl_status", pa.string(), False),
        ("text_chars", pa.int32(), False),
        ("bytes_downloaded", pa.int64(), False),
    ]
)

PARTITION_SCHEMA = pa.schema(
    [
        ("doc_id", pa.int64(), False),
        ("url", pa.string(), False),
        ("domain", pa.string(), False),
    ]
)

DEFAULT_CRAWLED_DIRS = [
    "data/crawl/runs/worker-0-local/corpus",
    "kaggle_outputs/work_1_1k/crawl_worker_1/corpus",
    "kaggle_outputs/work_2/crawl_worker_2/corpus",
    "kaggle_outputs/work_3/crawl_worker_3/corpus",
    "kaggle_outputs/work_4/crawl_worker_4/corpus",
    "kaggle_outputs/work_5/crawl_worker_5/corpus",
]


def domain_from_url(url: str) -> str:
    """Extract lowercase host domain from URL."""
    try:
        host = urlsplit(url).hostname
        return host.lower() if host else "unknown"
    except Exception:
        return "unknown"


def is_vietnamese_url(doc_id: int, url: str) -> bool:
    """Check if URL belongs to Vietnamese medical sources based on ID boundary and domain."""
    if doc_id <= 766441:
        return True
    dom = domain_from_url(url)
    return dom.endswith(".vn") or any(
        k in dom for k in ("hellobacsi.com", "youmed.vn", "thaythuocvietnam.vn")
    )


def is_pubmed_url(url: str) -> bool:
    """Check if URL belongs to PubMed."""
    return "pubmed.ncbi.nlm.nih.gov" in url.lower()


class PruningEngine:
    def __init__(
        self,
        golden_source_path: Path | str,
        crawled_dirs: list[Path | str],
        pruned_corpus_dir: Path | str,
        partitions_dir: Path | str,
        report_path: Path | str,
        rows_per_shard: int = 10000,
        min_chars: int = 50,
        dry_run: bool = False,
    ) -> None:
        self.golden_source_path = Path(golden_source_path)
        self.crawled_dirs = [Path(d) for d in crawled_dirs]
        self.pruned_corpus_dir = Path(pruned_corpus_dir)
        self.partitions_dir = Path(partitions_dir)
        self.report_path = Path(report_path)
        self.rows_per_shard = rows_per_shard
        self.min_chars = min_chars
        self.dry_run = dry_run

    def run(self) -> dict:
        start_time = time.time()
        logger.info("=== STARTING PRUNING & DELTA RECOVERY PIPELINE ===")
        logger.info("Golden source   : %s", self.golden_source_path)
        logger.info("Pruned output   : %s", self.pruned_corpus_dir)
        logger.info("Partitions out  : %s", self.partitions_dir)
        logger.info("Dry-run mode    : %s", self.dry_run)

        # -------------------------------------------------------------
        # Step 1: Load Golden Corpus
        # -------------------------------------------------------------
        if not self.golden_source_path.exists():
            raise FileNotFoundError(f"Golden source not found: {self.golden_source_path}")

        logger.info("Loading golden corpus from %s...", self.golden_source_path.name)
        golden_table = pq.read_table(self.golden_source_path, columns=["id", "url"])
        golden_ids_list = golden_table.column("id").to_pylist()
        golden_urls_list = golden_table.column("url").to_pylist()
        golden_total = len(golden_ids_list)

        golden_map: dict[int, str] = dict(zip(golden_ids_list, golden_urls_list, strict=True))
        golden_id_set: set[int] = set(golden_ids_list)
        logger.info("Loaded %d golden target documents.", golden_total)

        # Categorize golden corpus
        golden_vn_count = 0
        golden_pubmed_count = 0
        golden_zh_count = 0
        for gid, gurl in golden_map.items():
            if is_pubmed_url(gurl):
                golden_pubmed_count += 1
            elif is_vietnamese_url(gid, gurl):
                golden_vn_count += 1
            else:
                golden_zh_count += 1

        logger.info(
            "Golden composition: VN=%d, PubMed=%d, ZH=%d",
            golden_vn_count,
            golden_pubmed_count,
            golden_zh_count,
        )

        # -------------------------------------------------------------
        # Step 2: Stream Crawled Shards & Prune
        # -------------------------------------------------------------
        shard_files: list[Path] = []
        for cdir in self.crawled_dirs:
            if cdir.exists():
                found = sorted(cdir.glob("*.parquet"))
                logger.info("Found %d shards in %s", len(found), cdir)
                shard_files.extend(found)
            else:
                logger.warning("Directory does not exist: %s", cdir)

        logger.info("Total crawled shards to scan: %d", len(shard_files))

        if not self.dry_run:
            self.pruned_corpus_dir.mkdir(parents=True, exist_ok=True)

        seen_doc_ids: set[int] = set()
        discarded_domain_counts: Counter[str] = Counter()
        kept_domain_counts: Counter[str] = Counter()
        crawled_total_rows = 0
        discarded_outdated_rows = 0
        duplicates_skipped = 0
        empty_or_failed_rows = 0

        buffer: dict[str, list] = {col: [] for col in CORPUS_SCHEMA.names}
        shards_written = 0

        def flush_buffer() -> None:
            nonlocal shards_written, buffer
            if not buffer["doc_id"]:
                return
            if not self.dry_run:
                out_table = pa.Table.from_arrays(
                    [pa.array(buffer[col], type=CORPUS_SCHEMA.field(col).type) for col in CORPUS_SCHEMA.names],
                    schema=CORPUS_SCHEMA,
                )
                shard_name = f"part-{shards_written:06d}.parquet"
                shard_path = self.pruned_corpus_dir / shard_name
                pq.write_table(out_table, shard_path, compression="zstd")
            shards_written += 1
            buffer = {col: [] for col in CORPUS_SCHEMA.names}

        logger.info("Scanning and filtering crawled shards...")
        for idx, s_path in enumerate(shard_files, 1):
            try:
                s_table = pq.read_table(s_path)
            except Exception as e:
                logger.warning("Failed to read shard %s: %s", s_path, e)
                continue

            crawled_total_rows += len(s_table)
            dids = s_table.column("doc_id").to_pylist()
            statuses = s_table.column("crawl_status").to_pylist()
            texts = s_table.column("text").to_pylist()
            urls = s_table.column("url").to_pylist()
            domains = (
                s_table.column("domain").to_pylist()
                if "domain" in s_table.column_names
                else [domain_from_url(u) for u in urls]
            )

            final_urls = s_table.column("final_url").to_pylist() if "final_url" in s_table.column_names else urls
            content_types = (
                s_table.column("content_type").to_pylist()
                if "content_type" in s_table.column_names
                else [None] * len(dids)
            )
            titles = s_table.column("title").to_pylist() if "title" in s_table.column_names else [None] * len(dids)
            languages = (
                s_table.column("language").to_pylist() if "language" in s_table.column_names else [None] * len(dids)
            )
            hashes = (
                s_table.column("content_hash").to_pylist()
                if "content_hash" in s_table.column_names
                else [None] * len(dids)
            )
            http_statuses = (
                s_table.column("http_status").to_pylist()
                if "http_status" in s_table.column_names
                else [200] * len(dids)
            )
            text_chars = (
                s_table.column("text_chars").to_pylist()
                if "text_chars" in s_table.column_names
                else [len(t or "") for t in texts]
            )
            bytes_downloads = (
                s_table.column("bytes_downloaded").to_pylist()
                if "bytes_downloaded" in s_table.column_names
                else [0] * len(dids)
            )

            for i in range(len(dids)):
                did = dids[i]
                dom = domains[i] or "unknown"
                status = statuses[i]
                txt = texts[i] or ""

                # Condition 1: Must be in Golden Set
                if did not in golden_id_set:
                    discarded_outdated_rows += 1
                    discarded_domain_counts[dom] += 1
                    continue

                # Condition 2: Check duplication
                if did in seen_doc_ids:
                    duplicates_skipped += 1
                    continue

                # Condition 3: Must be clean success
                if status == "SUCCESS" and len(txt) >= self.min_chars:
                    seen_doc_ids.add(did)
                    kept_domain_counts[dom] += 1

                    buffer["doc_id"].append(did)
                    buffer["url"].append(urls[i])
                    buffer["final_url"].append(final_urls[i])
                    buffer["domain"].append(dom)
                    buffer["content_type"].append(content_types[i])
                    buffer["title"].append(titles[i])
                    buffer["text"].append(txt)
                    buffer["language"].append(languages[i])
                    buffer["content_hash"].append(hashes[i])
                    buffer["http_status"].append(http_statuses[i])
                    buffer["crawl_status"].append(status)
                    buffer["text_chars"].append(text_chars[i])
                    buffer["bytes_downloaded"].append(bytes_downloads[i])

                    if len(buffer["doc_id"]) >= self.rows_per_shard:
                        flush_buffer()
                else:
                    empty_or_failed_rows += 1

            if idx % 50 == 0 or idx == len(shard_files):
                logger.info(
                    "[%d/%d shards] Valid kept: %d | Outdated discarded: %d",
                    idx,
                    len(shard_files),
                    len(seen_doc_ids),
                    discarded_outdated_rows,
                )

        flush_buffer()
        logger.info(
            "Scan complete: %d valid docs preserved across %d pruned shards.",
            len(seen_doc_ids),
            shards_written,
        )

        # -------------------------------------------------------------
        # Step 3: Compute Missing Delta & Partition
        # -------------------------------------------------------------
        missing_ids = golden_id_set - seen_doc_ids
        logger.info("Computing missing documents: %d missing out of %d", len(missing_ids), golden_total)

        missing_vn: list[dict] = []
        missing_pubmed: list[dict] = []
        missing_zh: list[dict] = []

        for mid in sorted(missing_ids):
            murl = golden_map[mid]
            mdom = domain_from_url(murl)
            entry = {"doc_id": mid, "url": murl, "domain": mdom}

            if is_pubmed_url(murl):
                missing_pubmed.append(entry)
            elif is_vietnamese_url(mid, murl):
                missing_vn.append(entry)
            else:
                missing_zh.append(entry)

        logger.info(
            "Missing breakdown: VN=%d, PubMed=%d, ZH=%d",
            len(missing_vn),
            len(missing_pubmed),
            len(missing_zh),
        )

        # -------------------------------------------------------------
        # Step 4: Export Partition Files for Future Crawl
        # -------------------------------------------------------------
        generated_partitions: list[dict] = []

        def save_partition(filename: str, records: list[dict]) -> Path:
            p_path = self.partitions_dir / filename
            if not self.dry_run:
                self.partitions_dir.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_arrays(
                    [
                        pa.array([r["doc_id"] for r in records], type=pa.int64()),
                        pa.array([r["url"] for r in records], type=pa.string()),
                        pa.array([r["domain"] for r in records], type=pa.string()),
                    ],
                    schema=PARTITION_SCHEMA,
                )
                pq.write_table(table, p_path, compression="zstd")
            logger.info("Exported partition %s with %d rows", filename, len(records))
            generated_partitions.append({"filename": filename, "rows": len(records), "path": str(p_path)})
            return p_path

        if missing_vn:
            save_partition("recovery_vn.parquet", missing_vn)

        if missing_pubmed:
            half = len(missing_pubmed) // 2
            save_partition("pubmed_part_1.parquet", missing_pubmed[:half])
            save_partition("pubmed_part_2.parquet", missing_pubmed[half:])

        if missing_zh:
            save_partition("recovery_zh.parquet", missing_zh)

        # -------------------------------------------------------------
        # Step 5: Save Audit Report JSON
        # -------------------------------------------------------------
        elapsed = time.time() - start_time
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(elapsed, 2),
            "dry_run": self.dry_run,
            "golden_total": golden_total,
            "golden_composition": {
                "vietnamese": golden_vn_count,
                "pubmed_english": golden_pubmed_count,
                "chinese": golden_zh_count,
            },
            "crawled_scan": {
                "total_rows_scanned": crawled_total_rows,
                "valid_kept": len(seen_doc_ids),
                "outdated_discarded": discarded_outdated_rows,
                "duplicates_skipped": duplicates_skipped,
                "empty_or_failed": empty_or_failed_rows,
                "shards_written": shards_written,
            },
            "missing_delta": {
                "total_missing": len(missing_ids),
                "vietnamese": len(missing_vn),
                "pubmed_english": len(missing_pubmed),
                "chinese": len(missing_zh),
            },
            "partitions_exported": generated_partitions,
            "top_discarded_domains": discarded_domain_counts.most_common(15),
            "top_kept_domains": kept_domain_counts.most_common(15),
            "top_missing_vn_domains": Counter([r["domain"] for r in missing_vn]).most_common(15),
        }

        if not self.dry_run:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info("Saved report to %s", self.report_path)

        # Print Pretty Console Report
        self._print_report(report)
        return report

    def _print_report(self, report: dict) -> None:
        c = report["crawled_scan"]
        m = report["missing_delta"]
        g = report["golden_total"]

        print("\n" + "=" * 78)
        print("=== VIBIOMIR DATASET PRUNING & DELTA RECOVERY REPORT ===")
        print("=" * 78)
        print(f"Golden Targets Total     : {g:,} documents (links_corpus_vi_en_zh.parquet)")
        print(f"  - Vietnamese           : {report['golden_composition']['vietnamese']:,}")
        print(f"  - PubMed (English)     : {report['golden_composition']['pubmed_english']:,}")
        print(f"  - Chinese              : {report['golden_composition']['chinese']:,}")
        print("-" * 78)
        print(f"Crawled Rows Scanned     : {c['total_rows_scanned']:,}")
        print(f"Valid Preserved Rows     : {c['valid_kept']:,} (saved into {c['shards_written']} shards)")
        print(f"Outdated Rows Discarded  : {c['outdated_discarded']:,} (unneeded Chinese docs)")
        print(f"Duplicate Rows Skipped   : {c['duplicates_skipped']:,}")
        print("-" * 78)
        print(f"TOTAL MISSING TO CRAWL   : {m['total_missing']:,}")
        print(f"  1. Vietnamese Needed   : {m['vietnamese']:,} (mostly Worker 5 + retries)")
        print(f"  2. PubMed Needed       : {m['pubmed_english']:,} (100% new)")
        print(f"  3. Chinese Needed      : {m['chinese']:,}")
        print("-" * 78)
        print("Generated Recovery Partitions:")
        for p in report["partitions_exported"]:
            print(f"  - {p['filename']:<25} : {p['rows']:,} rows")
        print("-" * 78)
        print("Top 5 Missing Vietnamese Domains:")
        for dom, cnt in report["top_missing_vn_domains"][:5]:
            print(f"  - {dom:<30} : {cnt:,} URLs")
        print("=" * 78 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune old crawl shards and export recovery partitions.")
    parser.add_argument(
        "--source-file",
        default="data/source/links_corpus_vi_en_zh.parquet",
        help="Path to new official links corpus parquet file",
    )
    parser.add_argument(
        "--crawled-dirs",
        nargs="+",
        default=DEFAULT_CRAWLED_DIRS,
        help="Corpus shard directories from worker runs",
    )
    parser.add_argument(
        "--pruned-corpus-dir",
        default="data/crawl/pruned_corpus",
        help="Output directory for clean, filtered Parquet shards",
    )
    parser.add_argument(
        "--partitions-dir",
        default="partitions_v3",
        help="Output directory for new recovery partition files",
    )
    parser.add_argument(
        "--report-file",
        default="data/crawl/pruning_report.json",
        help="Output JSON summary report",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=10000,
        help="Number of documents per pruned Parquet shard (default: 10,000)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum extracted character length for clean text (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and compute statistics without writing files to disk",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    engine = PruningEngine(
        golden_source_path=args.source_file,
        crawled_dirs=args.crawled_dirs,
        pruned_corpus_dir=args.pruned_corpus_dir,
        partitions_dir=args.partitions_dir,
        report_path=args.report_file,
        rows_per_shard=args.rows_per_shard,
        min_chars=args.min_chars,
        dry_run=args.dry_run,
    )
    engine.run()


if __name__ == "__main__":
    main()
