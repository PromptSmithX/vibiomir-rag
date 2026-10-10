from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from crawler.assignment import validate_worker_partition
from crawler.config import AppConfig, apply_run_name, load_config
from crawler.manifest import Manifest
from crawler.pipeline import CrawlPipeline, RunLock
from crawler.progress import PROGRESS_MODES
from crawler.report import build_report
from crawler.review import create_review_sample
from crawler.source import create_pilot, inspect_source, pilot_ids, sync_source
from crawler.utils.logging import configure_logging
from crawler.writer import ParquetShardWriter, verify_corpus


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _source_path(config: AppConfig, value: str | None) -> Path:
    path = Path(value).resolve() if value else config.paths.source_file
    if not path.exists():
        raise FileNotFoundError(
            f"source file not found: {path}; run `python -m crawler source sync` first"
        )
    return path


def _ensure_source_analysis(config: AppConfig, source_path: Path) -> None:
    analysis_path = config.paths.source_dir / "analysis.json"
    valid = False
    if analysis_path.exists():
        try:
            report = json.loads(analysis_path.read_text(encoding="utf-8"))
            valid = (
                Path(report["source"]).resolve() == source_path.resolve()
                and int(report["source_size_bytes"]) == source_path.stat().st_size
                and int(report["source_mtime_ns"]) == source_path.stat().st_mtime_ns
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            valid = False
    if not valid:
        inspect_source(source_path, analysis_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vibiomir-crawler")
    parser.add_argument("--config", default="config/crawler.yaml")
    parser.add_argument(
        "--run-name",
        help="isolate manifest, corpus, temporary files, and logs under this run name",
    )
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("source", help="download or inspect source data")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    source_sync = source_commands.add_parser("sync")
    source_sync.add_argument("--force", action="store_true")
    source_inspect = source_commands.add_parser("inspect")
    source_inspect.add_argument("--source-path")
    source_inspect.add_argument("--output")

    pilot = commands.add_parser("pilot", help="create a reproducible pilot")
    pilot_commands = pilot.add_subparsers(dest="pilot_command", required=True)
    pilot_create = pilot_commands.add_parser("create")
    pilot_create.add_argument("--source-path")
    pilot_create.add_argument("--size", type=int, default=10_000)
    pilot_create.add_argument("--seed", type=int, default=20261004)
    pilot_create.add_argument("--output")

    manifest = commands.add_parser("manifest", help="manage crawl state")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_init = manifest_commands.add_parser("init")
    manifest_init.add_argument("--source-path")
    manifest_init.add_argument("--selection", help="pilot JSONL; omit to import all rows")
    manifest_init.add_argument(
        "--skip-source-analysis",
        action="store_true",
        help="use a prevalidated partition without replacing the shared source analysis",
    )

    crawl = commands.add_parser("crawl", help="run or inspect the crawler")
    crawl_commands = crawl.add_subparsers(dest="crawl_command", required=True)
    crawl_run = crawl_commands.add_parser("run")
    crawl_run.add_argument("--target-limit", type=int)
    crawl_run.add_argument("--progress", choices=PROGRESS_MODES, default="auto")
    crawl_run.add_argument(
        "--capture-fetch-timings",
        action="store_true",
        help="append per-attempt fetch timings to the run logs directory",
    )
    crawl_run.add_argument("--domain-assignment", type=Path)
    crawl_run.add_argument("--worker-id", type=int)
    crawl_run.add_argument(
        "--defer-domain",
        action="append",
        default=[],
        metavar="DOMAIN",
        help="leave this domain pending during the current run; may be repeated",
    )
    crawl_run.add_argument(
        "--no-recover",
        action="store_true",
        help="skip auto-recovery of adapter-targeted and network-error tasks on startup",
    )
    crawl_commands.add_parser("status")

    corpus = commands.add_parser("corpus", help="verify persistent output")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_commands.add_parser("verify")
    corpus_sample = corpus_commands.add_parser("sample")
    corpus_sample.add_argument("--output")
    corpus_sample.add_argument("--success-size", type=int, default=150)
    corpus_sample.add_argument("--error-size", type=int, default=50)
    corpus_sample.add_argument("--seed", type=int, default=20261004)

    report = commands.add_parser("report", help="build a crawl report")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    report_build = report_commands.add_parser("build")
    report_build.add_argument("--output")
    report_build.add_argument("--review-file")
    return parser


def execute(args: argparse.Namespace, config: AppConfig) -> int:
    if args.command == "source" and args.source_command == "sync":
        config.ensure_directories()
        _print(sync_source(config, force=args.force))
        return 0

    if args.command == "source" and args.source_command == "inspect":
        source_path = _source_path(config, args.source_path)
        output = (
            Path(args.output).resolve()
            if args.output
            else config.paths.source_dir / "analysis.json"
        )
        _print(inspect_source(source_path, output))
        return 0

    if args.command == "pilot" and args.pilot_command == "create":
        source_path = _source_path(config, args.source_path)
        output = (
            Path(args.output).resolve()
            if args.output
            else config.paths.pilots_dir / f"pilot-{args.size}-seed-{args.seed}.jsonl"
        )
        _print(create_pilot(source_path, output, args.size, args.seed))
        return 0

    if args.command == "manifest" and args.manifest_command == "init":
        source_path = _source_path(config, args.source_path)
        if not args.skip_source_analysis:
            _ensure_source_analysis(config, source_path)
        selection = pilot_ids(Path(args.selection).resolve()) if args.selection else None
        config.ensure_directories()
        with Manifest(config.paths.manifest) as manifest:
            _print(manifest.initialize_from_source(source_path, selection))
        return 0

    if args.command == "crawl" and args.crawl_command == "status":
        with Manifest(config.paths.manifest) as manifest:
            manifest.create_schema()
            _print(manifest.counts())
        return 0

    if args.command == "crawl" and args.crawl_command == "run":
        config.ensure_directories()
        deferred_domains = frozenset(
            domain.strip().lower().rstrip(".") for domain in args.defer_domain if domain.strip()
        )
        if len(deferred_domains) != len(args.defer_domain):
            raise ValueError("--defer-domain values must be non-empty and unique")
        if any("://" in domain or "/" in domain for domain in deferred_domains):
            raise ValueError("--defer-domain expects a hostname, not a URL")
        if (args.domain_assignment is None) != (args.worker_id is None):
            raise ValueError("--domain-assignment and --worker-id must be provided together")
        domain_concurrency_overrides: dict[str, int] = {}
        if args.domain_assignment is not None:
            suffix = "local" if args.worker_id == 0 else "kaggle"
            partition_file = (
                args.domain_assignment.resolve().parent
                / f"worker_{args.worker_id}_{suffix}.parquet"
            )
            _rows, domain_concurrency_overrides = validate_worker_partition(
                partition_file,
                args.domain_assignment.resolve(),
                args.worker_id,
            )
        lock_path = config.paths.manifest.with_suffix(".lock")
        with RunLock(lock_path), Manifest(config.paths.manifest) as manifest:
            manifest.create_schema()
            if manifest.counts().get("TOTAL", 0) == 0:
                raise RuntimeError("manifest is empty; run `manifest init` first")
            timing_path = (
                config.paths.logs_dir / "fetch_timings.jsonl"
                if args.capture_fetch_timings
                else None
            )
            counts = asyncio.run(
                CrawlPipeline(
                    config,
                    manifest,
                    args.progress,
                    fetch_timing_path=timing_path,
                    domain_concurrency_overrides=domain_concurrency_overrides,
                    deferred_domains=deferred_domains,
                    recover_adapters=not getattr(args, "no_recover", False),
                ).run(args.target_limit)
            )
            if deferred_domains:
                deferred_counts = manifest.domain_task_counts(deferred_domains)
                _print(
                    {
                        "status_counts": counts,
                        "deferred_domains": sorted(deferred_domains),
                        "deferred_url_count": sum(deferred_counts.values()),
                        "deferred_url_count_by_domain": deferred_counts,
                    }
                )
            else:
                _print(counts)
        return 0

    if args.command == "corpus" and args.corpus_command == "verify":
        with Manifest(config.paths.manifest) as manifest:
            manifest.create_schema()
            recovery = ParquetShardWriter(config, manifest).recover()
            result = verify_corpus(config, manifest)
            result["recovery"] = recovery
            _print(result)
            return 0 if result["ok"] else 2

    if args.command == "corpus" and args.corpus_command == "sample":
        config.ensure_directories()
        output = (
            Path(args.output).resolve()
            if args.output
            else config.paths.logs_dir / "pilot-review.jsonl"
        )
        with Manifest(config.paths.manifest) as manifest:
            manifest.create_schema()
            _print(
                create_review_sample(
                    config,
                    manifest,
                    output,
                    args.success_size,
                    args.error_size,
                    args.seed,
                )
            )
        return 0

    if args.command == "report" and args.report_command == "build":
        config.ensure_directories()
        output = (
            Path(args.output).resolve()
            if args.output
            else config.paths.logs_dir / "pilot-report.json"
        )
        review_file = (
            Path(args.review_file).resolve()
            if args.review_file
            else config.paths.logs_dir / "pilot-review.jsonl"
        )
        with Manifest(config.paths.manifest) as manifest:
            manifest.create_schema()
            _print(
                build_report(
                    config,
                    manifest,
                    output,
                    review_file,
                )
            )
        return 0
    raise RuntimeError("unknown command")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = apply_run_name(load_config(args.config), args.run_name)
        configure_logging(config.paths.logs_dir, args.verbose)
        raise SystemExit(execute(args, config))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
