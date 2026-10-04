from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

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

    crawl = commands.add_parser("crawl", help="run or inspect the crawler")
    crawl_commands = crawl.add_subparsers(dest="crawl_command", required=True)
    crawl_run = crawl_commands.add_parser("run")
    crawl_run.add_argument("--target-limit", type=int)
    crawl_run.add_argument("--progress", choices=PROGRESS_MODES, default="auto")
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
        lock_path = config.paths.manifest.with_suffix(".lock")
        with RunLock(lock_path), Manifest(config.paths.manifest) as manifest:
            manifest.create_schema()
            if manifest.counts().get("TOTAL", 0) == 0:
                raise RuntimeError("manifest is empty; run `manifest init` first")
            counts = asyncio.run(
                CrawlPipeline(config, manifest, args.progress).run(args.target_limit)
            )
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
