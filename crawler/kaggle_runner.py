from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crawler.assignment import validate_worker_partition
from crawler.config import AppConfig, load_config
from crawler.manifest import Manifest
from crawler.pipeline import CrawlPipeline, RunLock
from crawler.utils.files import replace_with_retry
from crawler.utils.hashing import sha256_file
from crawler.utils.logging import configure_logging
from crawler.writer import ParquetShardWriter, verify_corpus

LOGGER = logging.getLogger(__name__)
CHECKPOINT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def _committed_rows(manifest: Manifest) -> int:
    return int(
        manifest.connection.execute(
            "SELECT COUNT(*) FROM crawl_tasks WHERE output_shard IS NOT NULL"
        ).fetchone()[0]
    )


def _committed_shards(manifest: Manifest) -> int:
    return int(
        manifest.connection.execute(
            "SELECT COUNT(*) FROM shards WHERE status='COMMITTED'"
        ).fetchone()[0]
    )


class CheckpointWriter:
    def __init__(
        self,
        path: Path,
        manifest: Manifest,
        *,
        worker_id: int,
        partition_sha256: str,
        every_docs: int,
        started_monotonic: float,
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.worker_id = worker_id
        self.partition_sha256 = partition_sha256
        self.every_docs = every_docs
        self.started_monotonic = started_monotonic
        self.rows_since_checkpoint = 0

    def payload(self, reason: str) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "worker_id": self.worker_id,
            "partition_sha256": self.partition_sha256,
            "updated_at": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "reason": reason,
            "committed_documents": _committed_rows(self.manifest),
            "committed_shards": _committed_shards(self.manifest),
            "status_counts": self.manifest.counts(),
        }

    def write(self, reason: str) -> dict[str, Any]:
        payload = self.payload(reason)
        _atomic_json(self.path, payload)
        self.rows_since_checkpoint = 0
        return payload

    def on_shard_commit(self, rows: int, _buffer_bytes: int) -> None:
        self.rows_since_checkpoint += rows
        if self.rows_since_checkpoint >= self.every_docs:
            self.write("running")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint must contain a JSON object: {path}")
    return value


def _find_resume_output(resume_root: Path, worker_id: int) -> Path | None:
    if not resume_root.exists():
        return None
    expected_directory = f"crawl_worker_{worker_id}"
    for manifest_path in resume_root.rglob("manifest.sqlite"):
        if (
            manifest_path.parent.name == expected_directory
            and not (manifest_path.parent / "checkpoint.json").is_file()
        ):
            raise RuntimeError(f"resume output is missing checkpoint.json: {manifest_path.parent}")
    matches: list[Path] = []
    for checkpoint_path in resume_root.rglob("checkpoint.json"):
        try:
            checkpoint = _load_checkpoint(checkpoint_path)
            candidate_worker = int(checkpoint.get("worker_id", -1))
        except (RuntimeError, TypeError, ValueError):
            continue
        if candidate_worker != worker_id:
            continue
        candidate = checkpoint_path.parent
        if (candidate / "manifest.sqlite").exists() and (candidate / "corpus").is_dir():
            matches.append(candidate)
    unique = sorted({path.resolve() for path in matches})
    if len(unique) > 1:
        raise RuntimeError(
            f"multiple resume outputs found for worker {worker_id}: {', '.join(map(str, unique))}"
        )
    return unique[0] if unique else None


def _prepare_output(output_dir: Path, resume_root: Path | None, worker_id: int) -> str:
    local_manifest = output_dir / "manifest.sqlite"
    if local_manifest.exists():
        return "working-directory"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory exists without a manifest and is not empty: {output_dir}"
        )
    previous = _find_resume_output(resume_root, worker_id) if resume_root else None
    if previous is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return "fresh"
    shutil.copytree(previous, output_dir, dirs_exist_ok=True)
    return str(previous)


def _configure(
    config_path: Path,
    input_file: Path,
    output_dir: Path,
    *,
    global_concurrency: int,
    per_domain_concurrency: int,
    checkpoint_every_docs: int,
) -> AppConfig:
    config = load_config(config_path)
    config.paths.source_dir = output_dir / "source_metadata"
    config.paths.source_file = input_file
    config.paths.manifest = output_dir / "manifest.sqlite"
    config.paths.corpus_dir = output_dir / "corpus"
    # Keeping the temporary file beside the final shard guarantees an atomic rename.
    config.paths.temp_dir = config.paths.corpus_dir
    config.paths.pilots_dir = output_dir / "internal" / "pilots"
    config.paths.debug_raw_dir = output_dir / "debug_raw"
    config.paths.logs_dir = output_dir
    config.crawler.global_concurrency = global_concurrency
    config.crawler.per_domain_concurrency = per_domain_concurrency
    config.writer.rows_per_shard = checkpoint_every_docs
    config.ensure_directories()
    return config


def _require_kaggle_working_output(output_dir: Path) -> None:
    if not Path("/kaggle/input").is_dir():
        return
    working = Path("/kaggle/working").resolve()
    if output_dir != working and working not in output_dir.parents:
        raise ValueError("Kaggle worker output must be inside /kaggle/working")


def _validate_partition_ownership(input_file: Path, worker_id: int) -> tuple[int, dict[str, int]]:
    assignment_file = input_file.parent / "domain_assignment.parquet"
    if not assignment_file.is_file():
        raise FileNotFoundError(
            f"domain_assignment.parquet must be attached beside {input_file.name}"
        )
    return validate_worker_partition(input_file, assignment_file, worker_id)


def _metadata_value(manifest: Manifest, key: str) -> str | None:
    row = manifest.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return str(row[0]) if row else None


def _set_metadata(manifest: Manifest, key: str, value: str) -> None:
    manifest.connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, value)
    )


def _validate_resume(
    manifest: Manifest,
    checkpoint_path: Path,
    *,
    worker_id: int,
    partition_sha256: str,
) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise RuntimeError("resume output is missing checkpoint.json")
    checkpoint = _load_checkpoint(checkpoint_path)
    if int(checkpoint.get("worker_id", -1)) != worker_id:
        raise RuntimeError("resume checkpoint belongs to a different worker")
    if checkpoint.get("partition_sha256") != partition_sha256:
        raise RuntimeError("resume checkpoint does not match the current worker partition")
    stored_worker = _metadata_value(manifest, "worker_id")
    stored_hash = _metadata_value(manifest, "partition_sha256")
    if stored_worker != str(worker_id) or stored_hash != partition_sha256:
        raise RuntimeError("resume manifest metadata does not match worker/checkpoint")
    return checkpoint


async def _run_with_deadline(
    pipeline: CrawlPipeline,
    max_runtime_seconds: float,
    deadline_reached: asyncio.Event,
    target_limit: int | None = None,
) -> dict[str, int]:
    async def guard() -> None:
        await asyncio.sleep(max_runtime_seconds)
        deadline_reached.set()
        pipeline.request_stop()

    guard_task = asyncio.create_task(guard(), name="runtime-guard")
    try:
        return await pipeline.run(target_limit=target_limit)
    finally:
        guard_task.cancel()
        await asyncio.gather(guard_task, return_exceptions=True)


def run_worker(
    *,
    worker_id: int,
    input_file: Path,
    output_dir: Path,
    config_path: Path,
    resume_root: Path | None,
    max_runtime_hours: float,
    checkpoint_every_docs: int,
    global_concurrency: int,
    per_domain_concurrency: int,
    deadline_epoch: float | None = None,
    target_limit: int | None = None,
) -> dict[str, Any]:
    if worker_id not in range(1, 6):
        raise ValueError("Kaggle worker_id must be between 1 and 5")
    if max_runtime_hours <= 0:
        raise ValueError("max_runtime_hours must be positive")
    if checkpoint_every_docs <= 0:
        raise ValueError("checkpoint_every_docs must be positive")
    if global_concurrency < 1 or per_domain_concurrency < 1:
        raise ValueError("concurrency values must be positive")
    if per_domain_concurrency > global_concurrency:
        raise ValueError("per_domain_concurrency cannot exceed global_concurrency")
    if target_limit is not None and target_limit <= 0:
        raise ValueError("target_limit must be positive when provided")
    input_file = input_file.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    _require_kaggle_working_output(output_dir)
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    expected_name = f"worker_{worker_id}_kaggle.parquet"
    if input_file.name != expected_name:
        raise ValueError(f"worker {worker_id} expects input file {expected_name}")

    started_at = _utc_now()
    started = time.monotonic()
    partition_rows, domain_concurrency_overrides = _validate_partition_ownership(
        input_file, worker_id
    )
    partition_sha256 = sha256_file(input_file)
    resume_source = _prepare_output(output_dir, resume_root, worker_id)
    config = _configure(
        config_path,
        input_file,
        output_dir,
        global_concurrency=global_concurrency,
        per_domain_concurrency=per_domain_concurrency,
        checkpoint_every_docs=checkpoint_every_docs,
    )
    configure_logging(output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "summary.json"
    deadline_reached = asyncio.Event()
    exit_reason = "failed"
    result: dict[str, Any] = {}

    lock_path = config.paths.manifest.with_suffix(".lock")
    with RunLock(lock_path), Manifest(config.paths.manifest) as manifest:
        manifest.create_schema()
        if resume_source == "fresh":
            initialization = manifest.initialize_from_source(input_file)
            _set_metadata(manifest, "worker_id", str(worker_id))
            _set_metadata(manifest, "partition_sha256", partition_sha256)
        else:
            _validate_resume(
                manifest,
                checkpoint_path,
                worker_id=worker_id,
                partition_sha256=partition_sha256,
            )
            recovery = ParquetShardWriter(config, manifest).recover()
            preflight = verify_corpus(config, manifest)
            if not preflight["ok"]:
                raise RuntimeError(f"resume corpus verification failed: {preflight['problems']}")
            initialization = {"resume_recovery": recovery, "resume_preflight": preflight}

        checkpoint_writer = CheckpointWriter(
            checkpoint_path,
            manifest,
            worker_id=worker_id,
            partition_sha256=partition_sha256,
            every_docs=checkpoint_every_docs,
            started_monotonic=started,
        )
        checkpoint_writer.write("starting")
        pipeline = CrawlPipeline(
            config,
            manifest,
            progress_mode="off",
            on_shard_commit=checkpoint_writer.on_shard_commit,
            domain_concurrency_overrides=domain_concurrency_overrides,
        )
        try:
            deadline = deadline_epoch or (time.time() + max_runtime_hours * 3600)
            remaining_seconds = max(0.0, deadline - time.time())
            if remaining_seconds <= 0:
                deadline_reached.set()
                pipeline.request_stop()
            counts = asyncio.run(
                _run_with_deadline(
                    pipeline,
                    remaining_seconds,
                    deadline_reached,
                    target_limit,
                )
            )
            exit_reason = (
                "runtime_limit"
                if deadline_reached.is_set()
                else "completed"
                if manifest.unfinished_count() == 0
                else "target_limit"
                if pipeline.target_limit_reached
                else "stopped"
            )
            verification = verify_corpus(config, manifest)
            if not verification["ok"]:
                raise RuntimeError(f"final corpus verification failed: {verification['problems']}")
            checkpoint = checkpoint_writer.write(exit_reason)
            manifest.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            corpus_bytes = sum(
                path.stat().st_size for path in config.paths.corpus_dir.glob("part-*.parquet")
            )
            result = {
                "version": CHECKPOINT_VERSION,
                "worker_id": worker_id,
                "input_file": str(input_file),
                "partition_sha256": partition_sha256,
                "partition_rows": partition_rows,
                "resume_source": resume_source,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "exit_reason": exit_reason,
                "config": {
                    "max_runtime_hours": max_runtime_hours,
                    "checkpoint_every_docs": checkpoint_every_docs,
                    "global_concurrency": global_concurrency,
                    "per_domain_concurrency": per_domain_concurrency,
                    "domain_concurrency_overrides": domain_concurrency_overrides,
                    "target_limit": target_limit,
                },
                "initialization": initialization,
                "status_counts": counts,
                "checkpoint": checkpoint,
                "verification": verification,
                "corpus_bytes": corpus_bytes,
            }
            _atomic_json(summary_path, result)
        except BaseException as exc:
            LOGGER.exception("Kaggle worker failed")
            try:
                checkpoint_writer.write("failed")
                manifest.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                _atomic_json(
                    summary_path,
                    {
                        "version": CHECKPOINT_VERSION,
                        "worker_id": worker_id,
                        "finished_at": _utc_now(),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "exit_reason": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "status_counts": manifest.counts(),
                    },
                )
            except Exception:
                LOGGER.exception("failed to write failure checkpoint")
            raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ViBioMIR Kaggle crawl worker")
    parser.add_argument("--worker-id", required=True, type=int)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default=Path("config/crawler.yaml"), type=Path)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--max-runtime-hours", default=10.5, type=float)
    parser.add_argument("--checkpoint-every-docs", default=5_000, type=int)
    parser.add_argument("--global-concurrency", default=32, type=int)
    parser.add_argument("--per-domain-concurrency", default=2, type=int)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--target-limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_worker(
        worker_id=args.worker_id,
        input_file=args.input_file,
        output_dir=args.output_dir,
        config_path=args.config,
        resume_root=args.resume_root,
        max_runtime_hours=args.max_runtime_hours,
        checkpoint_every_docs=args.checkpoint_every_docs,
        global_concurrency=args.global_concurrency,
        per_domain_concurrency=args.per_domain_concurrency,
        deadline_epoch=args.deadline_epoch,
        target_limit=args.target_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
