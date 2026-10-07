from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import FrameType
from typing import Any

from crawler.config import AppConfig
from crawler.extractors import extract_fetch_outcome
from crawler.fetcher import ByteBudget, Fetcher
from crawler.manifest import Manifest
from crawler.metrics import Metrics
from crawler.models import (
    CrawlStatus,
    DocumentBatch,
    DocumentRecord,
    ExtractedContent,
    FetchOutcome,
)
from crawler.progress import CrawlProgress
from crawler.utils.files import replace_with_retry
from crawler.writer import ParquetShardWriter

LOGGER = logging.getLogger(__name__)


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        self.handle.write(b"0")
        self.handle.flush()
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError(f"another crawler process holds {self.path}") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if not self.handle:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


def _extraction_settings(config: AppConfig) -> dict[str, Any]:
    return config.extraction.model_dump()


class CrawlPipeline:
    def __init__(
        self, config: AppConfig, manifest: Manifest, progress_mode: str = "auto"
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.stop_requested = asyncio.Event()
        self.metrics_stop = asyncio.Event()
        self.metrics = Metrics(
            config.paths.logs_dir,
            config.metrics.interval_seconds,
            config.paths.corpus_dir,
        )
        self.byte_budget = ByteBudget(config.queues.response_byte_budget)
        self.fetch_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queues.fetch)
        self.extract_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queues.extract)
        self.write_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queues.write)
        worker_default = max(1, (os.cpu_count() or 2) // 2)
        automatic_worker_cap = 2 if os.name == "nt" else 4
        self.extraction_workers = config.crawler.extraction_workers or min(
            automatic_worker_cap, worker_default
        )
        self.pool = ProcessPoolExecutor(max_workers=self.extraction_workers)
        self.debug_raw_written = 0
        self.progress_mode = progress_mode
        self.progress: CrawlProgress | None = None
        self.in_flight = 0

    def request_stop(self) -> None:
        LOGGER.warning("graceful shutdown requested; no new targets will be scheduled")
        self.stop_requested.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError):
                previous = signal.getsignal(sig)

                def handler(
                    _signum: int,
                    _frame: FrameType | None,
                    *,
                    old=previous,
                ) -> None:
                    loop.call_soon_threadsafe(self.request_stop)
                    if callable(old) and old not in {signal.default_int_handler, handler}:
                        old(_signum, _frame)

                try:
                    signal.signal(sig, handler)
                except ValueError:
                    pass

    async def run(self, target_limit: int | None = None) -> dict[str, int]:
        self._install_signal_handlers()
        writer = ParquetShardWriter(self.config, self.manifest)
        recovery = writer.recover()
        LOGGER.info("startup recovery: %s", recovery)

        unfinished = self.manifest.unfinished_count()
        total = min(unfinished, target_limit) if target_limit is not None else unfinished
        self.progress = CrawlProgress(
            self.progress_mode, total, self.config.crawler.max_attempts
        )
        try:
            with self.progress:
                async with Fetcher(self.config.crawler, self.byte_budget) as fetcher:
                    async with asyncio.TaskGroup() as group:
                        writer_task = group.create_task(
                            writer.run(self.write_queue), name="writer"
                        )
                        group.create_task(
                            self.metrics.periodic(self.manifest, self.metrics_stop),
                            name="metrics",
                        )
                        for index in range(self.config.crawler.global_concurrency):
                            group.create_task(
                                self._fetch_worker(fetcher), name=f"fetch-{index:03d}"
                            )
                        for index in range(self.extraction_workers):
                            group.create_task(
                                self._extract_worker(), name=f"extract-{index:02d}"
                            )
                        group.create_task(
                            self._coordinate(writer_task, target_limit), name="coordinator"
                        )
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
        return self.manifest.counts()

    async def _coordinate(self, writer_task: asyncio.Task, target_limit: int | None) -> None:
        dispatched = 0
        batch_size = max(1, min(self.config.queues.fetch, 256))
        while not self.stop_requested.is_set():
            if target_limit is not None and dispatched >= target_limit:
                break
            remaining = (
                batch_size
                if target_limit is None
                else min(batch_size, target_limit - dispatched)
            )
            targets = self.manifest.claim_targets(remaining)
            if targets:
                self.in_flight += len(targets)
                for target in targets:
                    await self.fetch_queue.put(target)
                dispatched += len(targets)
                await asyncio.sleep(0)
                continue
            if self.in_flight > 0:
                await asyncio.sleep(0.2)
                continue
            delay = self.manifest.next_retry_delay()
            if delay is None:
                break
            try:
                await asyncio.wait_for(
                    self.stop_requested.wait(), timeout=min(max(delay, 0.2), 30.0)
                )
            except TimeoutError:
                pass

        await self.fetch_queue.join()
        for _ in range(self.config.crawler.global_concurrency):
            await self.fetch_queue.put(None)
        await self.fetch_queue.join()

        await self.extract_queue.join()
        for _ in range(self.extraction_workers):
            await self.extract_queue.put(None)
        await self.extract_queue.join()

        if self.progress is not None:
            self.progress.set_phase("flushing")
        await self.write_queue.join()
        await self.write_queue.put(None)
        await writer_task
        if self.progress is not None:
            self.progress.set_phase("done")
        self.metrics_stop.set()

    async def _fetch_worker(self, fetcher: Fetcher) -> None:
        while True:
            target = await self.fetch_queue.get()
            try:
                if target is None:
                    return
                if self.progress is not None:
                    self.progress.target_started(target)
                started = time.monotonic()
                outcome, reserved = await fetcher.fetch(target)
                self.metrics.record_fetch(target.domain, time.monotonic() - started)
                if outcome.retryable and target.attempts < self.config.crawler.max_attempts:
                    if reserved:
                        await self.byte_budget.release(reserved)
                    self.manifest.schedule_retry(
                        target.fetch_key,
                        outcome.retry_at or fetcher.backoff_time(target.attempts),
                        outcome.error_type,
                        outcome.error,
                        outcome.http_status,
                    )
                    if self.progress is not None:
                        self.progress.retry(target, outcome.error_type)
                    self._target_finished()
                    continue
                if outcome.status == CrawlStatus.SUCCESS and outcome.content is not None:
                    await self.extract_queue.put((outcome, reserved))
                else:
                    if reserved:
                        await self.byte_budget.release(reserved)
                    await self._emit_terminal(outcome, None)
            finally:
                self.fetch_queue.task_done()

    async def _extract_worker(self) -> None:
        loop = asyncio.get_running_loop()
        settings = _extraction_settings(self.config)
        while True:
            item = await self.extract_queue.get()
            try:
                if item is None:
                    return
                outcome, reserved = item
                try:
                    extracted = await loop.run_in_executor(
                        self.pool, extract_fetch_outcome, outcome, settings
                    )
                finally:
                    if reserved:
                        await self.byte_budget.release(reserved)
                self._maybe_write_debug_raw(outcome, extracted.status)
                outcome.content = None
                await self._emit_terminal(outcome, extracted)
            finally:
                self.extract_queue.task_done()

    async def _emit_terminal(
        self, outcome: FetchOutcome, extracted: ExtractedContent | None
    ) -> None:
        docs = self.manifest.documents_for_target(outcome.target.fetch_key)
        status = extracted.status if extracted else outcome.status
        error_type = (
            extracted.error_type if extracted and extracted.error_type else outcome.error_type
        )
        error = extracted.error if extracted and extracted.error else outcome.error
        records = [
            DocumentRecord(
                doc_id=doc.doc_id,
                url=doc.url,
                final_url=outcome.final_url,
                domain=doc.domain,
                content_type=outcome.content_type,
                title=extracted.title if extracted else None,
                text=extracted.text if extracted else None,
                language=extracted.language if extracted else None,
                content_hash=extracted.content_hash if extracted else None,
                http_status=outcome.http_status,
                crawl_status=status.value,
                text_chars=len(extracted.text) if extracted and extracted.text else 0,
                bytes_downloaded=outcome.bytes_downloaded,
                fetch_key=outcome.target.fetch_key,
                error_type=error_type,
                error=error,
            )
            for doc in docs
        ]
        for index, record in enumerate(records):
            self.metrics.record(
                status=record.crawl_status,
                domain=record.domain,
                http_status=record.http_status,
                size=record.bytes_downloaded if index == 0 else 0,
                chars=record.text_chars,
            )
        await self.write_queue.put(DocumentBatch(records))
        if self.progress is not None:
            self.progress.terminal(
                outcome.target,
                status,
                outcome.http_status,
                outcome.bytes_downloaded,
            )
        self._target_finished()

    def _target_finished(self) -> None:
        if self.in_flight <= 0:
            raise RuntimeError("pipeline in-flight counter underflow")
        self.in_flight -= 1

    def _maybe_write_debug_raw(self, outcome: FetchOutcome, status: CrawlStatus) -> None:
        limit = self.config.crawler.debug_raw_limit
        if (
            limit <= 0
            or self.debug_raw_written >= limit
            or outcome.content is None
            or status == CrawlStatus.SUCCESS
        ):
            return
        suffix = ".pdf" if outcome.content_type == "application/pdf" else ".html"
        path = self.config.paths.debug_raw_dir / f"{outcome.target.fetch_key}{suffix}"
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(outcome.content)
        replace_with_retry(temporary, path)
        self.debug_raw_written += 1
