from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


class Metrics:
    def __init__(self, logs_dir: Path, interval_seconds: float, corpus_dir: Path) -> None:
        self.started = time.monotonic()
        self.interval_seconds = interval_seconds
        self.processed = 0
        self.bytes_downloaded = 0
        self.text_chars = 0
        self.statuses: Counter[str] = Counter()
        self.http_statuses: Counter[str] = Counter()
        self.domain_errors: Counter[str] = Counter()
        self.domain_latency_total: dict[str, float] = defaultdict(float)
        self.domain_latency_count: Counter[str] = Counter()
        self.corpus_dir = corpus_dir
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = logs_dir / "metrics" / f"run-{run_id}-{os.getpid()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self, *, status: str, domain: str, http_status: int | None, size: int, chars: int
    ) -> None:
        self.processed += 1
        self.bytes_downloaded += size
        self.text_chars += chars
        self.statuses[status] += 1
        if http_status is not None:
            self.http_statuses[str(http_status)] += 1
        if status != "SUCCESS":
            self.domain_errors[domain] += 1

    def record_fetch(self, domain: str, elapsed_seconds: float) -> None:
        self.domain_latency_total[domain] += elapsed_seconds
        self.domain_latency_count[domain] += 1

    def snapshot(self, manifest_counts: dict[str, int]) -> dict[str, object]:
        elapsed = max(0.001, time.monotonic() - self.started)
        slow_domains = sorted(
            (
                (domain, total / self.domain_latency_count[domain])
                for domain, total in self.domain_latency_total.items()
                if self.domain_latency_count[domain]
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:20]
        disk = shutil.disk_usage(self.corpus_dir)
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "processed": self.processed,
            "documents_per_second": round(self.processed / elapsed, 3),
            "mb_per_second": round(self.bytes_downloaded / 1024 / 1024 / elapsed, 3),
            "bytes_downloaded": self.bytes_downloaded,
            "text_chars": self.text_chars,
            "statuses": dict(self.statuses),
            "http_statuses": dict(self.http_statuses),
            "manifest": manifest_counts,
            "top_error_domains": self.domain_errors.most_common(20),
            "top_slow_domains": slow_domains,
            "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total,
        }
        try:
            import psutil

            process = psutil.Process()
            payload["rss_bytes"] = process.memory_info().rss
            payload["cpu_percent"] = process.cpu_percent()
        except ImportError:
            pass
        return payload

    def write(self, manifest_counts: dict[str, int]) -> dict[str, object]:
        snapshot = self.snapshot(manifest_counts)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        return snapshot

    async def periodic(self, manifest, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                self.write(manifest.counts())
        self.write(manifest.counts())
