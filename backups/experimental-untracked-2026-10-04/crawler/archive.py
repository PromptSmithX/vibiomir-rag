from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import shutil
import sqlite3
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from crawler.config import AppConfig
from crawler.extractors import extract_fetch_outcome
from crawler.manifest import Manifest, utc_now
from crawler.models import CrawlStatus, DocumentRecord, FetchOutcome, FetchTarget
from crawler.utils.files import replace_with_retry
from crawler.utils.urls import normalize_fetch_url
from crawler.writer import ParquetShardWriter

LOGGER = logging.getLogger(__name__)
HTML_MIMES = {"text/html", "application/xhtml+xml", "text/plain"}
LIVE_BASELINE_SUCCESS_PER_SECOND = 3.11


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    fetch_key: str
    url: str
    domain: str
    index_name: str
    capture_timestamp: str
    filename: str
    offset: int
    length: int
    mime: str
    status: int


@dataclass(slots=True)
class RangeGroup:
    filename: str
    start: int
    end: int
    records: list[ArchiveRecord]


class ArchiveStore:
    """Resumable Common Crawl index/download state, separate from the corpus manifest."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS archive_targets (
                fetch_key TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                query_prefix TEXT,
                lookup_key TEXT,
                state TEXT NOT NULL DEFAULT 'UNRESOLVED',
                index_name TEXT,
                capture_timestamp TEXT,
                filename TEXT,
                offset INTEGER,
                length INTEGER,
                mime TEXT,
                http_status INTEGER,
                error TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_archive_state
                ON archive_targets(state, domain, fetch_key);
            """
        )
        columns = {
            row[1]
            for row in self.connection.execute("PRAGMA table_info(archive_targets)")
        }
        if "query_prefix" not in columns:
            self.connection.execute(
                "ALTER TABLE archive_targets ADD COLUMN query_prefix TEXT"
            )
        if "lookup_key" not in columns:
            self.connection.execute(
                "ALTER TABLE archive_targets ADD COLUMN lookup_key TEXT"
            )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_prefix_state "
            "ON archive_targets(query_prefix,state,fetch_key)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_lookup_state "
            "ON archive_targets(lookup_key,state)"
        )
        missing = self.connection.execute(
            """
            SELECT fetch_key,url FROM archive_targets
            WHERE query_prefix IS NULL OR lookup_key IS NULL
            """
        ).fetchall()
        if missing:
            self.connection.executemany(
                """
                UPDATE archive_targets SET query_prefix=?,lookup_key=?
                WHERE fetch_key=?
                """,
                [
                    (
                        archive_query_prefix(row["url"]),
                        archive_lookup_key(row["url"]),
                        row["fetch_key"],
                    )
                    for row in missing
                ],
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ArchiveStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def seed(self, rows: list[tuple[str, str, str]]) -> int:
        before = self.connection.total_changes
        with self.transaction():
            self.connection.executemany(
                """
                INSERT OR IGNORE INTO archive_targets(
                    fetch_key,url,domain,query_prefix,lookup_key,state,updated_at
                ) VALUES (?, ?, ?, ?, ?, 'UNRESOLVED', ?)
                """,
                [
                    (
                        key,
                        url,
                        domain,
                        archive_query_prefix(url),
                        archive_lookup_key(url),
                        utc_now(),
                    )
                    for key, url, domain in rows
                ],
            )
        return self.connection.total_changes - before

    def unresolved(self, limit: int | None = None) -> list[sqlite3.Row]:
        sql = (
            "SELECT fetch_key,url,domain FROM archive_targets "
            "WHERE state='UNRESOLVED' ORDER BY domain,fetch_key"
        )
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return list(self.connection.execute(sql, params))

    def set_match(self, fetch_key: str, record: ArchiveRecord | None, error: str | None) -> None:
        if record is None:
            self.connection.execute(
                """
                UPDATE archive_targets SET state=?,error=?,updated_at=?
                WHERE fetch_key=?
                """,
                ("UNRESOLVED" if error else "MISS", error, utc_now(), fetch_key),
            )
            return
        self.connection.execute(
            """
            UPDATE archive_targets SET state='FOUND',index_name=?,capture_timestamp=?,
                filename=?,offset=?,length=?,mime=?,http_status=?,error=NULL,updated_at=?
            WHERE fetch_key=?
            """,
            (
                record.index_name,
                record.capture_timestamp,
                record.filename,
                record.offset,
                record.length,
                record.mime,
                record.status,
                utc_now(),
                fetch_key,
            ),
        )

    def set_matches_for_capture(self, record: ArchiveRecord) -> int:
        before = self.connection.total_changes
        self.connection.execute(
            """
            UPDATE archive_targets SET state='FOUND',index_name=?,capture_timestamp=?,
                filename=?,offset=?,length=?,mime=?,http_status=?,error=NULL,updated_at=?
            WHERE lookup_key=? AND state='UNRESOLVED'
            """,
            (
                record.index_name,
                record.capture_timestamp,
                record.filename,
                record.offset,
                record.length,
                record.mime,
                record.status,
                utc_now(),
                archive_lookup_key(record.url),
            ),
        )
        return self.connection.total_changes - before

    def unresolved_prefixes(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                """
                SELECT DISTINCT query_prefix FROM archive_targets
                WHERE state='UNRESOLVED' AND query_prefix IS NOT NULL
                ORDER BY query_prefix
                """
            )
        ]

    def unresolved_for_prefix(self, prefix: str) -> int:
        return int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM archive_targets
                WHERE query_prefix=? AND state='UNRESOLVED'
                """,
                (prefix,),
            ).fetchone()[0]
        )

    def finish_prefix(self, prefix: str) -> int:
        before = self.connection.total_changes
        self.connection.execute(
            """
            UPDATE archive_targets SET state='MISS',error=NULL,updated_at=?
            WHERE query_prefix=? AND state='UNRESOLVED'
            """,
            (utc_now(), prefix),
        )
        return self.connection.total_changes - before

    def found(self, limit: int) -> list[ArchiveRecord]:
        rows = self.connection.execute(
            """
            SELECT * FROM archive_targets WHERE state='FOUND'
            ORDER BY filename,offset LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_archive_record(row) for row in rows]

    def mark_fetching(self, keys: list[str]) -> None:
        with self.transaction():
            self.connection.executemany(
                "UPDATE archive_targets SET state='FETCHING',updated_at=? WHERE fetch_key=?",
                [(utc_now(), key) for key in keys],
            )

    def mark_fallback(self, key: str, error: str) -> None:
        self.connection.execute(
            """
            UPDATE archive_targets SET state='FALLBACK',error=?,updated_at=?
            WHERE fetch_key=?
            """,
            (error[:2000], utc_now(), key),
        )

    def mark_success(self, keys: list[str]) -> None:
        with self.transaction():
            self.connection.executemany(
                """
                UPDATE archive_targets SET state='SUCCESS',error=NULL,updated_at=?
                WHERE fetch_key=?
                """,
                [(utc_now(), key) for key in set(keys)],
            )

    def mark_skipped(self, key: str) -> None:
        self.connection.execute(
            "UPDATE archive_targets SET state='SKIPPED',updated_at=? WHERE fetch_key=?",
            (utc_now(), key),
        )

    def recover(self, manifest: Manifest) -> dict[str, int]:
        committed = 0
        reset = 0
        rows = self.connection.execute(
            "SELECT fetch_key FROM archive_targets WHERE state='FETCHING'"
        ).fetchall()
        for row in rows:
            status_row = manifest.connection.execute(
                "SELECT status FROM fetch_targets WHERE fetch_key=?", (row["fetch_key"],)
            ).fetchone()
            if status_row and status_row["status"] == CrawlStatus.SUCCESS.value:
                self.mark_success([row["fetch_key"]])
                committed += 1
            else:
                manifest.reset_target(row["fetch_key"])
                self.connection.execute(
                    """
                    UPDATE archive_targets SET state='FOUND',updated_at=? WHERE fetch_key=?
                    """,
                    (utc_now(), row["fetch_key"]),
                )
                reset += 1
        return {"committed": committed, "reset": reset}

    def counts(self) -> dict[str, int]:
        return {
            row["state"]: int(row["count"])
            for row in self.connection.execute(
                "SELECT state,COUNT(*) count FROM archive_targets GROUP BY state"
            )
        }


def _archive_record(row: sqlite3.Row) -> ArchiveRecord:
    return ArchiveRecord(
        fetch_key=row["fetch_key"],
        url=row["url"],
        domain=row["domain"],
        index_name=row["index_name"],
        capture_timestamp=row["capture_timestamp"],
        filename=row["filename"],
        offset=int(row["offset"]),
        length=int(row["length"]),
        mime=row["mime"] or "application/octet-stream",
        status=int(row["http_status"] or 200),
    )


def archive_query_prefix(url: str) -> str:
    parts = urlsplit(normalize_fetch_url(url))
    segments = [segment for segment in parts.path.split("/") if segment]
    path = f"/{segments[0]}/" if len(segments) > 1 else "/"
    return f"{parts.hostname}{path}"


def archive_lookup_key(url: str) -> str:
    parts = urlsplit(normalize_fetch_url(url))
    return urlunsplit(("", parts.netloc, parts.path, parts.query, ""))


def _allocate_probe(counts: dict[str, int], size: int) -> dict[str, int]:
    import math

    size = min(size, sum(counts.values()))
    weights = {domain: math.sqrt(count) for domain, count in counts.items()}
    total_weight = sum(weights.values()) or 1.0
    allocation = {
        domain: min(counts[domain], max(1, int(size * weight / total_weight)))
        for domain, weight in weights.items()
    }
    while sum(allocation.values()) > size:
        domain = max(
            (name for name, value in allocation.items() if value > 1),
            key=lambda name: allocation[name],
        )
        allocation[domain] -= 1
    ordered = sorted(
        counts,
        key=lambda domain: (
            size * weights[domain] / total_weight - allocation[domain],
            domain,
        ),
        reverse=True,
    )
    cursor = 0
    while sum(allocation.values()) < size:
        domain = ordered[cursor % len(ordered)]
        if allocation[domain] < counts[domain]:
            allocation[domain] += 1
        cursor += 1
    return allocation


def select_probe_targets(
    manifest: Manifest, size: int, seed: int
) -> list[tuple[str, str, str]]:
    if size < 1:
        raise ValueError("probe size must be positive")
    counts = {
        row["domain"]: int(row["count"])
        for row in manifest.connection.execute(
            """
            SELECT domain,COUNT(*) count FROM fetch_targets
            WHERE status IN ('PENDING','RETRY_WAIT') GROUP BY domain
            """
        )
    }
    allocation = _allocate_probe(counts, size)
    selected: list[tuple[str, str, str]] = []
    for domain, quota in allocation.items():
        pivot = hashlib.sha256(f"{seed}:{domain}".encode()).hexdigest()
        rows = manifest.connection.execute(
            """
            SELECT fetch_key,request_url,domain FROM fetch_targets
            WHERE domain=? AND status IN ('PENDING','RETRY_WAIT') AND fetch_key>=?
            ORDER BY fetch_key LIMIT ?
            """,
            (domain, pivot, quota),
        ).fetchall()
        if len(rows) < quota:
            rows += manifest.connection.execute(
                """
                SELECT fetch_key,request_url,domain FROM fetch_targets
                WHERE domain=? AND status IN ('PENDING','RETRY_WAIT') AND fetch_key<?
                ORDER BY fetch_key LIMIT ?
                """,
                (domain, pivot, quota - len(rows)),
            ).fetchall()
        selected.extend((row[0], row[1], row[2]) for row in rows)
    return selected


def seed_all_pending(store: ArchiveStore, manifest: Manifest, limit: int | None) -> int:
    sql = (
        "SELECT fetch_key,request_url,domain FROM fetch_targets "
        "WHERE status IN ('PENDING','RETRY_WAIT') ORDER BY domain,fetch_key"
    )
    params: tuple[object, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    inserted = 0
    batch: list[tuple[str, str, str]] = []
    for row in manifest.connection.execute(sql, params):
        batch.append((row[0], row[1], row[2]))
        if len(batch) >= 10_000:
            inserted += store.seed(batch)
            batch.clear()
    if batch:
        inserted += store.seed(batch)
    return inserted


class CommonCrawlClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.session = None

    async def __aenter__(self) -> CommonCrawlClient:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.config.archive.request_timeout_seconds)
        connector = aiohttp.TCPConnector(limit=self.config.archive.download_concurrency)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": self.config.crawler.user_agent},
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.session:
            await self.session.close()

    async def collections(self) -> list[str]:
        if self.session is None:
            raise RuntimeError("CommonCrawlClient is not open")
        url = f"{self.config.archive.index_base_url.rstrip('/')}/collinfo.json"
        async with self.session.get(url) as response:
            response.raise_for_status()
            payload = await response.json(content_type=None)
        values = [str(item.get("id") or item.get("name")) for item in payload]
        return sorted(
            (value for value in values if value and value != "None"), reverse=True
        )

    async def lookup(self, url: str, collections: list[str]) -> ArchiveRecord | None:
        if self.session is None:
            raise RuntimeError("CommonCrawlClient is not open")
        normalized = normalize_fetch_url(url)
        key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        domain = urlsplit(normalized).hostname or ""
        for index_name in collections:
            endpoint = (
                f"{self.config.archive.index_base_url.rstrip('/')}/{index_name}-index"
            )
            params = {
                "url": normalized,
                "output": "json",
                "filter": "status:200",
            }
            for attempt in range(3):
                try:
                    async with self.session.get(endpoint, params=params) as response:
                        if response.status == 404:
                            break
                        if response.status in {429, 500, 502, 503, 504}:
                            await asyncio.sleep(2**attempt)
                            continue
                        response.raise_for_status()
                        text = await response.text()
                    rows = []
                    for line in text.splitlines():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    candidates = []
                    for row in rows:
                        mime = str(row.get("mime-detected") or row.get("mime") or "")
                        if mime not in HTML_MIMES and mime != "application/pdf":
                            continue
                        try:
                            candidates.append(
                                ArchiveRecord(
                                    key,
                                    normalized,
                                    domain,
                                    index_name,
                                    str(row["timestamp"]),
                                    str(row["filename"]),
                                    int(row["offset"]),
                                    int(row["length"]),
                                    mime,
                                    int(row.get("status", 200)),
                                )
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                    if candidates:
                        return max(candidates, key=lambda item: item.capture_timestamp)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)
        return None

    async def fetch_group(self, group: RangeGroup) -> bytes:
        if self.session is None:
            raise RuntimeError("CommonCrawlClient is not open")
        url = f"{self.config.archive.data_base_url.rstrip('/')}/{group.filename}"
        headers = {"Range": f"bytes={group.start}-{group.end - 1}"}
        async with self.session.get(url, headers=headers) as response:
            if response.status != 206:
                raise RuntimeError(f"WARC range returned HTTP {response.status}")
            payload = await response.read()
        expected = group.end - group.start
        if len(payload) != expected:
            raise RuntimeError(f"short WARC range: expected {expected}, got {len(payload)}")
        return payload

    async def scan_prefix(
        self,
        prefix: str,
        index_name: str,
        on_record,
    ) -> int:
        """Stream one hostname/path prefix from a CC index without retaining it."""
        if self.session is None:
            raise RuntimeError("CommonCrawlClient is not open")
        endpoint = f"{self.config.archive.index_base_url.rstrip('/')}/{index_name}-index"
        base_params = {
            "url": f"{prefix}*",
            "output": "json",
            "filter": "status:200",
            "collapse": "urlkey",
        }
        pages = 1
        try:
            async with self.session.get(
                endpoint, params={**base_params, "showNumPages": "true"}
            ) as response:
                if response.status == 404:
                    return 0
                response.raise_for_status()
                page_info = await response.json(content_type=None)
            pages = max(1, int(page_info.get("pages", 1)))
        except (TypeError, ValueError, json.JSONDecodeError):
            pages = 1

        matched = 0
        for page in range(pages):
            params = dict(base_params)
            if pages > 1:
                params["page"] = str(page)
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 404:
                    continue
                response.raise_for_status()
                async for raw_line in response.content:
                    try:
                        row = json.loads(raw_line)
                        normalized = normalize_fetch_url(str(row["url"]))
                        mime = str(row.get("mime-detected") or row.get("mime") or "")
                        if mime not in HTML_MIMES and mime != "application/pdf":
                            continue
                        record = ArchiveRecord(
                            hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                            normalized,
                            urlsplit(normalized).hostname or "",
                            index_name,
                            str(row["timestamp"]),
                            str(row["filename"]),
                            int(row["offset"]),
                            int(row["length"]),
                            mime,
                            int(row.get("status", 200)),
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                    matched += int(on_record(record))
        return matched


async def resolve_archive_targets(
    store: ArchiveStore,
    client: CommonCrawlClient,
    *,
    limit: int | None,
) -> dict[str, int]:
    collections = await client.collections()
    rows = store.unresolved(limit)
    queue: asyncio.Queue[sqlite3.Row | None] = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    for _ in range(client.config.archive.index_concurrency):
        queue.put_nowait(None)
    found = 0
    missed = 0
    failed = 0
    counter_lock = asyncio.Lock()
    updates: list[tuple[str, ArchiveRecord | None, str | None]] = []

    async def worker() -> None:
        nonlocal found, missed, failed
        while True:
            row = await queue.get()
            try:
                if row is None:
                    return
                try:
                    record = await client.lookup(row["url"], collections)
                    updates.append((row["fetch_key"], record, None))
                    async with counter_lock:
                        if record:
                            found += 1
                        else:
                            missed += 1
                except Exception as exc:
                    LOGGER.warning("archive index lookup failed for %s: %s", row["url"], exc)
                    updates.append(
                        (
                            row["fetch_key"],
                            None,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                    async with counter_lock:
                        failed += 1
            finally:
                queue.task_done()

    tasks = [
        asyncio.create_task(worker())
        for _ in range(client.config.archive.index_concurrency)
    ]
    await queue.join()
    await asyncio.gather(*tasks)
    with store.transaction():
        for key, record, error in updates:
            store.set_match(key, record, error)
    return {"checked": len(rows), "found": found, "missed": missed, "failed": failed}


async def resolve_archive_prefixes(
    store: ArchiveStore,
    client: CommonCrawlClient,
) -> dict[str, int]:
    """Resolve the production index in host/path batches rather than per URL."""
    collections = await client.collections()
    prefixes = store.unresolved_prefixes()
    found = 0
    failed_prefixes = 0
    for position, prefix in enumerate(prefixes, start=1):
        try:
            for index_name in collections:
                with store.transaction():
                    found += await client.scan_prefix(
                        prefix, index_name, store.set_matches_for_capture
                    )
                if store.unresolved_for_prefix(prefix) == 0:
                    break
            store.finish_prefix(prefix)
        except Exception as exc:
            failed_prefixes += 1
            LOGGER.warning("archive prefix failed for %s: %s", prefix, exc)
        if position % 10 == 0 or position == len(prefixes):
            LOGGER.info(
                "archive index prefixes %d/%d found=%d",
                position,
                len(prefixes),
                found,
            )
    return {
        "prefixes": len(prefixes),
        "found": found,
        "failed_prefixes": failed_prefixes,
        "missed": store.counts().get("MISS", 0),
    }


def merge_ranges(
    records: list[ArchiveRecord], max_bytes: int, max_gap: int
) -> list[RangeGroup]:
    groups: list[RangeGroup] = []
    by_filename: dict[str, list[ArchiveRecord]] = defaultdict(list)
    for record in records:
        by_filename[record.filename].append(record)
    for filename, items in by_filename.items():
        current: RangeGroup | None = None
        for record in sorted(items, key=lambda item: item.offset):
            record_end = record.offset + record.length
            if (
                current is None
                or record.offset - current.end > max_gap
                or record_end - current.start > max_bytes
            ):
                current = RangeGroup(filename, record.offset, record_end, [record])
                groups.append(current)
            else:
                current.end = max(current.end, record_end)
                current.records.append(record)
    return groups


def _decode_chunked(body: bytes) -> bytes:
    output = bytearray()
    position = 0
    while True:
        line_end = body.find(b"\r\n", position)
        if line_end < 0:
            raise ValueError("invalid chunked payload")
        size = int(body[position:line_end].split(b";", 1)[0], 16)
        position = line_end + 2
        if size == 0:
            return bytes(output)
        output.extend(body[position : position + size])
        position += size + 2


def parse_warc_record(blob: bytes) -> tuple[bytes, str, int]:
    raw = gzip.decompress(blob)
    _, separator, response = raw.partition(b"\r\n\r\n")
    if not separator or not response.startswith(b"HTTP/"):
        raise ValueError("WARC record has no HTTP response")
    raw_headers, separator, body = response.partition(b"\r\n\r\n")
    if not separator:
        raise ValueError("archived HTTP response has no body separator")
    lines = raw_headers.split(b"\r\n")
    try:
        status = int(lines[0].split()[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("invalid archived HTTP status") from exc
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, colon, value = line.partition(b":")
        if colon:
            headers[name.decode("latin-1").lower()] = value.decode("latin-1").strip()
    if "chunked" in headers.get("transfer-encoding", "").lower():
        body = _decode_chunked(body)
    encoding = headers.get("content-encoding", "").lower()
    if encoding == "gzip":
        body = gzip.decompress(body)
    elif encoding == "deflate":
        body = zlib.decompress(body)
    elif encoding and encoding != "identity":
        raise ValueError(f"unsupported archived content encoding: {encoding}")
    content_type = headers.get("content-type", "application/octet-stream").split(";", 1)[0]
    return body, content_type.lower(), status


def _extraction_settings(config: AppConfig) -> dict[str, Any]:
    return config.extraction.model_dump()


class ArchiveRunner:
    def __init__(
        self, config: AppConfig, manifest: Manifest, store: ArchiveStore
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.store = store
        self.processed = 0
        self.success = 0
        self.fallback = 0

    async def run(self, limit: int | None = None) -> dict[str, object]:
        writer = ParquetShardWriter(
            self.config,
            self.manifest,
            on_commit=lambda records: self.store.mark_success(
                [record.fetch_key for record in records]
            ),
        )
        writer_recovery = writer.recover()
        archive_recovery = self.store.recover(self.manifest)
        workers = self.config.crawler.extraction_workers or min(
            2 if __import__("os").name == "nt" else 4,
            max(1, (__import__("os").cpu_count() or 2) // 2),
        )
        pool = ProcessPoolExecutor(max_workers=workers)
        started = time.monotonic()
        try:
            async with CommonCrawlClient(self.config) as client:
                remaining = limit
                while remaining is None or remaining > 0:
                    if shutil.disk_usage(self.config.paths.temp_dir).free < (
                        self.config.archive.min_disk_free_bytes
                    ):
                        raise RuntimeError("archive run stopped: disk free space is below limit")
                    batch_limit = self.config.archive.batch_size
                    if remaining is not None:
                        batch_limit = min(batch_limit, remaining)
                    candidates = self.store.found(batch_limit)
                    if not candidates:
                        break
                    claimed: list[tuple[ArchiveRecord, FetchTarget]] = []
                    for record in candidates:
                        target = self.manifest.claim_specific_target(record.fetch_key)
                        if target is None:
                            self.store.mark_skipped(record.fetch_key)
                            continue
                        claimed.append((record, target))
                    if not claimed:
                        continue
                    self.store.mark_fetching([record.fetch_key for record, _ in claimed])
                    await self._process_claimed(client, writer, pool, claimed)
                    if remaining is not None:
                        remaining -= len(claimed)
            if writer.records:
                writer.flush()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        elapsed = max(0.001, time.monotonic() - started)
        return {
            "processed": self.processed,
            "success": self.success,
            "fallback": self.fallback,
            "elapsed_seconds": round(elapsed, 3),
            "success_per_second": round(self.success / elapsed, 3),
            "archive": self.store.counts(),
            "writer_recovery": writer_recovery,
            "archive_recovery": archive_recovery,
        }

    async def _process_claimed(
        self,
        client: CommonCrawlClient,
        writer: ParquetShardWriter,
        pool: ProcessPoolExecutor,
        claimed: list[tuple[ArchiveRecord, FetchTarget]],
    ) -> None:
        target_by_key = {record.fetch_key: target for record, target in claimed}
        groups = merge_ranges(
            [record for record, _ in claimed],
            self.config.archive.max_merged_range_bytes,
            self.config.archive.merge_gap_bytes,
        )
        semaphore = asyncio.Semaphore(self.config.archive.download_concurrency)

        async def download(group: RangeGroup) -> tuple[RangeGroup, bytes | Exception]:
            async with semaphore:
                try:
                    return group, await client.fetch_group(group)
                except Exception as exc:
                    return group, exc

        results = await asyncio.gather(*(download(group) for group in groups))
        loop = asyncio.get_running_loop()
        extraction_tasks: list[tuple[ArchiveRecord, FetchTarget, FetchOutcome, Any]] = []
        for group, payload in results:
            if isinstance(payload, Exception):
                for record in group.records:
                    self._fallback(record.fetch_key, f"WARC_DOWNLOAD: {payload}")
                continue
            for record in group.records:
                start = record.offset - group.start
                blob = payload[start : start + record.length]
                try:
                    content, content_type, status = parse_warc_record(blob)
                    maximum = (
                        self.config.crawler.max_pdf_bytes
                        if content_type == "application/pdf"
                        else self.config.crawler.max_html_bytes
                    )
                    if len(content) > maximum:
                        raise ValueError(f"archived payload exceeds {maximum} bytes")
                    target = target_by_key[record.fetch_key]
                    outcome = FetchOutcome(
                        target,
                        CrawlStatus.SUCCESS,
                        final_url=record.url,
                        content_type=content_type,
                        content=content,
                        http_status=status,
                        bytes_downloaded=len(content),
                    )
                    future = loop.run_in_executor(
                        pool, extract_fetch_outcome, outcome, _extraction_settings(self.config)
                    )
                    extraction_tasks.append((record, target, outcome, future))
                except Exception as exc:
                    self._fallback(record.fetch_key, f"WARC_PARSE: {type(exc).__name__}: {exc}")

        for record, target, outcome, future in extraction_tasks:
            extracted = await future
            if extracted.status != CrawlStatus.SUCCESS:
                self._fallback(
                    record.fetch_key,
                    f"ARCHIVE_{extracted.status.value}: {extracted.error or ''}",
                )
                continue
            self.processed += 1
            documents = self.manifest.documents_for_target(record.fetch_key)
            retrieved_at = datetime.now(UTC).isoformat()
            output = [
                DocumentRecord(
                    doc_id=document.doc_id,
                    url=document.url,
                    final_url=outcome.final_url,
                    domain=document.domain,
                    content_type=outcome.content_type,
                    title=extracted.title,
                    text=extracted.text,
                    language=extracted.language,
                    content_hash=extracted.content_hash,
                    http_status=outcome.http_status,
                    crawl_status=CrawlStatus.SUCCESS.value,
                    text_chars=len(extracted.text or ""),
                    bytes_downloaded=outcome.bytes_downloaded,
                    fetch_key=target.fetch_key,
                    retrieval_source="common_crawl",
                    retrieved_at=retrieved_at,
                    archive_index=record.index_name,
                    capture_timestamp=record.capture_timestamp,
                    warc_filename=record.filename,
                    warc_offset=record.offset,
                    warc_length=record.length,
                )
                for document in documents
            ]
            writer.records.extend(output)
            writer.buffer_bytes += sum(
                len(item.text.encode("utf-8")) if item.text else 0 for item in output
            )
            self.success += 1
            if (
                len(writer.records) >= self.config.writer.rows_per_shard
                or writer.buffer_bytes >= self.config.writer.max_buffer_bytes
            ):
                writer.flush()

    def _fallback(self, key: str, error: str) -> None:
        self.processed += 1
        self.fallback += 1
        self.store.mark_fallback(key, error)
        self.manifest.reset_target(key)


async def run_probe(
    config: AppConfig,
    manifest: Manifest,
    store: ArchiveStore,
    *,
    size: int,
    seed: int,
    fetch_size: int,
    output: Path,
) -> dict[str, object]:
    selected = select_probe_targets(manifest, size, seed)
    inserted = store.seed(selected)
    started = time.monotonic()
    async with CommonCrawlClient(config) as client:
        resolved = await resolve_archive_targets(store, client, limit=size)
        payload_probe = await probe_archive_payloads(
            config, client, store.found(min(fetch_size, resolved["found"]))
        )
    elapsed = max(0.001, time.monotonic() - started)
    coverage = resolved["found"] / resolved["checked"] if resolved["checked"] else 0.0
    pending_total = int(
        manifest.connection.execute(
            """
            SELECT COUNT(*) FROM fetch_targets
            WHERE status IN ('PENDING','RETRY_WAIT')
            """
        ).fetchone()[0]
    )
    download_rate = payload_probe["download_success_pct"] / 100.0
    extraction_rate = payload_probe["extraction_success_pct"] / 100.0
    projected_archive_successes = pending_total * coverage * download_rate * extraction_rate
    projected_archive_seconds = (
        projected_archive_successes / payload_probe["payload_success_per_second"]
        if payload_probe["payload_success_per_second"]
        else float("inf")
    )
    projected_live_fallback = max(0.0, pending_total - projected_archive_successes)
    projected_hybrid_seconds = (
        projected_archive_seconds
        + projected_live_fallback / LIVE_BASELINE_SUCCESS_PER_SECOND
    )
    projected_live_seconds = pending_total / LIVE_BASELINE_SUCCESS_PER_SECOND
    projected_reduction = (
        1.0 - projected_hybrid_seconds / projected_live_seconds
        if projected_hybrid_seconds != float("inf")
        else 0.0
    )
    report: dict[str, object] = {
        "sample_size": len(selected),
        "new_archive_targets": inserted,
        **resolved,
        "coverage_pct": round(coverage * 100, 3),
        "elapsed_seconds": round(elapsed, 3),
        "index_urls_per_second": round(resolved["checked"] / elapsed, 3),
        "index_gate_passed": coverage >= 0.40,
        "pending_targets": pending_total,
        "projected_archive_successes": round(projected_archive_successes),
        "projected_hybrid_hours": round(projected_hybrid_seconds / 3600.0, 2),
        "projected_live_only_hours": round(projected_live_seconds / 3600.0, 2),
        "projected_wall_time_reduction_pct": round(projected_reduction * 100.0, 2),
        **payload_probe,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    report["probe_gate_passed"] = bool(
        report["index_gate_passed"]
        and payload_probe["download_success_pct"] >= 95.0
        and payload_probe["extraction_success_pct"] >= 85.0
        and payload_probe["payload_success_per_second"]
        >= LIVE_BASELINE_SUCCESS_PER_SECOND * 3
        and projected_reduction >= 0.50
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


async def probe_archive_payloads(
    config: AppConfig,
    client: CommonCrawlClient,
    records: list[ArchiveRecord],
) -> dict[str, object]:
    if not records:
        return {
            "payload_sample_size": 0,
            "download_success_pct": 0.0,
            "extraction_success_pct": 0.0,
            "payload_success_per_second": 0.0,
            "payload_bytes": 0,
        }
    groups = merge_ranges(
        records,
        config.archive.max_merged_range_bytes,
        config.archive.merge_gap_bytes,
    )
    semaphore = asyncio.Semaphore(config.archive.download_concurrency)

    async def download(group: RangeGroup) -> tuple[RangeGroup, bytes | Exception]:
        async with semaphore:
            try:
                return group, await client.fetch_group(group)
            except Exception as exc:
                return group, exc

    started = time.monotonic()
    downloaded = 0
    extracted_success = 0
    payload_bytes = 0
    loop = asyncio.get_running_loop()
    settings = _extraction_settings(config)
    workers = config.crawler.extraction_workers or min(
        2 if __import__("os").name == "nt" else 4,
        max(1, (__import__("os").cpu_count() or 2) // 2),
    )
    pool = ProcessPoolExecutor(max_workers=workers)
    futures = []
    try:
        for group, payload in await asyncio.gather(*(download(group) for group in groups)):
            if isinstance(payload, Exception):
                continue
            for record in group.records:
                start = record.offset - group.start
                try:
                    content, content_type, status = parse_warc_record(
                        payload[start : start + record.length]
                    )
                except Exception:
                    continue
                downloaded += 1
                payload_bytes += len(content)
                target = FetchTarget(record.fetch_key, record.url, record.domain, 0)
                outcome = FetchOutcome(
                    target,
                    CrawlStatus.SUCCESS,
                    final_url=record.url,
                    content_type=content_type,
                    content=content,
                    http_status=status,
                    bytes_downloaded=len(content),
                )
                futures.append(
                    loop.run_in_executor(pool, extract_fetch_outcome, outcome, settings)
                )
        for extracted in await asyncio.gather(*futures):
            extracted_success += int(extracted.status == CrawlStatus.SUCCESS)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    elapsed = max(0.001, time.monotonic() - started)
    return {
        "payload_sample_size": len(records),
        "download_success_pct": round(100.0 * downloaded / len(records), 3),
        "extraction_success_pct": round(
            100.0 * extracted_success / downloaded if downloaded else 0.0, 3
        ),
        "payload_success_per_second": round(extracted_success / elapsed, 3),
        "payload_bytes": payload_bytes,
    }


def export_provenance(manifest: Manifest, output: Path) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("doc_id", pa.int64(), nullable=False),
            pa.field("retrieval_source", pa.string(), nullable=False),
            pa.field("retrieved_at", pa.string()),
            pa.field("archive_index", pa.string()),
            pa.field("capture_timestamp", pa.string()),
            pa.field("warc_filename", pa.string()),
            pa.field("warc_offset", pa.int64()),
            pa.field("warc_length", pa.int64()),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    writer = pq.ParquetWriter(temporary, schema=schema, compression="zstd")
    rows_written = 0
    cursor = manifest.connection.execute(
        "SELECT * FROM document_provenance ORDER BY doc_id"
    )
    try:
        while rows := cursor.fetchmany(50_000):
            values = [dict(row) for row in rows]
            writer.write_table(pa.Table.from_pylist(values, schema=schema))
            rows_written += len(values)
    finally:
        writer.close()
    replace_with_retry(temporary, output)
    return {"output": str(output), "rows": rows_written}
