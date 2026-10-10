from __future__ import annotations

import sqlite3
import time
from collections import deque
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crawler.models import DocumentRecord, FetchTarget, SourceDocument
from crawler.source import iter_source_rows
from crawler.utils.hashing import sha256_text
from crawler.utils.urls import domain_from_url, fetch_key, normalize_fetch_url


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self._claim_domains: deque[str] | None = None
        self._claim_excluded_domains: frozenset[str] | None = None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Manifest:
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

    def create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fetch_targets (
                fetch_key TEXT PRIMARY KEY,
                request_url TEXT NOT NULL,
                domain TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                attempts INTEGER NOT NULL DEFAULT 0,
                next_retry_at REAL,
                http_status INTEGER,
                final_url TEXT,
                content_type TEXT,
                error_type TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS crawl_tasks (
                doc_id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                normalized_url TEXT NOT NULL,
                fetch_key TEXT NOT NULL REFERENCES fetch_targets(fetch_key),
                domain TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                attempts INTEGER NOT NULL DEFAULT 0,
                http_status INTEGER,
                final_url TEXT,
                content_type TEXT,
                bytes_downloaded INTEGER,
                text_chars INTEGER,
                output_shard TEXT,
                error_type TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_fetch_status_retry
                ON fetch_targets(status, next_retry_at, domain);
            CREATE INDEX IF NOT EXISTS idx_fetch_domain_status_retry
                ON fetch_targets(domain, status, next_retry_at, fetch_key);
            CREATE INDEX IF NOT EXISTS idx_tasks_fetch_key ON crawl_tasks(fetch_key);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON crawl_tasks(status);

            CREATE TABLE IF NOT EXISTS shards (
                shard_name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                temp_path TEXT NOT NULL,
                final_path TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                committed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS shard_items (
                shard_name TEXT NOT NULL REFERENCES shards(shard_name) ON DELETE CASCADE,
                doc_id INTEGER NOT NULL REFERENCES crawl_tasks(doc_id),
                fetch_key TEXT NOT NULL,
                crawl_status TEXT NOT NULL,
                http_status INTEGER,
                final_url TEXT,
                content_type TEXT,
                bytes_downloaded INTEGER NOT NULL,
                text_chars INTEGER NOT NULL,
                error_type TEXT,
                error TEXT,
                PRIMARY KEY(shard_name, doc_id)
            );
            """
        )

    def initialize_from_source(
        self, source_path: Path, selected_ids: set[int] | None = None
    ) -> dict[str, int]:
        self.create_schema()
        inserted = 0
        existing = 0
        invalid = 0
        matched_selected = 0
        batch: list[tuple[int, str, str, str, str]] = []

        def flush() -> None:
            nonlocal inserted, existing
            if not batch:
                return
            with self.transaction():
                for doc_id, url, normalized, key, domain in batch:
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO fetch_targets(
                            fetch_key, request_url, domain, status, updated_at
                        ) VALUES (?, ?, ?, 'PENDING', ?)
                        """,
                        (key, normalized, domain, utc_now()),
                    )
                    current = self.connection.execute(
                        "SELECT url FROM crawl_tasks WHERE doc_id=?", (doc_id,)
                    ).fetchone()
                    if current:
                        if current["url"] != url:
                            raise ValueError(
                                f"doc_id {doc_id} already exists with a different URL"
                            )
                        existing += 1
                        continue
                    self.connection.execute(
                        """
                        INSERT INTO crawl_tasks(
                            doc_id, url, normalized_url, fetch_key, domain, status, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'PENDING', ?)
                        """,
                        (doc_id, url, normalized, key, domain, utc_now()),
                    )
                    inserted += 1
            batch.clear()

        for doc_id, url in iter_source_rows(source_path):
            if selected_ids is not None and doc_id not in selected_ids:
                continue
            if selected_ids is not None:
                matched_selected += 1
            try:
                normalized = normalize_fetch_url(url)
                key = fetch_key(normalized)
                domain = domain_from_url(normalized)
            except ValueError:
                invalid += 1
                normalized = url.strip()
                key = sha256_text(f"invalid-url:{normalized}")
                domain = ""
            batch.append(
                (doc_id, url, normalized, key, domain)
            )
            if selected_ids is None and len(batch) >= 10_000:
                flush()
        if selected_ids is not None and matched_selected != len(selected_ids):
            raise ValueError(
                f"selection contains {len(selected_ids)} IDs but only "
                f"{matched_selected} exist in source"
            )
        flush()
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('source_path', ?)",
            (str(source_path),),
        )
        self._claim_domains = None
        return {"inserted": inserted, "existing": existing, "invalid": invalid}

    def recover_unstaged_fetches(self) -> int:
        with self.transaction():
            keys = [
                row[0]
                for row in self.connection.execute(
                    """
                    SELECT f.fetch_key FROM fetch_targets f
                    WHERE f.status='FETCHING'
                      AND NOT EXISTS (
                          SELECT 1 FROM shard_items i WHERE i.fetch_key=f.fetch_key
                      )
                    """
                )
            ]
            if keys:
                self.connection.executemany(
                    "UPDATE fetch_targets SET status='PENDING', updated_at=? WHERE fetch_key=?",
                    [(utc_now(), key) for key in keys],
                )
                self.connection.executemany(
                    "UPDATE crawl_tasks SET status='PENDING', updated_at=? WHERE fetch_key=?",
                    [(utc_now(), key) for key in keys],
                )
            return len(keys)

    def recover_adapter_targets(self, adapters: dict[str, Any]) -> dict[str, int]:
        recovered_counts: dict[str, int] = {}
        now = utc_now()
        with self.transaction():
            for domain, adapter in adapters.items():
                can_recover = getattr(adapter, "can_recover_task", None)
                if not callable(can_recover):
                    continue
                rows = self.connection.execute(
                    """
                    SELECT fetch_key, status, http_status, bytes_downloaded, error_type
                    FROM crawl_tasks
                    WHERE domain = ? AND status != 'SUCCESS'
                    """,
                    (domain,),
                ).fetchall()
                matching_keys = [
                    row[0]
                    for row in rows
                    if can_recover(row[1], row[2], row[3], row[4])
                ]
                if matching_keys:
                    self.connection.executemany(
                        """
                        UPDATE crawl_tasks
                        SET status='PENDING', attempts=0, error_type=NULL, error=NULL, updated_at=?
                        WHERE fetch_key=?
                        """,
                        [(now, key) for key in matching_keys],
                    )
                    self.connection.executemany(
                        """
                        UPDATE fetch_targets
                        SET status='PENDING', attempts=0, error_type=NULL, error=NULL, updated_at=?
                        WHERE fetch_key=?
                        """,
                        [(now, key) for key in matching_keys],
                    )
                    recovered_counts[domain] = len(matching_keys)
        return recovered_counts

    def recover_transient_network_errors(self) -> int:
        now = utc_now()
        with self.transaction():
            keys = [
                row[0]
                for row in self.connection.execute(
                    """
                    SELECT fetch_key FROM crawl_tasks
                    WHERE status = 'FAILED' AND error_type = 'NETWORK_ERROR'
                    """
                ).fetchall()
            ]
            if keys:
                self.connection.executemany(
                    """
                    UPDATE crawl_tasks
                    SET status='PENDING', attempts=0, error_type=NULL, error=NULL, updated_at=?
                    WHERE fetch_key=?
                    """,
                    [(now, key) for key in keys],
                )
                self.connection.executemany(
                    """
                    UPDATE fetch_targets
                    SET status='PENDING', attempts=0, error_type=NULL, error=NULL, updated_at=?
                    WHERE fetch_key=?
                    """,
                    [(now, key) for key in keys],
                )
            return len(keys)

    def claim_targets(
        self, limit: int, excluded_domains: frozenset[str] | None = None
    ) -> list[FetchTarget]:
        if limit <= 0:
            return []
        excluded = excluded_domains or frozenset()
        now = time.time()
        with self.transaction():
            if self._claim_domains is None or self._claim_excluded_domains != excluded:
                self._claim_domains = deque(
                    row[0]
                    for row in self.connection.execute(
                        "SELECT DISTINCT domain FROM fetch_targets ORDER BY domain"
                    )
                    if row[0] not in excluded
                )
                self._claim_excluded_domains = excluded
            if not self._claim_domains:
                return []

            domains = list(self._claim_domains)
            selected: list[sqlite3.Row] = []
            candidates = domains
            remaining = limit
            while remaining > 0 and candidates:
                quota = max(1, (remaining + len(candidates) - 1) // len(candidates))
                buckets: list[list[sqlite3.Row]] = []
                may_have_more: list[str] = []
                for domain in candidates:
                    retry_rows = self.connection.execute(
                        """
                        SELECT fetch_key,request_url,domain,attempts
                        FROM fetch_targets INDEXED BY idx_fetch_domain_status_retry
                        WHERE domain=? AND status='RETRY_WAIT' AND next_retry_at<=?
                        ORDER BY next_retry_at,fetch_key LIMIT ?
                        """,
                        (domain, now, quota),
                    ).fetchall()
                    rows = list(retry_rows)
                    if len(rows) < quota:
                        rows.extend(
                            self.connection.execute(
                                """
                                SELECT fetch_key,request_url,domain,attempts
                                FROM fetch_targets
                                INDEXED BY idx_fetch_domain_status_retry
                                WHERE domain=? AND status='PENDING'
                                ORDER BY next_retry_at,fetch_key LIMIT ?
                                """,
                                (domain, quota - len(rows)),
                            ).fetchall()
                        )
                    buckets.append(rows)
                    if len(rows) == quota:
                        may_have_more.append(domain)

                round_rows: list[sqlite3.Row] = []
                for rank in range(quota):
                    for rows in buckets:
                        if rank < len(rows):
                            round_rows.append(rows[rank])
                            if len(round_rows) == remaining:
                                break
                    if len(round_rows) == remaining:
                        break
                if not round_rows:
                    break

                updated_at = utc_now()
                for row in round_rows:
                    attempts = int(row["attempts"]) + 1
                    self.connection.execute(
                        """
                        UPDATE fetch_targets
                        SET status='FETCHING', attempts=?, next_retry_at=NULL, updated_at=?
                        WHERE fetch_key=?
                        """,
                        (attempts, updated_at, row["fetch_key"]),
                    )
                    self.connection.execute(
                        """
                        UPDATE crawl_tasks SET status='FETCHING', attempts=?, updated_at=?
                        WHERE fetch_key=?
                        """,
                        (attempts, updated_at, row["fetch_key"]),
                    )
                selected.extend(round_rows)
                remaining -= len(round_rows)
                candidates = may_have_more

            self._claim_domains.rotate(-1)
            targets: list[FetchTarget] = []
            for row in selected:
                targets.append(
                    FetchTarget(
                        fetch_key=row["fetch_key"],
                        request_url=row["request_url"],
                        domain=row["domain"],
                        attempts=int(row["attempts"]) + 1,
                    )
                )
            return targets

    def schedule_retry(
        self,
        fetch_key_value: str,
        retry_at: float,
        error_type: str | None,
        error: str | None,
        http_status: int | None,
    ) -> None:
        with self.transaction():
            self.connection.execute(
                """
                UPDATE fetch_targets SET status='RETRY_WAIT', next_retry_at=?, error_type=?,
                    error=?, http_status=?, updated_at=? WHERE fetch_key=?
                """,
                (retry_at, error_type, error, http_status, utc_now(), fetch_key_value),
            )
            self.connection.execute(
                """
                UPDATE crawl_tasks SET status='RETRY_WAIT', error_type=?, error=?,
                    http_status=?, updated_at=? WHERE fetch_key=?
                """,
                (error_type, error, http_status, utc_now(), fetch_key_value),
            )

    def documents_for_target(self, fetch_key_value: str) -> list[SourceDocument]:
        return [
            SourceDocument(row["doc_id"], row["url"], row["domain"])
            for row in self.connection.execute(
                "SELECT doc_id, url, domain FROM crawl_tasks WHERE fetch_key=? ORDER BY doc_id",
                (fetch_key_value,),
            )
        ]

    def next_shard_name(self) -> str:
        row = self.connection.execute(
            "SELECT shard_name FROM shards ORDER BY shard_name DESC LIMIT 1"
        ).fetchone()
        number = int(row[0].split("-")[1].split(".")[0]) + 1 if row else 0
        return f"part-{number:06d}.parquet"

    def stage_shard(
        self, shard_name: str, temp_path: Path, final_path: Path, records: Sequence[DocumentRecord]
    ) -> None:
        with self.transaction():
            self.connection.execute(
                """
                INSERT INTO shards(shard_name, status, temp_path, final_path, row_count, created_at)
                VALUES (?, 'WRITING', ?, ?, ?, ?)
                """,
                (shard_name, str(temp_path), str(final_path), len(records), utc_now()),
            )
            self.connection.executemany(
                """
                INSERT INTO shard_items(
                    shard_name, doc_id, fetch_key, crawl_status, http_status, final_url,
                    content_type, bytes_downloaded, text_chars, error_type, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        shard_name,
                        record.doc_id,
                        record.fetch_key,
                        record.crawl_status,
                        record.http_status,
                        record.final_url,
                        record.content_type,
                        record.bytes_downloaded,
                        record.text_chars,
                        record.error_type,
                        record.error,
                    )
                    for record in records
                ],
            )

    def commit_staged_shard(self, shard_name: str) -> None:
        with self.transaction():
            items = self.connection.execute(
                "SELECT * FROM shard_items WHERE shard_name=? ORDER BY doc_id", (shard_name,)
            ).fetchall()
            if not items:
                raise RuntimeError(f"no staged items for {shard_name}")
            self.connection.executemany(
                """
                UPDATE crawl_tasks SET status=?, http_status=?, final_url=?, content_type=?,
                    bytes_downloaded=?, text_chars=?, output_shard=?, error_type=?, error=?,
                    updated_at=? WHERE doc_id=?
                """,
                [
                    (
                        row["crawl_status"],
                        row["http_status"],
                        row["final_url"],
                        row["content_type"],
                        row["bytes_downloaded"],
                        row["text_chars"],
                        shard_name,
                        row["error_type"],
                        row["error"],
                        utc_now(),
                        row["doc_id"],
                    )
                    for row in items
                ],
            )
            by_fetch: dict[str, str] = {}
            for row in items:
                by_fetch[row["fetch_key"]] = row["crawl_status"]
            self.connection.executemany(
                """
                UPDATE fetch_targets SET status=?, next_retry_at=NULL, updated_at=?
                WHERE fetch_key=?
                """,
                [(status, utc_now(), key) for key, status in by_fetch.items()],
            )
            self.connection.execute(
                "UPDATE shards SET status='COMMITTED', committed_at=? WHERE shard_name=?",
                (utc_now(), shard_name),
            )
            self.connection.execute("DELETE FROM shard_items WHERE shard_name=?", (shard_name,))

    def abort_staged_shard(self, shard_name: str) -> None:
        with self.transaction():
            keys = [
                row[0]
                for row in self.connection.execute(
                    "SELECT DISTINCT fetch_key FROM shard_items WHERE shard_name=?", (shard_name,)
                )
            ]
            self.connection.executemany(
                "UPDATE fetch_targets SET status='PENDING', updated_at=? WHERE fetch_key=?",
                [(utc_now(), key) for key in keys],
            )
            self.connection.executemany(
                "UPDATE crawl_tasks SET status='PENDING', updated_at=? WHERE fetch_key=?",
                [(utc_now(), key) for key in keys],
            )
            self.connection.execute("DELETE FROM shards WHERE shard_name=?", (shard_name,))

    def writing_shards(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM shards WHERE status='WRITING'"))

    def staged_doc_ids(self, shard_name: str) -> list[int]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT doc_id FROM shard_items WHERE shard_name=? ORDER BY doc_id", (shard_name,)
            )
        ]

    def counts(self) -> dict[str, int]:
        result = {
            row["status"]: row["count"]
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS count FROM crawl_tasks GROUP BY status"
            )
        }
        result["TOTAL"] = sum(result.values())
        return result

    def active_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM fetch_targets WHERE status='FETCHING'"
            ).fetchone()[0]
        )

    @staticmethod
    def _domain_exclusion_clause(excluded_domains: frozenset[str]) -> tuple[str, tuple[str, ...]]:
        if not excluded_domains:
            return "", ()
        placeholders = ",".join("?" for _ in excluded_domains)
        return f" AND domain NOT IN ({placeholders})", tuple(sorted(excluded_domains))

    def unfinished_count(self, excluded_domains: frozenset[str] | None = None) -> int:
        clause, parameters = self._domain_exclusion_clause(excluded_domains or frozenset())
        return int(
            self.connection.execute(
                f"""
                SELECT COUNT(*) FROM fetch_targets
                WHERE status IN ('PENDING', 'FETCHING', 'RETRY_WAIT')
                {clause}
                """,
                parameters,
            ).fetchone()[0]
        )

    def next_retry_delay(self, excluded_domains: frozenset[str] | None = None) -> float | None:
        clause, parameters = self._domain_exclusion_clause(excluded_domains or frozenset())
        value = self.connection.execute(
            f"""SELECT MIN(next_retry_at) FROM fetch_targets
            WHERE status='RETRY_WAIT'{clause}""",
            parameters,
        ).fetchone()[0]
        return max(0.0, float(value) - time.time()) if value is not None else None

    def domain_task_counts(self, domains: frozenset[str]) -> dict[str, int]:
        if not domains:
            return {}
        placeholders = ",".join("?" for _ in domains)
        return {
            str(row["domain"]): int(row["count"])
            for row in self.connection.execute(
                f"""SELECT domain, COUNT(*) AS count FROM crawl_tasks
                WHERE domain IN ({placeholders}) GROUP BY domain ORDER BY domain""",
                tuple(sorted(domains)),
            )
        }

    def task_rows(self) -> Iterable[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM crawl_tasks ORDER BY doc_id")
