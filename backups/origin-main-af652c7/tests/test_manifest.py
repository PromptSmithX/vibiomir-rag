import time
from pathlib import Path

from crawler.manifest import Manifest, utc_now
from crawler.models import CrawlStatus, DocumentRecord


def seed_group(manifest: Manifest, key: str = "key-1") -> None:
    manifest.connection.execute(
        """
        INSERT INTO fetch_targets(fetch_key, request_url, domain, status, updated_at)
        VALUES (?, 'https://example.com/a', 'example.com', 'PENDING', ?)
        """,
        (key, utc_now()),
    )
    manifest.connection.executemany(
        """
        INSERT INTO crawl_tasks(
            doc_id, url, normalized_url, fetch_key, domain, status, updated_at
        ) VALUES (?, ?, 'https://example.com/a', ?, 'example.com', 'PENDING', ?)
        """,
        [
            (1, "https://example.com/a#one", key, utc_now()),
            (2, "https://example.com/a#two", key, utc_now()),
        ],
    )


def records(key: str = "key-1") -> list[DocumentRecord]:
    return [
        DocumentRecord(
            doc_id=doc_id,
            url=f"https://example.com/a#{doc_id}",
            final_url="https://example.com/a",
            domain="example.com",
            content_type="text/html",
            title="Title",
            text="Body",
            language="en",
            content_hash="abc",
            http_status=200,
            crawl_status=CrawlStatus.SUCCESS.value,
            text_chars=4,
            bytes_downloaded=100,
            fetch_key=key,
        )
        for doc_id in (1, 2)
    ]


def test_duplicate_url_claims_once_and_fans_out(tmp_path: Path) -> None:
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        manifest.create_schema()
        seed_group(manifest)
        claimed = manifest.claim_targets(10)
        assert len(claimed) == 1
        assert claimed[0].attempts == 1
        assert [doc.doc_id for doc in manifest.documents_for_target("key-1")] == [1, 2]


def test_staged_shard_commit_is_atomic_for_manifest(tmp_path: Path) -> None:
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        manifest.create_schema()
        seed_group(manifest)
        manifest.claim_targets(1)
        manifest.stage_shard(
            "part-000000.parquet",
            tmp_path / "part.tmp",
            tmp_path / "part.parquet",
            records(),
        )
        assert manifest.counts()[CrawlStatus.FETCHING.value] == 2
        manifest.commit_staged_shard("part-000000.parquet")
        counts = manifest.counts()
        assert counts[CrawlStatus.SUCCESS.value] == 2
        rows = list(manifest.task_rows())
        assert {row["output_shard"] for row in rows} == {"part-000000.parquet"}
        assert manifest.writing_shards() == []


def test_abort_staged_shard_resets_without_losing_attempts(tmp_path: Path) -> None:
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        manifest.create_schema()
        seed_group(manifest)
        manifest.claim_targets(1)
        manifest.stage_shard(
            "part-000000.parquet",
            tmp_path / "part.tmp",
            tmp_path / "part.parquet",
            records(),
        )
        manifest.abort_staged_shard("part-000000.parquet")
        rows = list(manifest.task_rows())
        assert {row["status"] for row in rows} == {CrawlStatus.PENDING.value}
        assert {row["attempts"] for row in rows} == {1}


def test_claim_interleaves_domains(tmp_path: Path) -> None:
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        manifest.create_schema()
        for domain_index in range(4):
            for item_index in range(3):
                key = f"{domain_index}-{item_index}"
                domain = f"d{domain_index}.example"
                manifest.connection.execute(
                    """
                    INSERT INTO fetch_targets(
                        fetch_key, request_url, domain, status, updated_at
                    ) VALUES (?, ?, ?, 'PENDING', ?)
                    """,
                    (key, f"https://{domain}/{item_index}", domain, utc_now()),
                )
        claimed = manifest.claim_targets(4)
        assert len({target.domain for target in claimed}) == 4


def test_claim_prioritizes_due_retry_and_ignores_future_retry(tmp_path: Path) -> None:
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        manifest.create_schema()
        now = time.time()
        manifest.connection.executemany(
            """
            INSERT INTO fetch_targets(
                fetch_key,request_url,domain,status,next_retry_at,updated_at
            ) VALUES (?,?,?,?,?,?)
            """,
            [
                ("pending", "https://example.com/p", "example.com", "PENDING", None, utc_now()),
                ("due", "https://example.com/d", "example.com", "RETRY_WAIT", now - 1, utc_now()),
                (
                    "future",
                    "https://example.com/f",
                    "future.example",
                    "RETRY_WAIT",
                    now + 60,
                    utc_now(),
                ),
            ],
        )

        claimed = manifest.claim_targets(2)

        assert [target.fetch_key for target in claimed] == ["due", "pending"]
        assert manifest.claim_targets(1) == []


def test_claim_uses_domain_status_index(tmp_path: Path) -> None:
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        manifest.create_schema()
        indexes = {
            row[1] for row in manifest.connection.execute("PRAGMA index_list(fetch_targets)")
        }
        assert "idx_fetch_domain_status_retry" in indexes
        plan = " ".join(
            row[3]
            for row in manifest.connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT fetch_key,request_url,domain,attempts
                FROM fetch_targets INDEXED BY idx_fetch_domain_status_retry
                WHERE domain=? AND status='PENDING'
                ORDER BY next_retry_at,fetch_key LIMIT ?
                """,
                ("example.com", 3),
            )
        )
        assert "idx_fetch_domain_status_retry" in plan
        assert "TEMP B-TREE" not in plan
