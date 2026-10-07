import gzip
from pathlib import Path

import pytest

from crawler.archive import (
    ArchiveRecord,
    ArchiveStore,
    archive_lookup_key,
    archive_query_prefix,
    merge_ranges,
    parse_warc_record,
    resolve_archive_prefixes,
    select_probe_targets,
)
from crawler.manifest import Manifest, utc_now


def archive_record(key: str, filename: str, offset: int, length: int) -> ArchiveRecord:
    return ArchiveRecord(
        key,
        f"https://example.com/{key}",
        "example.com",
        "CC-MAIN-2026-01",
        "20260101000000",
        filename,
        offset,
        length,
        "text/html",
        200,
    )


def test_parse_warc_record_extracts_archived_http_payload() -> None:
    html = b"<html><main>medical article</main></html>"
    raw = (
        b"WARC/1.0\r\nWARC-Type: response\r\n\r\n"
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
        + html
    )

    body, content_type, status = parse_warc_record(gzip.compress(raw))

    assert body == html
    assert content_type == "text/html"
    assert status == 200


def test_merge_ranges_combines_nearby_records_without_crossing_files() -> None:
    records = [
        archive_record("a", "one.warc.gz", 100, 50),
        archive_record("b", "one.warc.gz", 175, 25),
        archive_record("c", "two.warc.gz", 110, 20),
    ]

    groups = merge_ranges(records, max_bytes=1024, max_gap=32)

    assert [(group.filename, group.start, group.end) for group in groups] == [
        ("one.warc.gz", 100, 200),
        ("two.warc.gz", 110, 130),
    ]
    assert [item.fetch_key for item in groups[0].records] == ["a", "b"]


def test_archive_store_is_resumable(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path / "archive.sqlite") as store:
        assert store.seed([("key", "https://example.com/a", "example.com")]) == 1
        store.set_match(
            "key", archive_record("key", "one.warc.gz", 10, 20), None
        )
        assert store.counts() == {"FOUND": 1}
        store.mark_fetching(["key"])
        assert store.counts() == {"FETCHING": 1}
        store.mark_success(["key"])
        assert store.counts() == {"SUCCESS": 1}


def test_archive_prefix_groups_host_and_first_path_segment() -> None:
    assert (
        archive_query_prefix("https://Example.com/question/123?q=1")
        == "example.com/question/"
    )
    assert archive_query_prefix("http://example.com/article.html") == "example.com/"
    assert archive_lookup_key("https://example.com/a?q=1") == "//example.com/a?q=1"
    assert archive_lookup_key("http://example.com/a?q=1") == "//example.com/a?q=1"


def test_probe_selection_is_fixed_seed_and_domain_stratified(tmp_path: Path) -> None:
    with Manifest(tmp_path / "manifest.sqlite") as manifest:
        manifest.create_schema()
        for domain_index in range(2):
            domain = f"d{domain_index}.example"
            for item_index in range(10):
                key = f"{domain_index}{item_index:063d}"
                manifest.connection.execute(
                    """
                    INSERT INTO fetch_targets(
                        fetch_key,request_url,domain,status,updated_at
                    ) VALUES (?,?,?,'PENDING',?)
                    """,
                    (key, f"https://{domain}/{item_index}", domain, utc_now()),
                )

        first = select_probe_targets(manifest, size=6, seed=42)
        second = select_probe_targets(manifest, size=6, seed=42)

    assert first == second
    assert len(first) == 6
    assert {row[2] for row in first} == {"d0.example", "d1.example"}


@pytest.mark.asyncio
async def test_prefix_resolution_matches_capture_across_http_schemes(
    tmp_path: Path,
) -> None:
    class FakeClient:
        async def collections(self):
            return ["CC-MAIN-2026-01"]

        async def scan_prefix(self, prefix, index_name, on_record):
            assert prefix == "example.com/article/"
            return on_record(
                ArchiveRecord(
                    "capture-key",
                    "http://example.com/article/1",
                    "example.com",
                    index_name,
                    "20260101000000",
                    "one.warc.gz",
                    10,
                    20,
                    "text/html",
                    200,
                )
            )

    with ArchiveStore(tmp_path / "archive.sqlite") as store:
        store.seed(
            [
                (
                    "source-key",
                    "https://example.com/article/1",
                    "example.com",
                )
            ]
        )

        result = await resolve_archive_prefixes(store, FakeClient())

        assert result["found"] == 1
        assert store.counts() == {"FOUND": 1}
