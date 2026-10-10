from __future__ import annotations

import tempfile
from pathlib import Path

from crawler.config import load_config
from crawler.manifest import Manifest
from crawler.writer import ParquetShardWriter


def test_manifest_adapter_recovery_integration() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_manifest.sqlite"
        config = load_config("config/crawler.yaml")
        config.paths.manifest = db_path
        config.paths.corpus_dir = Path(tmp_dir) / "corpus"
        config.paths.temp_dir = Path(tmp_dir) / "tmp"

        with Manifest(db_path) as manifest:
            manifest.create_schema()

            # Insert sample targets & tasks
            targets = [
                ("k_woman", "http://woman.39.net/a/1.html", "woman.39.net", "EMPTY_CONTENT"),
                ("k_ask_empty", "http://ask.39.net/q/1.html", "ask.39.net", "EMPTY_CONTENT"),
                ("k_ask_succ", "http://ask.39.net/q/2.html", "ask.39.net", "SUCCESS"),
                ("k_hn", "https://hanoimoi.vn/news/1.html", "hanoimoi.vn", "FAILED"),
                ("k_baidu", "https://www.baidu.com/bh/dict/1", "www.baidu.com", "ROBOTS_DENIED"),
                ("k_generic_fail", "https://example.com/page", "example.com", "FAILED"),
            ]
            for key, url, domain, status in targets:
                manifest.connection.execute(
                    """
                    INSERT INTO fetch_targets(fetch_key, request_url, domain, status, updated_at)
                    VALUES (?, ?, ?, ?, '2026-10-07T00:00:00Z')
                    """,
                    (key, url, domain, status),
                )

            tasks = [
                (1, "k_woman", "woman.39.net", "EMPTY_CONTENT", 200, 2287, None),
                (2, "k_ask_empty", "ask.39.net", "EMPTY_CONTENT", 200, 2287, None),
                (3, "k_ask_succ", "ask.39.net", "SUCCESS", 200, 1500, None),
                (4, "k_hn", "hanoimoi.vn", "FAILED", 403, 0, "HTTP_403"),
                (5, "k_baidu", "www.baidu.com", "ROBOTS_DENIED", None, 0, "ROBOTS_DENIED"),
                (6, "k_generic_fail", "example.com", "FAILED", 500, 0, "SERVER_ERROR"),
            ]
            for doc_id, key, domain, status, http, b_down, err_type in tasks:
                manifest.connection.execute(
                    """
                    INSERT INTO crawl_tasks(
                        doc_id, url, normalized_url, fetch_key, domain, status,
                        http_status, bytes_downloaded, error_type, attempts, updated_at
                    )
                    VALUES (?, 'http://u', 'http://u', ?, ?, ?, ?, ?, ?, 3, '2026-10-07T00:00:00Z')
                    """,
                    (doc_id, key, domain, status, http, b_down, err_type),
                )
            manifest.connection.commit()

            # Execute recovery
            writer = ParquetShardWriter(config, manifest)
            res = writer.recover(recover_adapters=True, recover_network_errors=True)

            assert res["reset_targets"] == 4  # woman, ask_empty, hn, baidu

            # Verify statuses
            cur = manifest.connection.cursor()
            status_map = dict(
                cur.execute("SELECT doc_id, status FROM crawl_tasks").fetchall()
            )
            assert status_map[1] == "PENDING"  # woman recovered
            assert status_map[2] == "PENDING"  # ask 2287b recovered
            assert status_map[3] == "SUCCESS"  # ask success untouched
            assert status_map[4] == "PENDING"  # hanoimoi recovered
            assert status_map[5] == "PENDING"  # baidu recovered
            assert status_map[6] == "FAILED"   # generic fail untouched
