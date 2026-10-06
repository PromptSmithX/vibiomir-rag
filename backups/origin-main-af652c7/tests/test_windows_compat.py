from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from crawler.pipeline import RunLock
from crawler.utils import files
from crawler.utils.urls import normalize_fetch_url


@pytest.mark.skipif(os.name != "nt", reason="Windows file locking behavior")
def test_windows_run_lock_rejects_second_process_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "crawler.lock"
    with RunLock(lock_path):
        with pytest.raises(RuntimeError, match="another crawler process"):
            with RunLock(lock_path):
                pass


@pytest.mark.skipif(os.name != "nt", reason="Windows transient rename behavior")
def test_windows_atomic_replace_retries_scanner_lock(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.parquet"
    source.write_bytes(b"ready")
    real_replace = files.os.replace
    calls = 0

    def flaky_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls < 3:
            error = PermissionError("file is temporarily locked")
            error.winerror = 32
            raise error
        return real_replace(src, dst)

    monkeypatch.setattr(files.os, "replace", flaky_replace)
    monkeypatch.setattr(files.time, "sleep", lambda _seconds: None)
    files.replace_with_retry(source, destination)

    assert calls == 3
    assert destination.read_bytes() == b"ready"


@pytest.mark.skipif(os.name != "nt", reason="Windows spawn process behavior")
def test_windows_process_pool_can_import_crawler_function() -> None:
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(normalize_fetch_url, "HTTPS://Example.COM/a#fragment")
        assert future.result(timeout=20) == "https://example.com/a"
