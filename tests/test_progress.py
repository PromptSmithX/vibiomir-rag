from io import StringIO

from crawler.models import CrawlStatus, FetchTarget
from crawler.progress import CrawlProgress


def test_full_progress_prints_terminal_url_and_advances() -> None:
    stream = StringIO()
    target = FetchTarget("key", "https://example.com/medical/article", "example.com", 1)
    progress = CrawlProgress(
        "full", 2, 3, stream=stream, is_terminal=True, width=140
    )

    with progress:
        progress.target_started(target)
        progress.terminal(target, CrawlStatus.SUCCESS, 200, 2048)

    output = stream.getvalue()
    assert "SUCCESS" in output
    assert "https://example.com/medical/article" in output
    assert "200" in output
    assert "2.0 KB" in output
    assert progress.completed == 1
    assert progress.active == 0


def test_retry_prints_line_without_advancing() -> None:
    stream = StringIO()
    target = FetchTarget("key", "https://example.com/retry", "example.com", 1)
    progress = CrawlProgress(
        "full", 1, 3, stream=stream, is_terminal=True, width=120
    )

    with progress:
        progress.target_started(target)
        progress.retry(target, "FETCH_ERROR")

    assert "RETRY 1/3" in stream.getvalue()
    assert progress.completed == 0
    assert progress.active == 0


def test_auto_progress_is_off_for_non_terminal_stream() -> None:
    progress = CrawlProgress(
        "auto", 10, 3, stream=StringIO(), is_terminal=False, width=80
    )
    assert not progress.enabled


def test_long_url_is_truncated_to_terminal_width() -> None:
    stream = StringIO()
    target = FetchTarget("key", "https://example.com/" + "a" * 200, "example.com", 1)
    progress = CrawlProgress(
        "full", 1, 3, stream=stream, is_terminal=True, width=100
    )

    with progress:
        progress.target_started(target)
        progress.terminal(target, CrawlStatus.FAILED, 500, 0)

    assert "…" in stream.getvalue()


def test_progress_displays_flushing_phase() -> None:
    stream = StringIO()
    progress = CrawlProgress(
        "bar", 1, 3, stream=stream, is_terminal=True, width=140
    )

    with progress:
        progress.set_phase("flushing")

    assert "flushing" in stream.getvalue()
