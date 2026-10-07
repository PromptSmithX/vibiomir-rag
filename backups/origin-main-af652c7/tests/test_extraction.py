import pytest

pytest.importorskip("selectolax")
pytest.importorskip("trafilatura")
pytest.importorskip("lingua")

from crawler.extractors import extract_fetch_outcome  # noqa: E402
from crawler.models import CrawlStatus, FetchOutcome, FetchTarget  # noqa: E402


def test_html_extraction_keeps_markdown_structure_and_medical_values() -> None:
    html = b"""
    <html><head><title>Hypertension</title></head><body>
      <nav>menu menu</nav>
      <article><h1>Treatment</h1><h2>Dose</h2>
      <p>Give 0.5 mg or 1:1000 solution. SpO2 95%.</p>
      <ul><li>IV</li><li>IM</li></ul></article>
    </body></html>
    """
    outcome = FetchOutcome(
        FetchTarget("key", "https://example.com/a", "example.com", 1),
        CrawlStatus.SUCCESS,
        final_url="https://example.com/a",
        content_type="text/html",
        content=html,
        http_status=200,
        bytes_downloaded=len(html),
    )
    result = extract_fetch_outcome(
        outcome,
        {"min_text_chars": 20, "pdf_min_chars_per_page": 50, "language_confidence": 0.15},
    )
    assert result.status == CrawlStatus.SUCCESS
    assert "# Treatment" in result.text
    assert "## Dose" in result.text
    assert "0.5 mg" in result.text
    assert "1:1000" in result.text
    assert "SpO2 95%" in result.text
    assert "- IV" in result.text
    assert "menu menu" not in result.text


def test_javascript_shell_is_classified() -> None:
    html = b'<html><body><div id="__next"></div><script>app()</script></body></html>'
    outcome = FetchOutcome(
        FetchTarget("key", "https://example.com/a", "example.com", 1),
        CrawlStatus.SUCCESS,
        final_url="https://example.com/a",
        content_type="text/html",
        content=html,
    )
    result = extract_fetch_outcome(
        outcome,
        {"min_text_chars": 100, "pdf_min_chars_per_page": 50, "language_confidence": 0.15},
    )
    assert result.status == CrawlStatus.NEEDS_JS


def test_html_extraction_removes_structural_related_and_share_blocks() -> None:
    html = b"""
    <html><head><title>Article title</title></head><body>
      <main>
        <article>
          <h1>Article title</h1>
          <p>The medical body contains a 5 mg dose and enough useful detail.</p>
          <div class="article-related"><p>Unrelated story one</p></div>
          <section id="social-sharing"><p>Share this article everywhere</p></section>
        </article>
      </main>
    </body></html>
    """
    outcome = FetchOutcome(
        FetchTarget("key", "https://example.com/article", "example.com", 1),
        CrawlStatus.SUCCESS,
        final_url="https://example.com/article",
        content_type="text/html",
        content=html,
        http_status=200,
        bytes_downloaded=len(html),
    )
    result = extract_fetch_outcome(
        outcome,
        {"min_text_chars": 20, "pdf_min_chars_per_page": 50, "language_confidence": 0.15},
    )
    assert result.status == CrawlStatus.SUCCESS
    assert "5 mg" in result.text
    assert "Unrelated story" not in result.text
    assert "Share this article" not in result.text


def test_html_extraction_accepts_attributes_without_values() -> None:
    html = b"""
    <html><head><title>Safe attributes</title></head><body>
      <main>
        <article>
          <h1>Safe attributes</h1>
          <p class>This medical article keeps its 10 mg dose after extraction.</p>
          <div id><p>This paragraph is also valid content.</p></div>
        </article>
      </main>
    </body></html>
    """
    outcome = FetchOutcome(
        FetchTarget("key", "https://example.com/safe", "example.com", 1),
        CrawlStatus.SUCCESS,
        final_url="https://example.com/safe",
        content_type="text/html",
        content=html,
        http_status=200,
        bytes_downloaded=len(html),
    )

    result = extract_fetch_outcome(
        outcome,
        {"min_text_chars": 20, "pdf_min_chars_per_page": 50, "language_confidence": 0.15},
    )

    assert result.status == CrawlStatus.SUCCESS
    assert result.error is None
    assert "10 mg" in result.text
    assert "also valid content" in result.text
