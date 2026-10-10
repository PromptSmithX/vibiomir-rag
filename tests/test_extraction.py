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


def test_39_net_verification_redirect_is_classified_as_bot_challenge() -> None:
    html = """<html><head><title>39健康</title></head><body></body></html>""".encode()
    outcome = FetchOutcome(
        FetchTarget("key", "https://ask.39.net/question/1.html", "ask.39.net", 1),
        CrawlStatus.SUCCESS,
        final_url="https://image.39.net/verify.html?referer=https%3A%2F%2Fask.39.net",
        content_type="text/html",
        content=html,
        http_status=200,
        bytes_downloaded=len(html),
    )

    result = extract_fetch_outcome(
        outcome,
        {"min_text_chars": 20, "pdf_min_chars_per_page": 50, "language_confidence": 0.15},
    )

    assert result.status == CrawlStatus.NEEDS_JS
    assert result.error_type == "BOT_CHALLENGE"
    assert result.text is None


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


def test_ask_39_extractor_desktop() -> None:
    html = (
        "<html><head><title>慢性咽炎_39问医生</title></head><body>"
        '<h1 class="ask_tit">慢性咽炎会导致scc升高吗</h1>'
        '<div class="ask_cont"><p class="txt_ms">病情描述：咽喉不适</p></div>'
        '<div class="txt_label"><a>咽喉科</a><a>慢性咽炎</a></div>'
        '<p class="sele_txt">慢性咽炎一般不会导致scc升高，建议定期复查并配合治疗。</p>'
        "</body></html>"
    ).encode()
    outcome = FetchOutcome(
        FetchTarget("key", "https://ask.39.net/question/1.html", "ask.39.net", 1),
        CrawlStatus.SUCCESS,
        final_url="https://ask.39.net/question/1.html",
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
    assert result.extractor == "domain:ask.39.net"
    assert "慢性咽炎" in (result.title or "")
    assert "建议定期复查" in (result.text or "")


def test_ask_39_extractor_mobile() -> None:
    html = (
        '<html><head><meta itemprop="name" content="咽炎咬牙"/></head><body>'
        '<h1 class="title">咽炎咬牙怎么办</h1>'
        '<div class="doctor-info"><span class="doctor-name">李医生</span></div>'
        '<div class="pJingbianContent">请注意保持口腔清洁，多喝温水，可以适量服用药物。</div>'
        "</body></html>"
    ).encode()
    outcome = FetchOutcome(
        FetchTarget("key", "https://ask.39.net/question/2.html", "ask.39.net", 1),
        CrawlStatus.SUCCESS,
        final_url="https://wapask.39.net/question/2.html",
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
    assert result.extractor == "domain:wapask.39.net"
    assert "咽炎" in (result.title or "")
    assert "李医生" in (result.text or "")
    assert "保持口腔清洁" in (result.text or "")


def test_qdnd_extractor() -> None:
    html = (
        "<html><head><title>Bảo đảm y tế dịp đại lễ</title></head><body>"
        "<h1>Bảo đảm y tế dịp đại lễ</h1>"
        '<div class="articleContent">'
        "<p>Để bảo đảm an toàn tuyệt đối, không để dịch bệnh bùng phát, ngành y tế đã triển khai phương án chi tiết.</p>"
        "<p>Các bệnh viện và trung tâm y tế duy trì trực cấp cứu 24/24 trong toàn bộ đợt cao điểm.</p>"
        "</div>"
        "</body></html>"
    ).encode()
    outcome = FetchOutcome(
        FetchTarget("key", "https://www.qdnd.vn/y-te/123", "www.qdnd.vn", 1),
        CrawlStatus.SUCCESS,
        final_url="https://www.qdnd.vn/y-te/123",
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
    assert result.extractor == "domain:www.qdnd.vn"
    assert "Bảo đảm y tế" in (result.title or "")
    assert "trực cấp cứu" in (result.text or "")


