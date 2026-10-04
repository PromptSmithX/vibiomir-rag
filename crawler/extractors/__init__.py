from __future__ import annotations

from typing import Any

from crawler.cleaner import clean_text, looks_like_javascript_shell, normalize_for_hash
from crawler.language import detect_language
from crawler.models import CrawlStatus, ExtractedContent, FetchOutcome
from crawler.utils.hashing import sha256_text


def extract_fetch_outcome(outcome: FetchOutcome, settings: dict[str, Any]) -> ExtractedContent:
    try:
        content = outcome.content or b""
        content_type = (outcome.content_type or "").lower()
        if content.startswith(b"%PDF-") or content_type == "application/pdf":
            from crawler.extractors.pdf import extract_pdf

            title, text, extractor, needs_ocr = extract_pdf(
                content,
                min_chars_per_page=int(settings["pdf_min_chars_per_page"]),
            )
            status = CrawlStatus.NEEDS_OCR if needs_ocr else CrawlStatus.SUCCESS
        elif content_type.startswith("text/plain"):
            title = None
            text = content.decode("utf-8", errors="replace")
            extractor = "plain_text"
            status = CrawlStatus.SUCCESS
        else:
            from crawler.extractors.html_generic import extract_html

            title, text, extractor = extract_html(
                content, outcome.final_url or outcome.target.request_url
            )
            status = CrawlStatus.SUCCESS

        title = clean_text(title)
        text = clean_text(text)
        if status == CrawlStatus.SUCCESS and len(text) < int(settings["min_text_chars"]):
            status = (
                CrawlStatus.NEEDS_JS
                if looks_like_javascript_shell(content, text)
                else CrawlStatus.EMPTY_CONTENT
            )
        language = None
        content_hash = None
        if text:
            language = detect_language(
                f"{title}\n{text[:3000]}", float(settings["language_confidence"])
            )
            content_hash = sha256_text(normalize_for_hash(text))
        return ExtractedContent(
            status=status,
            title=title or None,
            text=text or None,
            language=language,
            content_hash=content_hash,
            extractor=extractor,
        )
    except Exception as exc:
        return ExtractedContent(
            status=CrawlStatus.FAILED,
            error_type="EXTRACTION_ERROR",
            error=f"{type(exc).__name__}: {exc}"[:2000],
        )
