from __future__ import annotations


def extract_pdf(
    content: bytes, min_chars_per_page: int = 50
) -> tuple[str | None, str, str, bool]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required; run `uv sync`") from exc

    pages: list[str] = []
    with fitz.open(stream=content, filetype="pdf") as document:
        metadata_title = (document.metadata or {}).get("title") or None
        for page_number, page in enumerate(document, start=1):
            blocks = sorted(page.get_text("blocks"), key=lambda value: (value[1], value[0]))
            page_text = "\n".join(
                str(block[4]).strip() for block in blocks if str(block[4]).strip()
            )
            pages.append(f"## Page {page_number}\n\n{page_text}".strip())
        useful_chars = sum(len(page.split("\n\n", 1)[-1].strip()) for page in pages)
        threshold = max(100, len(document) * min_chars_per_page)
        needs_ocr = len(document) > 0 and useful_chars < threshold
    return metadata_title, "\n\n".join(pages), "pymupdf", needs_ocr
