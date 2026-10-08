from __future__ import annotations

from selectolax.parser import HTMLParser
import trafilatura


def extract_thanhnien(content: bytes, url: str) -> tuple[str | None, str]:
    """Extract article title and text from thanhnien.vn."""
    tree = HTMLParser(content)
    title = None
    for sel in ('meta[property="og:title"]', "h1", "title"):
        node = tree.css_first(sel)
        if node:
            t = (
                node.attributes.get("content")
                if sel.startswith("meta")
                else node.text(strip=True)
            )
            if t:
                title = t.strip()
                break

    for sel in (
        "#box-detail-related",
        ".detail__related",
        ".detail__comment",
        ".detail-social",
        "script",
        "style",
    ):
        for node in tree.css(sel):
            node.decompose()

    body_node = tree.css_first(".detail-cmain") or tree.css_first(".detail-content")
    if body_node:
        paragraphs: list[str] = []
        sapo = tree.css_first(".detail-sapo")
        if sapo and sapo.text(strip=True):
            paragraphs.append(sapo.text(strip=True))
        for child in body_node.css("div, p"):
            t = child.text(separator=" ", strip=True)
            if t and len(t) > 20 and not child.css("div, p"):
                if not paragraphs or paragraphs[-1] != t:
                    paragraphs.append(t)
        text = "\n\n".join(paragraphs)
        if len(text) >= 200:
            return title, text

    fallback = trafilatura.extract(content, url=url, output_format="markdown")
    return title, fallback or ""
