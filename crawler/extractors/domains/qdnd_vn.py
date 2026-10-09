from __future__ import annotations

from selectolax.parser import HTMLParser
import trafilatura


def extract_qdnd(content: bytes, url: str) -> tuple[str | None, str]:
    """Extract article title and text from qdnd.vn."""
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
        "script",
        "style",
        "figure",
        ".relate-news",
        ".detail-related",
        ".author-info",
        ".float-image",
        ".post-tool",
        ".comment-box",
    ):
        for node in tree.css(sel):
            node.decompose()

    body_node = (
        tree.css_first("div.articleContent")
        or tree.css_first("div.post-content")
        or tree.css_first("div.detail-post")
    )
    if body_node:
        paragraphs: list[str] = []
        for p in body_node.css("p"):
            t = p.text(separator=" ", strip=True)
            if t and len(t) > 15:
                if not paragraphs or paragraphs[-1] != t:
                    paragraphs.append(t)
        text = "\n\n".join(paragraphs)
        if len(text) >= 100:
            return title, text

    fallback = trafilatura.extract(content, url=url, output_format="markdown")
    return title, fallback or ""
