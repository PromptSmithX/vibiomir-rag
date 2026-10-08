from __future__ import annotations

from selectolax.parser import HTMLParser
import trafilatura


def extract_baothanhhoa(content: bytes, url: str) -> tuple[str | None, str]:
    """Extract article title and text from baothanhhoa.vn."""
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

    for sel in (".box-content", ".content-list", "script", "style"):
        for node in tree.css(sel):
            node.decompose()

    body_node = tree.css_first("#dcontent") or tree.css_first(".article__body")
    if body_node:
        paragraphs: list[str] = []
        sapo = tree.css_first(".article__sapo") or tree.css_first(".sapo")
        if sapo and sapo.text(strip=True):
            paragraphs.append(sapo.text(strip=True))
        for p in body_node.css("p"):
            t = p.text(separator=" ", strip=True)
            if t and len(t) > 10:
                paragraphs.append(t)
        text = "\n\n".join(paragraphs)
        if len(text) >= 200:
            return title, text

    fallback = trafilatura.extract(content, url=url, output_format="markdown")
    return title, fallback or ""
