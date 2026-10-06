from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlsplit

NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "aside",
    "form",
    "button",
    "svg",
    "canvas",
    "iframe",
)

BLOCK_TAGS = {"p", "blockquote", "pre"}
HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}

# These markers target page chrome by HTML structure, not by visible words.  Text
# keyword filtering is deliberately avoided because medical articles can
# legitimately discuss comments, navigation, recommendations, or advertising.
NOISE_ATTRIBUTE_MARKERS = {
    "advert",
    "advertisement",
    "breadcrumb",
    "comment",
    "comments",
    "copyright",
    "footer",
    "menu",
    "nav",
    "newsletter",
    "outbrain",
    "pager",
    "pagination",
    "recommend",
    "recommended",
    "related",
    "share",
    "sharing",
    "sidebar",
    "social",
    "taboola",
}


def _attribute_tokens(node) -> set[str]:
    value = " ".join(
        (node.attributes.get("id") or "", node.attributes.get("class") or "")
    ).lower()
    return {token for token in re.split(r"[^a-z0-9]+", value) if token}


def _remove_structural_noise(tree) -> None:
    """Remove common page-chrome containers before choosing the main root."""
    # Re-query after every removal: selectolax invalidates sibling node handles
    # obtained by the same traversal when one of them is decomposed.
    while True:
        noise = next(
            (
                node
                for node in tree.css("[id], [class]")
                if _attribute_tokens(node) & NOISE_ATTRIBUTE_MARKERS
            ),
            None,
        )
        if noise is None:
            return
        noise.decompose()


def _node_text(node) -> str:
    return " ".join(node.text(separator=" ", strip=True).split())


def _table_markdown(table) -> str:
    rows: list[list[str]] = []
    for row in table.css("tr"):
        cells = [_node_text(cell).replace("|", "\\|") for cell in row.css("th, td")]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _serialize(root) -> str:
    blocks: list[str] = []
    for node in root.traverse(include_text=False):
        tag = node.tag.lower() if node.tag else ""
        if tag in HEADING_TAGS:
            text = _node_text(node)
            if text:
                blocks.append(f"{'#' * HEADING_TAGS[tag]} {text}")
        elif tag in BLOCK_TAGS:
            text = _node_text(node)
            if text:
                blocks.append(text)
        elif tag == "li":
            text = _node_text(node)
            if text:
                blocks.append(f"- {text}")
        elif tag == "table":
            rendered = _table_markdown(node)
            if rendered:
                blocks.append(rendered)
    deduplicated: list[str] = []
    for block in blocks:
        if not deduplicated or deduplicated[-1] != block:
            deduplicated.append(block)
    return "\n\n".join(deduplicated)


def _link_density(root) -> float:
    text = root.text(separator=" ", strip=True)
    if not text:
        return 1.0
    linked = sum(len(node.text(separator=" ", strip=True)) for node in root.css("a"))
    return linked / max(1, len(text))


def extract_html(content: bytes, url: str) -> tuple[str | None, str, str]:
    from crawler.extractors.domains import get

    domain = urlsplit(url).hostname or ""
    custom = get(domain)
    if custom is not None:
        title, text = custom(content, url)
        return title, text, f"domain:{domain}"
    try:
        from selectolax.parser import HTMLParser
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("selectolax is required; run `uv sync`") from exc

    tree = HTMLParser(content)
    for tag in NOISE_TAGS:
        for node in tree.css(tag):
            node.decompose()
    _remove_structural_noise(tree)

    title = None
    for selector in ('meta[property="og:title"]', "h1", "title"):
        node = tree.css_first(selector)
        if not node:
            continue
        value = node.attributes.get("content") if selector.startswith("meta") else _node_text(node)
        if value:
            title = unescape(value.strip())
            break

    candidates = tree.css("article, main, [role=main]")
    if not candidates and tree.body:
        candidates = [tree.body]
    root = max(candidates, key=lambda node: len(node.text(strip=True)), default=tree.root)
    text = _serialize(root)
    density = _link_density(root)

    if len(text) < 100 or density > 0.45:
        try:
            import trafilatura

            fallback = trafilatura.extract(
                tree.html.encode("utf-8"),
                url=url,
                output_format="markdown",
                include_tables=True,
                include_links=False,
                favor_precision=True,
                with_metadata=False,
            )
            if fallback and len(fallback) > len(text):
                return title, fallback, "trafilatura"
        except Exception:
            if not text:
                raise
    return title, text, f"selectolax:{domain or 'unknown'}"
