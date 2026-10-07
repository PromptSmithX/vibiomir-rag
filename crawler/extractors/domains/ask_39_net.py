from __future__ import annotations

import json
import re
from html import unescape

from selectolax.parser import HTMLParser


def _clean_str(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(s.split())


def extract_ask_39(html_bytes: bytes, url: str) -> tuple[str | None, str]:
    tree = HTMLParser(html_bytes)

    # 1. Title extraction
    title = None
    for sel in [
        "h1.title",
        "h1.ask_tit",
        "h1",
        'meta[itemprop="name"]',
        'meta[property="og:title"]',
        "title",
    ]:
        node = tree.css_first(sel)
        if node:
            is_meta = sel.startswith("meta")
            val = node.attributes.get("content") if is_meta else node.text(strip=True)
            if val:
                val = unescape(val.strip())
                val = re.sub(r"_(?:39问医生|39健康网).*$", "", val).strip()
                if val:
                    title = val
                    break

    blocks: list[str] = []
    if title:
        blocks.append(f"# {title}")

    # 2. Extract structured Q&A
    # Question details
    q_parts: list[str] = []
    for sel in [".ask_hid .txt_ms", ".ask_cont .txt_ms", ".ask_hid .txt_bc", ".mation"]:
        for node in tree.css(sel):
            t = _clean_str(node.text(strip=True))
            if t and t not in q_parts:
                q_parts.append(t)

    # Question tags/labels (only genuine medical tags, exclude related articles)
    labels = [_clean_str(node.text(strip=True)) for node in tree.css(".txt_label a")]
    labels = [label for label in labels if label]

    if q_parts:
        blocks.append("## 问题描述")
        blocks.extend(q_parts)
    if labels:
        blocks.append(f"**标签**: {', '.join(labels[:10])}")

    # Doctor answers / Content
    answers: list[str] = []
    seen_texts: set[str] = set()

    # Mobile answers
    mobile_nodes = tree.css(".pJingbianContent") or tree.css(".article-content")
    for node in mobile_nodes:
        t = _clean_str(node.text(strip=True))
        if t and len(t) >= 20 and t not in seen_texts:
            seen_texts.add(t)
            doc_node = tree.css_first(".doctor-info")
            doc_desc = _clean_str(doc_node.text(strip=True)) if doc_node else ""
            if doc_desc:
                answers.append(f"### 医生回复 ({doc_desc})\n{t}")
            else:
                answers.append(f"### 医生回复\n{t}")

    # Desktop answers
    for node in tree.css(".sele_txt, .zwAll, .ans_cont"):
        t = _clean_str(node.text(strip=True))
        if t and len(t) >= 20 and t not in seen_texts:
            seen_texts.add(t)
            answers.append(f"### 医生回复\n{t}")

    if answers:
        blocks.append("## 医生回答")
        blocks.extend(answers)

    # 3. Fallback to ld+json if body is still brief
    text = "\n\n".join(blocks).strip()
    if len(text) < 100:
        for script in tree.css('script[type="application/ld+json"]'):
            try:
                raw_json = script.text(strip=True)
                if not raw_json:
                    continue
                data = json.loads(raw_json)
                if isinstance(data, dict):
                    ld_desc = _clean_str(data.get("description"))
                    ld_title = _clean_str(data.get("title"))
                    if not title and ld_title:
                        title = ld_title
                        if not blocks:
                            blocks.append(f"# {title}")
                    if ld_desc and len(ld_desc) >= 20 and ld_desc not in seen_texts:
                        seen_texts.add(ld_desc)
                        blocks.append("## 内容概要")
                        blocks.append(ld_desc)
            except Exception:
                pass
        text = "\n\n".join(blocks).strip()

    return title, text
