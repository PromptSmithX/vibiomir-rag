from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?%])")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines: list[str] = []
    for raw_line in value.split("\n"):
        line = _SPACE.sub(" ", raw_line).strip()
        line = _SPACE_BEFORE_PUNCT.sub(r"\1", line)
        lines.append(line)
    value = "\n".join(lines).strip()
    return _BLANK_LINES.sub("\n\n", value)


def normalize_for_hash(text: str) -> str:
    value = clean_text(text)
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def looks_like_javascript_shell(html: bytes, text: str) -> bool:
    if len(text.strip()) >= 100:
        return False
    lower = html[:200_000].lower()
    markers = (
        b"__next_data__",
        b"id=\"__next\"",
        b"id=\"root\"",
        b"id=\"app\"",
        b"enable javascript",
        b"requires javascript",
    )
    return any(marker in lower for marker in markers)

