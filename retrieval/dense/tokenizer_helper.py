"""Tokenizer helper and token counting abstractions for chunking and embedding."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# CJK regex pattern
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
# Western word pattern
WORD_PATTERN = re.compile(r"\b\w+(?:[-.]\w+)*\b")


class BaseTokenCounter(Protocol):
    """Protocol for counting and budgeting tokens across chunking strategies."""

    def count_tokens(self, text: str | None) -> int:
        """Count tokens for the given text without adding special tokens."""
        ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text so its token count does not exceed max_tokens."""
        ...


class FallbackTokenCounter:
    """Multilingual token counter used when HuggingFace tokenizer is unavailable
    (e.g., in strict offline test environments or before model download).
    Accurately counts English, Vietnamese syllables, and CJK characters.
    """

    def __init__(self) -> None:
        pass

    def count_tokens(self, text: str | None) -> int:
        if not text:
            return 0
        norm = unicodedata.normalize("NFC", text.strip())
        if not norm:
            return 0

        # Count CJK characters (1 CJK char ~= 1-1.5 token in BPE)
        cjk_chars = len(CJK_PATTERN.findall(norm))
        # Non-CJK part
        non_cjk = CJK_PATTERN.sub(" ", norm)
        words = non_cjk.split()
        # English / Vietnamese words average ~1.2 tokens per whitespace token
        word_token_estimate = int(len(words) * 1.25)

        total = max(1, cjk_chars + word_token_estimate)
        return total

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        if self.count_tokens(text) <= max_tokens:
            return text

        words = text.split()
        low = 1
        high = len(words)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            cand = " ".join(words[:mid])
            if self.count_tokens(cand) <= max_tokens:
                best = cand
                low = mid + 1
            else:
                high = mid - 1
        return best or words[0][:max_tokens * 3]


class HuggingFaceTokenCounter:
    """Token counter using actual Hugging Face AutoTokenizer for exact BPE counting."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count_tokens(self, text: str | None) -> int:
        if not text:
            return 0
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) <= max_tokens:
            return text
        truncated_ids = encoded[:max_tokens]
        return self.tokenizer.decode(truncated_ids, skip_special_tokens=True)


def get_token_counter(
    model_path_or_id: str | None = None,
    offline_mode: bool = False,
    fallback_if_missing: bool = True,
) -> BaseTokenCounter:
    """Resolve a token counter.

    Tries to load Hugging Face AutoTokenizer if transformers is installed.
    Falls back to FallbackTokenCounter if transformers is not installed or
    model files are not found (when fallback_if_missing=True).
    """
    if model_path_or_id:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(
                model_path_or_id,
                local_files_only=offline_mode,
                trust_remote_code=True,
            )
            return HuggingFaceTokenCounter(tok)
        except Exception as e:
            if not fallback_if_missing:
                raise RuntimeError(
                    f"Failed to load tokenizer for '{model_path_or_id}': {e}"
                ) from e
            logger.warning(
                "Could not load Hugging Face tokenizer for '%s' (%s). Using fallback counter.",
                model_path_or_id,
                e,
            )

    return FallbackTokenCounter()
