"""Sentence-aware chunking engine for ViBioMIR Experiment A and extensible registry."""

from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

from retrieval.dense.config import ChunkingConfig
from retrieval.dense.tokenizer_helper import BaseTokenCounter, FallbackTokenCounter
from retrieval.models import DocumentChunk

logger = logging.getLogger(__name__)

# Common Vietnamese & English honorifics, medical titles, and abbreviations
ABBREVIATIONS = frozenset(
    {
        "bs",
        "ts",
        "pgs",
        "gs",
        "ths",
        "th.s",
        "bs.cki",
        "bs.ckii",
        "dr",
        "prof",
        "mr",
        "mrs",
        "ms",
        "e.g",
        "i.e",
        "etc",
        "v.v",
        "al",
        "tp",
        "p",
        "q",
        "no",
        "vs",
        "fig",
        "bv",
        "cdc",
        "who",
    }
)

# CJK sentence delimiters
CJK_PUNCT = "。！？；"


def split_sentences_multilingual(text: str) -> list[str]:
    """Split text into sentences respecting Vietnamese, English, and CJK boundaries.

    Avoids false splits on decimal points (3.14), drug dosages (10.5 mg),
    and common abbreviations (BS., TS., Dr., e.g., v.v.).
    """
    if not text:
        return []

    norm = unicodedata.normalize("NFC", text).strip()
    if not norm:
        return []

    # First split by paragraph breaks
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", norm) if p.strip()]
    all_sentences: list[str] = []

    for para in paragraphs:
        # Also split on single newlines if they are likely header/list boundaries
        lines = [line.strip() for line in para.split("\n") if line.strip()]
        for line in lines:
            sentences = _split_line_to_sentences(line)
            all_sentences.extend(sentences)

    return all_sentences


def _split_line_to_sentences(line: str) -> list[str]:
    """Split a single line into sentences."""
    if not line:
        return []

    # Boundary tokens scanner
    chunks: list[str] = []
    current: list[str] = []
    n = len(line)
    i = 0

    while i < n:
        char = line[i]
        current.append(char)

        # 1. CJK sentence terminators (always terminate)
        if char in CJK_PUNCT:
            # Include any immediate closing quotes/brackets
            while i + 1 < n and line[i + 1] in "”’'\"）)》]":
                i += 1
                current.append(line[i])
            s = "".join(current).strip()
            if s:
                chunks.append(s)
            current = []
            i += 1
            continue

        # 2. Western sentence terminators: . ! ? ;
        if char in ".!?;":
            is_boundary = False

            if char in "!?":
                # Check for multiple punctuation like !? or !!!
                while i + 1 < n and line[i + 1] in "!?":
                    i += 1
                    current.append(line[i])
                is_boundary = True
            elif char in ".;":
                # Check if it's a decimal number: e.g. 3.14, 0.5
                prev_char = line[i - 1] if i > 0 else ""
                next_char = line[i + 1] if i + 1 < n else ""
                if prev_char.isdigit() and next_char.isdigit():
                    is_boundary = False
                elif char == "." and _is_abbreviation("".join(current)):
                    is_boundary = False
                else:
                    # Lookahead: is followed by whitespace, quote, or end of string?
                    if i + 1 == n or next_char.isspace() or next_char in "”’'\"":
                        is_boundary = True

            if is_boundary:
                # Consume any trailing quote/bracket
                while i + 1 < n and line[i + 1] in "”’'\"）)》]":
                    i += 1
                    current.append(line[i])
                s = "".join(current).strip()
                if s:
                    chunks.append(s)
                current = []
                i += 1
                # Skip following whitespace
                while i < n and line[i].isspace():
                    i += 1
                continue

        i += 1

    rem = "".join(current).strip()
    if rem:
        chunks.append(rem)

    return chunks


def _is_abbreviation(buffer: str) -> bool:
    """Check if the text ending at current period is a recognized abbreviation."""
    # Find the last word token before period
    match = re.search(r"(\b[\w.]+)$", buffer.strip())
    if not match:
        return False
    last_word = match.group(1).lower().rstrip(".")
    if last_word in ABBREVIATIONS:
        return True
    # Single letter initials like 'A.' or 'B.'
    if len(last_word) == 1 and last_word.isalpha():
        return True
    return False


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, config: ChunkingConfig, token_counter: BaseTokenCounter | None = None) -> None:
        self.config = config
        self.token_counter = token_counter or FallbackTokenCounter()

    @abstractmethod
    def chunk_document(
        self,
        doc_id: int,
        title: str | None,
        text: str | None,
        language: str | None = None,
        heading: str | None = None,
    ) -> list[DocumentChunk]:
        """Split a document into DocumentChunk instances."""
        pass


class SentenceAwareChunker(BaseChunker):
    """Experiment A sentence-aware token-budget chunker.

    Splits document body into sentence units, accumulating consecutive sentences
    up to target_tokens (default 400), with a strict max token budget of 512
    (including title/heading context and special token reserves).
    """

    def chunk_document(
        self,
        doc_id: int,
        title: str | None,
        text: str | None,
        language: str | None = None,
        heading: str | None = None,
    ) -> list[DocumentChunk]:
        clean_text = (text or "").strip()
        clean_title = (title or "").strip() if self.config.include_document_title else ""
        clean_heading = (heading or "").strip() if self.config.include_heading_if_available else ""

        if not clean_text and not clean_title:
            return []

        # Prepare title prefix
        title_prefix = ""
        if clean_title:
            truncated_title = self.token_counter.truncate_to_tokens(
                clean_title, self.config.max_title_tokens
            )
            title_prefix = f"Title: {truncated_title}\n\n"

        heading_prefix = ""
        if clean_heading:
            truncated_heading = self.token_counter.truncate_to_tokens(
                clean_heading, self.config.max_title_tokens // 2
            )
            heading_prefix = f"Heading: {truncated_heading}\n\n"

        context_prefix = f"{title_prefix}{heading_prefix}"
        prefix_tokens = self.token_counter.count_tokens(context_prefix)

        # Token budgets
        effective_max = max(
            50,
            self.config.max_total_tokens - prefix_tokens - self.config.special_tokens_reserve,
        )
        effective_target = min(self.config.target_tokens, effective_max)

        # If document text is empty, but title exists:
        if not clean_text:
            chunk_text = clean_title
            embed_text = f"Title: {clean_title}"
            return [
                DocumentChunk(
                    chunk_id=f"{doc_id}_c0_{self.config.chunking_version}",
                    doc_id=doc_id,
                    chunk_index=0,
                    text=chunk_text,
                    embed_text=embed_text,
                    title=clean_title or None,
                    heading=clean_heading or None,
                    language=language,
                    token_count=self.token_counter.count_tokens(embed_text),
                    chunking_strategy=self.config.strategy,
                    chunking_version=self.config.chunking_version,
                )
            ]

        # 1. Split body into sentences
        sentences = split_sentences_multilingual(clean_text)
        if not sentences:
            sentences = [clean_text]

        # 2. Check if entire document fits into one chunk
        total_body_tokens = self.token_counter.count_tokens(clean_text)
        if total_body_tokens <= effective_max:
            embed_text = f"{context_prefix}{clean_text}" if context_prefix else clean_text
            return [
                DocumentChunk(
                    chunk_id=f"{doc_id}_c0_{self.config.chunking_version}",
                    doc_id=doc_id,
                    chunk_index=0,
                    text=clean_text,
                    embed_text=embed_text,
                    title=clean_title or None,
                    heading=clean_heading or None,
                    language=language,
                    token_count=self.token_counter.count_tokens(embed_text),
                    chunking_strategy=self.config.strategy,
                    chunking_version=self.config.chunking_version,
                )
            ]

        # 3. Accumulate sentences into chunks
        chunks: list[DocumentChunk] = []
        chunk_idx = 0
        current_sentences: list[str] = []
        current_tokens = 0

        def emit_chunk(body_str: str) -> None:
            nonlocal chunk_idx
            body_str = body_str.strip()
            if not body_str:
                return
            embed_t = f"{context_prefix}{body_str}" if context_prefix else body_str
            total_toks = self.token_counter.count_tokens(embed_t)
            c = DocumentChunk(
                chunk_id=f"{doc_id}_c{chunk_idx}_{self.config.chunking_version}",
                doc_id=doc_id,
                chunk_index=chunk_idx,
                text=body_str,
                embed_text=embed_t,
                title=clean_title or None,
                heading=clean_heading or None,
                language=language,
                token_count=total_toks,
                chunking_strategy=self.config.strategy,
                chunking_version=self.config.chunking_version,
            )
            chunks.append(c)
            chunk_idx += 1

        for sent in sentences:
            sent_tokens = self.token_counter.count_tokens(sent)

            # Case: single sentence is larger than effective_max
            if sent_tokens > effective_max:
                if current_sentences:
                    emit_chunk(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0

                # Split this oversized sentence by sliding token window
                sub_chunks = self._split_oversized_sentence(sent, effective_target, effective_max)
                for sc in sub_chunks:
                    emit_chunk(sc)
                continue

            # Check if adding exceeds effective_max
            if current_tokens + sent_tokens > effective_max:
                # Finalize current chunk
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    emit_chunk(chunk_text)

                    # Compute overlap from trailing sentences
                    overlap_sentences: list[str] = []
                    overlap_toks = 0
                    for s_rev in reversed(current_sentences):
                        s_tok = self.token_counter.count_tokens(s_rev)
                        if (
                            overlap_toks + s_tok <= self.config.overlap_tokens
                            and len(overlap_sentences) < len(current_sentences) - 1
                        ):
                            overlap_sentences.insert(0, s_rev)
                            overlap_toks += s_tok
                        else:
                            break

                    current_sentences = overlap_sentences
                    current_tokens = overlap_toks

            current_sentences.append(sent)
            current_tokens += sent_tokens

            # If current accumulated exceeds effective_target, we can emit chunk
            if current_tokens >= effective_target:
                chunk_text = " ".join(current_sentences)
                emit_chunk(chunk_text)

                overlap_sentences = []
                overlap_toks = 0
                for s_rev in reversed(current_sentences):
                    s_tok = self.token_counter.count_tokens(s_rev)
                    if (
                        overlap_toks + s_tok <= self.config.overlap_tokens
                        and len(overlap_sentences) < len(current_sentences) - 1
                    ):
                        overlap_sentences.insert(0, s_rev)
                        overlap_toks += s_tok
                    else:
                        break
                current_sentences = overlap_sentences
                current_tokens = overlap_toks

        # Flush remaining sentences
        if current_sentences:
            remaining_text = " ".join(current_sentences).strip()
            # Avoid emitting a duplicate if it completely matches the previous chunk's text
            if not chunks or remaining_text != chunks[-1].text:
                emit_chunk(remaining_text)

        return chunks

    def _split_oversized_sentence(
        self, sentence: str, target_tokens: int, max_tokens: int
    ) -> list[str]:
        """Split an excessively long sentence into token-sized windows."""
        words = sentence.split()
        if not words:
            return [sentence]

        stride_tokens = max(50, target_tokens - self.config.overlap_tokens)
        sub_chunks: list[str] = []
        start_idx = 0

        while start_idx < len(words):
            end_idx = start_idx
            accum_toks = 0

            while end_idx < len(words):
                w_tok = self.token_counter.count_tokens(words[end_idx])
                if accum_toks + w_tok > max_tokens and end_idx > start_idx:
                    break
                accum_toks += w_tok
                end_idx += 1
                if accum_toks >= target_tokens:
                    break

            chunk_slice = " ".join(words[start_idx:end_idx]).strip()
            if chunk_slice:
                sub_chunks.append(chunk_slice)

            if end_idx >= len(words):
                break

            # Advance by stride
            advance_idx = start_idx + 1
            advance_toks = 0
            while advance_idx < end_idx:
                w_tok = self.token_counter.count_tokens(words[advance_idx])
                advance_toks += w_tok
                if advance_toks >= stride_tokens:
                    break
                advance_idx += 1

            start_idx = max(start_idx + 1, advance_idx)

        return sub_chunks


class ChunkerRegistry:
    """Registry allowing seamless configuration-based selection between
    Experiment A (sentence_aware) and Experiment B (adaptive_medical)."""

    _registry: dict[str, type[BaseChunker]] = {
        "sentence_aware": SentenceAwareChunker,
    }

    @classmethod
    def register(cls, name: str, chunker_cls: type[BaseChunker]) -> None:
        cls._registry[name] = chunker_cls

    @classmethod
    def get(
        cls,
        config: ChunkingConfig,
        token_counter: BaseTokenCounter | None = None,
    ) -> BaseChunker:
        chunker_cls = cls._registry.get(config.strategy)
        if not chunker_cls:
            raise ValueError(
                f"Unknown chunking strategy: '{config.strategy}'. "
                f"Available strategies: {list(cls._registry.keys())}"
            )
        return chunker_cls(config, token_counter=token_counter)
