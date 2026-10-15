"""High-precision source passage expansion engine for ViBioMIR submissions.

Expands retrieved chunks into continuous passages from authentic source documents
up to a strict token budget (e.g. 768 or 1024 tokens) while strictly preserving
document boundaries, candidate retrieval ranking, and textual provenance.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from retrieval.dense.config import PassageExpansionConfig
from retrieval.dense.tokenizer_helper import BaseTokenCounter, FallbackTokenCounter
from retrieval.submission.chunk_store import ChunkStore

logger = logging.getLogger(__name__)

# Sentence boundary patterns (period, exclamation, question mark, newline, CJK terminators)
SENTENCE_PUNCT_REGEX = re.compile(r"(?:[.\!?。！？]+[\s\n]+|\n{1,2})")


def strip_synthetic_prefixes(text: str) -> str:
    """Ensure no model instructions, PubMed IDs or prompt prefixes leak into passage."""
    s = text.strip()
    changed = True
    while changed:
        orig = s
        # 1. PubMed prefix with optional ID (e.g. pubmed:12345, pubmed: 12345, pubmed:)
        m_pm = re.match(r"^pubmed(?::\d+|\s*:\s*|\s+)", s, re.IGNORECASE)
        if m_pm:
            s = s[m_pm.end() :].strip()

        # 2. Instruct and Query prompt wrapper
        if "Instruct:" in s:
            idx = s.find("\nQuery: ")
            if idx != -1:
                s = s[idx + len("\nQuery: ") :].strip()
            else:
                idx2 = s.find("Instruct:")
                if idx2 == 0:
                    s = s[len("Instruct:") :].strip()

        if s.startswith("Query:"):
            s = s[len("Query:") :].strip()

        # 3. Bracketed metadata
        while s.startswith("[Tiêu đề:") or s.startswith("[Mục:"):
            end_bracket = s.find("]")
            if end_bracket != -1:
                s = s[end_bracket + 1 :].strip()
            else:
                break

        changed = s != orig

    return s.strip()


@dataclass
class PassageExpansionResult:
    """Detailed result of expanding a single chunk candidate."""

    doc_id: int
    chunk_id: str
    original_text: str
    expanded_text: str
    original_tokens: int
    expanded_tokens: int
    is_expanded: bool
    strategy_used: str  # "full_doc_slice", "full_document", "adjacent_chunks", "unexpanded"
    fallback_reason: str | None = None
    reached_max_tokens: bool = False
    start_offset: int | None = None
    end_offset: int | None = None


class SourcePassageExpander:
    """Context expansion engine that expands a retrieved chunk into a continuous
    source passage within its authentic parent document up to a configured token limit.
    """

    def __init__(
        self,
        config: PassageExpansionConfig | None = None,
        token_counter: BaseTokenCounter | None = None,
    ) -> None:
        self.config = config or PassageExpansionConfig()
        self.token_counter = token_counter or FallbackTokenCounter()
        self.results_history: list[PassageExpansionResult] = []

    def reset_history(self) -> None:
        """Reset historical results tracking."""
        self.results_history.clear()

    def expand_chunk(
        self,
        doc_id: int,
        chunk_id: str,
        chunk_text: str,
        chunk_meta: dict[str, Any] | None = None,
        chunk_store: ChunkStore | None = None,
    ) -> PassageExpansionResult:
        """Expand retrieved chunk text to adjacent continuous source text.

        Core invariants:
        1. Output is always a single continuous substring of the authentic source document.
        2. Never crosses document boundaries (doc_id is strictly preserved).
        3. Never synthesizes, paraphrases, or translates text.
        4. Token length never exceeds config.max_tokens.
        5. If chunk is already >= max_tokens, original text is preserved without arbitrary truncation.
        6. If ambiguous or missing source, safely falls back to original chunk and logs reason.
        """
        clean_chunk = strip_synthetic_prefixes(chunk_text)
        if not clean_chunk:
            res = PassageExpansionResult(
                doc_id=doc_id,
                chunk_id=chunk_id,
                original_text="",
                expanded_text="",
                original_tokens=0,
                expanded_tokens=0,
                is_expanded=False,
                strategy_used="unexpanded",
                fallback_reason="empty_chunk_text",
            )
            self.results_history.append(res)
            return res

        max_tokens = self.config.max_tokens
        curr_tokens = self.token_counter.count_tokens(clean_chunk)

        # Invariant 5: If chunk already at or above budget, retain chunk intact
        if curr_tokens >= max_tokens:
            res = PassageExpansionResult(
                doc_id=doc_id,
                chunk_id=chunk_id,
                original_text=clean_chunk,
                expanded_text=clean_chunk,
                original_tokens=curr_tokens,
                expanded_tokens=curr_tokens,
                is_expanded=False,
                strategy_used="unexpanded",
                fallback_reason="already_at_or_above_budget",
                reached_max_tokens=True,
            )
            self.results_history.append(res)
            return res

        # Strategy 1: Expand using full document text from corpus
        if chunk_store:
            corpus_doc_text = chunk_store.get_text_from_corpus(doc_id)
            if corpus_doc_text:
                clean_full = strip_synthetic_prefixes(corpus_doc_text)
                exp_res = self._expand_within_full_text(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    clean_chunk=clean_chunk,
                    full_text=clean_full,
                    curr_tokens=curr_tokens,
                    max_tokens=max_tokens,
                )
                if exp_res:
                    self.results_history.append(exp_res)
                    return exp_res

            # Strategy 2: Expand using verified adjacent sequential chunks
            doc_chunks = chunk_store.get_chunks_for_doc(doc_id)
            if doc_chunks and len(doc_chunks) > 1:
                adj_res = self._expand_from_adjacent_chunks(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    clean_chunk=clean_chunk,
                    doc_chunks=doc_chunks,
                    curr_tokens=curr_tokens,
                    max_tokens=max_tokens,
                )
                if adj_res:
                    self.results_history.append(adj_res)
                    return adj_res

        # Fallback: retain original chunk text
        fb_res = PassageExpansionResult(
            doc_id=doc_id,
            chunk_id=chunk_id,
            original_text=clean_chunk,
            expanded_text=clean_chunk,
            original_tokens=curr_tokens,
            expanded_tokens=curr_tokens,
            is_expanded=False,
            strategy_used="unexpanded",
            fallback_reason="no_safe_source_context_found",
        )
        self.results_history.append(fb_res)
        return fb_res

    def _expand_within_full_text(
        self,
        doc_id: int,
        chunk_id: str,
        clean_chunk: str,
        full_text: str,
        curr_tokens: int,
        max_tokens: int,
    ) -> PassageExpansionResult | None:
        """Locate retrieved chunk inside full source document text and expand symmetrically."""
        # Step 1: Check for ambiguous multiple occurrences first.
        # Refusal to guess anchor position if identical chunk appears multiple times in document.
        count_exact = full_text.count(clean_chunk)
        if count_exact == 0:
            norm_full = unicodedata.normalize("NFC", full_text)
            norm_chunk = unicodedata.normalize("NFC", clean_chunk)
            if norm_full.count(norm_chunk) > 1:
                count_exact = norm_full.count(norm_chunk)

        if count_exact > 1:
            logger.debug("Ambiguous match: chunk %s appears %d times in doc %d", chunk_id, count_exact, doc_id)
            return PassageExpansionResult(
                doc_id=doc_id,
                chunk_id=chunk_id,
                original_text=clean_chunk,
                expanded_text=clean_chunk,
                original_tokens=curr_tokens,
                expanded_tokens=curr_tokens,
                is_expanded=False,
                strategy_used="unexpanded",
                fallback_reason="ambiguous_multiple_matches",
            )

        # Step 2: Entire document fits within token budget
        total_doc_tokens = self.token_counter.count_tokens(full_text)
        if total_doc_tokens <= max_tokens:
            if clean_chunk in full_text or unicodedata.normalize("NFC", clean_chunk) in unicodedata.normalize("NFC", full_text):
                raw_full = full_text
                exp_text = raw_full.strip()
                is_diff = exp_text != clean_chunk
                l_ws = len(raw_full) - len(raw_full.lstrip())
                r_ws = len(raw_full) - len(raw_full.rstrip())
                return PassageExpansionResult(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    original_text=clean_chunk,
                    expanded_text=exp_text,
                    original_tokens=curr_tokens,
                    expanded_tokens=total_doc_tokens,
                    is_expanded=is_diff,
                    strategy_used="full_document" if is_diff else "unexpanded",
                    fallback_reason=None if is_diff else "chunk_equals_entire_document",
                    start_offset=l_ws,
                    end_offset=len(raw_full) - r_ws,
                )

        # Step 3: Document is larger than budget. Locate chunk uniquely.
        if count_exact == 1:
            start_pos = full_text.find(clean_chunk)
            end_pos = start_pos + len(clean_chunk)
        else:
            # Try NFC normalized lookup
            norm_full = unicodedata.normalize("NFC", full_text)
            norm_chunk = unicodedata.normalize("NFC", clean_chunk)
            if norm_full.count(norm_chunk) == 1:
                start_pos = norm_full.find(norm_chunk)
                end_pos = start_pos + len(norm_chunk)
                full_text = norm_full
            else:
                logger.debug("Chunk %s not found in full text of doc %d", chunk_id, doc_id)
                return PassageExpansionResult(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    original_text=clean_chunk,
                    expanded_text=clean_chunk,
                    original_tokens=curr_tokens,
                    expanded_tokens=curr_tokens,
                    is_expanded=False,
                    strategy_used="unexpanded",
                    fallback_reason="chunk_not_found_in_corpus",
                )

        # Verified unique slice: full_text[start_pos:end_pos] == clean_chunk
        remaining_budget = max_tokens - curr_tokens
        if remaining_budget <= 0:
            return PassageExpansionResult(
                doc_id=doc_id,
                chunk_id=chunk_id,
                original_text=clean_chunk,
                expanded_text=clean_chunk,
                original_tokens=curr_tokens,
                expanded_tokens=curr_tokens,
                is_expanded=False,
                strategy_used="unexpanded",
                fallback_reason="already_at_or_above_budget",
                reached_max_tokens=True,
                start_offset=start_pos,
                end_offset=end_pos,
            )

        # Find sentence boundaries
        sentence_starts = self._find_sentence_starts(full_text[:start_pos])
        sentence_ends = self._find_sentence_ends(full_text[end_pos:], offset_base=end_pos)

        target_left_budget = remaining_budget // 2
        target_right_budget = remaining_budget - target_left_budget

        # Step 1: Expand Left up to target_left_budget
        best_start = start_pos
        for s in sentence_starts:
            candidate_left = full_text[s:end_pos]
            cand_tokens = self.token_counter.count_tokens(candidate_left)
            if cand_tokens <= curr_tokens + target_left_budget:
                best_start = s
                break

        tokens_used_left = self.token_counter.count_tokens(full_text[best_start:end_pos]) - curr_tokens

        # Step 2: Expand Right with remaining budget (including unused left budget)
        available_right = remaining_budget - tokens_used_left
        best_end = end_pos
        for e in sentence_ends:
            candidate_both = full_text[best_start:e]
            cand_tokens = self.token_counter.count_tokens(candidate_both)
            if cand_tokens <= max_tokens:
                best_end = e
                break

        tokens_used_right = self.token_counter.count_tokens(full_text[best_start:best_end]) - self.token_counter.count_tokens(full_text[best_start:end_pos])

        # Step 3: Boundary redistribution: If right reached document end or could not use full budget, reallocate left
        leftover = max_tokens - self.token_counter.count_tokens(full_text[best_start:best_end])
        if leftover > 0 and best_start > 0:
            for s in sentence_starts:
                if s >= best_start:
                    continue
                cand_realloc = full_text[s:best_end]
                if self.token_counter.count_tokens(cand_realloc) <= max_tokens:
                    best_start = s
                    break

        raw_slice = full_text[best_start:best_end]
        expanded_passage = raw_slice.strip()
        leading_ws = len(raw_slice) - len(raw_slice.lstrip())
        trailing_ws = len(raw_slice) - len(raw_slice.rstrip())
        actual_start = best_start + leading_ws
        actual_end = best_end - trailing_ws
        exp_tokens = self.token_counter.count_tokens(expanded_passage)

        # Invariant 4: Token length never exceeds max_tokens
        if exp_tokens > max_tokens:
            # Back off to original chunk if boundary snap somehow overshot
            expanded_passage = clean_chunk
            exp_tokens = curr_tokens
            actual_start = start_pos
            actual_end = end_pos

        is_expanded = expanded_passage != clean_chunk and exp_tokens > curr_tokens
        reached_max = exp_tokens >= (max_tokens - 15)

        return PassageExpansionResult(
            doc_id=doc_id,
            chunk_id=chunk_id,
            original_text=clean_chunk,
            expanded_text=expanded_passage,
            original_tokens=curr_tokens,
            expanded_tokens=exp_tokens,
            is_expanded=is_expanded,
            strategy_used="full_doc_slice" if is_expanded else "unexpanded",
            fallback_reason=None if is_expanded else "cannot_expand_within_budget",
            reached_max_tokens=reached_max,
            start_offset=actual_start,
            end_offset=actual_end,
        )

    def _expand_from_adjacent_chunks(
        self,
        doc_id: int,
        chunk_id: str,
        clean_chunk: str,
        doc_chunks: list[dict[str, Any]],
        curr_tokens: int,
        max_tokens: int,
    ) -> PassageExpansionResult | None:
        """Expand using sequential adjacent chunks when full corpus document is unavailable."""
        curr_idx = -1
        for i, c in enumerate(doc_chunks):
            if c.get("chunk_id") == chunk_id:
                curr_idx = i
                break

        if curr_idx == -1:
            return None

        combined = clean_chunk
        combined_toks = curr_tokens

        # Symmetrically try appending next chunk
        if curr_idx + 1 < len(doc_chunks):
            next_text = strip_synthetic_prefixes(doc_chunks[curr_idx + 1].get("text", ""))
            if next_text:
                # Deduplicate overlapping prefix if next_text overlaps with tail of combined
                clean_next = self._strip_overlapping_prefix(combined, next_text)
                if clean_next:
                    cand = f"{combined} {clean_next}".strip()
                    cand_tokens = self.token_counter.count_tokens(cand)
                    if cand_tokens <= max_tokens:
                        combined = cand
                        combined_toks = cand_tokens

        # Symmetrically try prepending previous chunk if budget remains
        if curr_idx > 0 and combined_toks < max_tokens:
            prev_text = strip_synthetic_prefixes(doc_chunks[curr_idx - 1].get("text", ""))
            if prev_text:
                clean_prev = self._strip_overlapping_suffix(prev_text, combined)
                if clean_prev:
                    cand = f"{clean_prev} {combined}".strip()
                    cand_tokens = self.token_counter.count_tokens(cand)
                    if cand_tokens <= max_tokens:
                        combined = cand
                        combined_toks = cand_tokens

        if combined != clean_chunk and combined_toks > curr_tokens:
            return PassageExpansionResult(
                doc_id=doc_id,
                chunk_id=chunk_id,
                original_text=clean_chunk,
                expanded_text=combined,
                original_tokens=curr_tokens,
                expanded_tokens=combined_toks,
                is_expanded=True,
                strategy_used="adjacent_chunks",
                reached_max_tokens=combined_toks >= (max_tokens - 15),
            )

        return None

    @staticmethod
    def _find_sentence_starts(text_before: str) -> list[int]:
        """Find valid sentence start indices scanning backwards from end of text_before.

        Ordered from farthest away (start of text) to closest to chunk anchor.
        """
        if not text_before:
            return [0]

        starts = [0]
        for match in SENTENCE_PUNCT_REGEX.finditer(text_before):
            idx = match.end()
            if 0 < idx < len(text_before):
                starts.append(idx)

        # Return in increasing distance order (from farthest left index 0 to closest)
        # To test larger expansions first, sort ascending: 0, s1, s2...
        return sorted(set(starts))

    @staticmethod
    def _find_sentence_ends(text_after: str, offset_base: int) -> list[int]:
        """Find valid sentence end indices scanning forwards from chunk anchor.

        Ordered from farthest away (end of document) to closest to chunk anchor.
        """
        if not text_after:
            return [offset_base]

        ends = [offset_base + len(text_after)]
        for match in SENTENCE_PUNCT_REGEX.finditer(text_after):
            idx = offset_base + match.end()
            ends.append(idx)

        # Return sorted descending (farthest end of document first)
        return sorted(set(ends), reverse=True)

    @staticmethod
    def _strip_overlapping_prefix(lead_text: str, follow_text: str) -> str:
        """Strip redundant sentences from start of follow_text if they already exist in lead_text."""
        words_lead = lead_text.split()
        words_follow = follow_text.split()

        max_overlap_words = min(len(words_lead), len(words_follow), 50)
        best_overlap_len = 0

        for k in range(max_overlap_words, 2, -1):
            if words_lead[-k:] == words_follow[:k]:
                best_overlap_len = k
                break

        if best_overlap_len > 0:
            return " ".join(words_follow[best_overlap_len:]).strip()
        return follow_text

    @staticmethod
    def _strip_overlapping_suffix(lead_text: str, follow_text: str) -> str:
        """Strip redundant sentences from end of lead_text if they already exist in follow_text."""
        words_lead = lead_text.split()
        words_follow = follow_text.split()

        max_overlap_words = min(len(words_lead), len(words_follow), 50)
        best_overlap_len = 0

        for k in range(max_overlap_words, 2, -1):
            if words_lead[-k:] == words_follow[:k]:
                best_overlap_len = k
                break

        if best_overlap_len > 0:
            return " ".join(words_lead[:-best_overlap_len]).strip()
        return lead_text
