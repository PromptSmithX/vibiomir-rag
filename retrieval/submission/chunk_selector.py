"""Intelligent chunk selection and passage expansion engine for ViBioMIR submissions.

Separates chunk candidate filtering, multi-chunk allocation per document,
boilerplate filtering, content deduplication, and optional passage expansion
from submission serialization.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from retrieval.dense.config import ChunkSelectionConfig, PassageExpansionConfig
from retrieval.dense.tokenizer_helper import FallbackTokenCounter
from retrieval.submission.chunk_store import ChunkStore
from retrieval.submission.passage_expander import SourcePassageExpander

logger = logging.getLogger(__name__)

# Known repetitive web chrome, menus, breadcrumbs, footers, and advertisement patterns
BOILERPLATE_PATTERNS = [
    # Breadcrumbs & navigation
    re.compile(r"^(?:trang chủ|home)\s*[>»/|]", re.IGNORECASE),
    re.compile(r"^(?:chuyên mục|danh mục|bài viết liên quan)\s*:", re.IGNORECASE),
    # Social sharing
    re.compile(r"chia sẻ (?:qua|bài viết|trên):?\s*(?:facebook|zalo|twitter|email|linkedin)", re.IGNORECASE),
    # Advertisements & contact
    re.compile(r"(?:liên hệ quảng cáo|hợp tác truyền thông|hotline tư vấn|tổng đài hỗ trợ):?\s*[\d\.\s-]{8,}", re.IGNORECASE),
    # Copyright & legal footers
    re.compile(r"(?:bản quyền thuộc về|all rights reserved|giấy phép hoạt động báo chí|chịu trách nhiệm nội dung)", re.IGNORECASE),
    re.compile(r"©\s*\d{4}", re.IGNORECASE),
    # Cookie & privacy banners
    re.compile(r"website này sử dụng cookie|chính sách bảo mật và điều khoản sử dụng", re.IGNORECASE),
]

FORBIDDEN_PREFIXES = (
    "Instruct:",
    "[Tiêu đề:",
    "[Mục:",
    "Query:",
    "pubmed:",
    "PubMed:",
    "PUBMED:",
)


def strip_synthetic_prefixes(text: str) -> str:
    """Ensure no model prompt instructions, pubmed: or [Tiêu đề: ...] prefixes leak into submission."""
    s = text.strip()
    changed = True
    while changed:
        orig = s
        m_pm = re.match(r"^pubmed(?::\d+|\s*:\s*|\s+)", s, re.IGNORECASE)
        if m_pm:
            s = s[m_pm.end() :].strip()

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

        while s.startswith("[Tiêu đề:") or s.startswith("[Mục:"):
            end_bracket = s.find("]")
            if end_bracket != -1:
                s = s[end_bracket + 1 :].strip()
            else:
                break

        changed = s != orig

    return s.strip()


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for exact and near-duplicate matching."""
    norm = unicodedata.normalize("NFC", text).lower()
    norm = re.sub(r"[^\w\s]", " ", norm)
    return " ".join(norm.split())


def compute_shingle_jaccard(text_a: str, text_b: str, k: int = 3) -> float:
    """Compute word-level k-shingle Jaccard similarity between two texts."""
    words_a = text_a.split()
    words_b = text_b.split()
    if not words_a or not words_b:
        return 0.0

    if len(words_a) < k or len(words_b) < k:
        # Fallback to word-level set Jaccard
        set_a = set(words_a)
        set_b = set(words_b)
        union = set_a | set_b
        return len(set_a & set_b) / len(union) if union else 0.0

    shingles_a = {" ".join(words_a[i : i + k]) for i in range(len(words_a) - k + 1)}
    shingles_b = {" ".join(words_b[i : i + k]) for i in range(len(words_b) - k + 1)}

    union = shingles_a | shingles_b
    if not union:
        return 0.0
    return len(shingles_a & shingles_b) / len(union)


def is_duplicate_chunk(
    candidate_norm: str,
    seen_norms: list[str],
    threshold: float = 0.85,
) -> bool:
    """Check if candidate text is identical or near-duplicate to any already selected text."""
    if not candidate_norm:
        return True

    for prev_norm in seen_norms:
        # 1. Exact match
        if candidate_norm == prev_norm:
            return True
        # 2. Substring containment if high coverage
        if len(candidate_norm) > 50 and len(prev_norm) > 50:
            if candidate_norm in prev_norm or prev_norm in candidate_norm:
                len_ratio = min(len(candidate_norm), len(prev_norm)) / max(len(candidate_norm), len(prev_norm))
                if len_ratio > 0.8:
                    return True
        # 3. Jaccard similarity check
        sim = compute_shingle_jaccard(candidate_norm, prev_norm, k=3)
        if sim >= threshold:
            return True

    return False


def is_boilerplate_chunk(text: str) -> bool:
    """Detect if chunk consists predominantly of website boilerplate, menus, or footer ads.

    Ensures medical and clinical texts are NEVER rejected simply because of brevity
    or unconventional formatting.
    """
    clean = text.strip()
    if not clean:
        return True

    words = clean.split()
    lower_text = clean.lower()

    # Check for obvious repetitive boilerplate patterns
    for pat in BOILERPLATE_PATTERNS:
        match = pat.search(clean)
        if match:
            # If text is relatively short (< 35 words) and contains clear boilerplate markers
            if len(words) < 35 and any(
                marker in lower_text
                for marker in ("bản quyền", "all rights reserved", "trang chủ", "hotline", "chuyên mục", "chia sẻ", "©")
            ):
                return True

            matched_len = len(match.group(0))
            if matched_len / max(len(clean), 1) > 0.3:
                return True
            if pat.pattern.startswith("^") and len(words) < 20:
                return True

    return False



@dataclass
class ChunkSelectionDiagnostics:
    """Diagnostic metrics recorded during chunk selection for a query."""

    query_id: int
    requested_chunks: int
    selected_chunks: int
    unique_docs: int
    chunks_skipped_cap: int
    chunks_skipped_boilerplate: int
    chunks_skipped_dedup: int
    doc_chunk_distribution: dict[int, int] = field(default_factory=dict)


class ChunkSelector:
    """Selects top relevant source chunks for a query from retrieved documents.

    Key capabilities:
    1. Leverages dense chunk candidate matches and similarity scores.
    2. Enforces document provenance: prioritizes chunks from final relevant_docs.
    3. Multi-chunk support: allows up to max_chunks_per_document per doc.
    4. Pragmatic deduplication: avoids redundant passages.
    5. Boilerplate filtering: strips website chrome while preserving medical content.
    6. Optional source passage expansion: expands chunk with continuous neighboring text.
    7. Safe fallback: maintains documents without source chunks in relevant_docs without faking IDs.
    """

    def __init__(self, config: ChunkSelectionConfig | None = None) -> None:
        self.config = config or ChunkSelectionConfig()
        self.token_counter = FallbackTokenCounter()
        self.passage_expander = SourcePassageExpander(
            config=self.config.passage_expansion,
            token_counter=self.token_counter,
        )
        self.last_query_diagnostics: ChunkSelectionDiagnostics | None = None
        self.all_query_diagnostics: list[ChunkSelectionDiagnostics] = []
        self.total_chunks_skipped_cap: int = 0

    def reset_diagnostics(self) -> None:
        """Reset diagnostics trackers across all queries."""
        self.last_query_diagnostics = None
        self.all_query_diagnostics = []
        self.total_chunks_skipped_cap = 0
        self.passage_expander.reset_history()

    def get_diagnostics_summary(self) -> dict[str, Any]:
        """Aggregate diagnostics statistics across all queries processed."""
        if not self.all_query_diagnostics:
            return {
                "total_queries": 0,
                "total_chunks_selected": 0,
                "total_chunks_skipped_cap": 0,
                "total_chunks_skipped_boilerplate": 0,
                "total_chunks_skipped_dedup": 0,
                "queries_under_target": [],
                "queries_under_target_count": 0,
                "total_deficit": 0,
                "docs_with_1_chunk": 0,
                "docs_with_2_chunks": 0,
                "docs_with_3_chunks": 0,
                "docs_with_gt_2_chunks": 0,
                "avg_unique_docs": 0.0,
                "median_unique_docs": 0.0,
            }

        total_queries = len(self.all_query_diagnostics)
        total_chunks = sum(d.selected_chunks for d in self.all_query_diagnostics)
        total_cap_skipped = sum(d.chunks_skipped_cap for d in self.all_query_diagnostics)
        total_bp_skipped = sum(d.chunks_skipped_boilerplate for d in self.all_query_diagnostics)
        total_dup_skipped = sum(d.chunks_skipped_dedup for d in self.all_query_diagnostics)

        under_target = [
            d.query_id for d in self.all_query_diagnostics if d.selected_chunks < d.requested_chunks
        ]
        total_deficit = sum(
            max(0, d.requested_chunks - d.selected_chunks) for d in self.all_query_diagnostics
        )

        docs_1 = 0
        docs_2 = 0
        docs_3 = 0
        docs_gt_2 = 0
        for d in self.all_query_diagnostics:
            for cnt in d.doc_chunk_distribution.values():
                if cnt == 1:
                    docs_1 += 1
                elif cnt == 2:
                    docs_2 += 1
                elif cnt == 3:
                    docs_3 += 1
                if cnt > 2:
                    docs_gt_2 += 1

        unique_docs_list = [d.unique_docs for d in self.all_query_diagnostics]
        unique_docs_list.sort()
        avg_unique = sum(unique_docs_list) / total_queries if total_queries > 0 else 0.0
        mid = total_queries // 2
        median_unique = (
            (unique_docs_list[mid] + unique_docs_list[mid - 1]) / 2.0
            if total_queries % 2 == 0 and total_queries > 0
            else float(unique_docs_list[mid])
            if total_queries > 0
            else 0.0
        )

        return {
            "total_queries": total_queries,
            "total_chunks_selected": total_chunks,
            "total_chunks_skipped_cap": total_cap_skipped,
            "total_chunks_skipped_boilerplate": total_bp_skipped,
            "total_chunks_skipped_dedup": total_dup_skipped,
            "queries_under_target": under_target,
            "queries_under_target_count": len(under_target),
            "total_deficit": total_deficit,
            "docs_with_1_chunk": docs_1,
            "docs_with_2_chunks": docs_2,
            "docs_with_3_chunks": docs_3,
            "docs_with_gt_2_chunks": docs_gt_2,
            "avg_unique_docs": round(avg_unique, 2),
            "median_unique_docs": round(median_unique, 2),
        }

    def select_chunks(
        self,
        query_id: int,
        relevant_docs: list[int],
        chunk_store: ChunkStore,
        dense_chunk_hits: list[dict[str, Any]] | None = None,
        dense_best_chunk_map: dict[int, str] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Select top relevant chunks for a single query.

        Args:
            query_id: Official query ID.
            relevant_docs: Ranked list of integer doc_ids for this query.
            chunk_store: ChunkStore with preloaded chunk shards and fallback text.
            dense_chunk_hits: Optional list of dense chunk hits for this query,
                each containing {"chunk_id": str, "doc_id": int, "score": float}.
            dense_best_chunk_map: Optional mapping of doc_id -> best_chunk_id.
            top_k: Optional override for maximum chunks to return.

        Returns:
            List of dicts: [{"doc_id": int, "chunk_text": str}, ...]
        """
        limit_chunks = top_k if top_k is not None else self.config.top_k
        max_per_doc = self.config.max_chunks_per_document

        if not relevant_docs or limit_chunks <= 0:
            return []

        rel_doc_set = set(relevant_docs)
        doc_rank_order = {doc_id: idx for idx, doc_id in enumerate(relevant_docs)}

        # 1. Gather all chunk candidates belonging to relevant_docs
        # Track doc_id -> list of candidate chunk dicts
        candidates: list[dict[str, Any]] = []
        seen_cand_cids: set[str] = set()

        # Priority A: Dense chunk hits matching relevant_docs (sorted by score descending)
        if dense_chunk_hits:
            for hit in dense_chunk_hits:
                cid = hit.get("chunk_id", "")
                did = hit.get("doc_id")
                if not cid or did not in rel_doc_set:
                    continue
                if cid in seen_cand_cids:
                    continue
                seen_cand_cids.add(cid)
                candidates.append(
                    {
                        "chunk_id": cid,
                        "doc_id": int(did),
                        "score": float(hit.get("score", 0.0)),
                        "source": "dense_hit",
                        "doc_rank": doc_rank_order.get(int(did), 999999),
                    }
                )

        # Priority B: Best chunk from document-level aggregation if not yet captured
        if dense_best_chunk_map:
            for did in relevant_docs:
                best_cid = dense_best_chunk_map.get(did)
                if best_cid and best_cid not in seen_cand_cids:
                    seen_cand_cids.add(best_cid)
                    candidates.append(
                        {
                            "chunk_id": best_cid,
                            "doc_id": int(did),
                            "score": 0.5,  # Nominal fallback score
                            "source": "best_chunk",
                            "doc_rank": doc_rank_order.get(int(did), 999999),
                        }
                    )

        # Priority C: Additional chunks for relevant_docs from ChunkStore
        for did in relevant_docs:
            doc_chunks = chunk_store.get_chunks_for_doc(did)
            for c_info in doc_chunks:
                cid = c_info.get("chunk_id", "")
                if cid and cid not in seen_cand_cids:
                    seen_cand_cids.add(cid)
                    candidates.append(
                        {
                            "chunk_id": cid,
                            "doc_id": int(did),
                            "chunk_index": c_info.get("chunk_index", 0),
                            "score": 0.0,
                            "source": "store",
                            "doc_rank": doc_rank_order.get(int(did), 999999),
                        }
                    )

        # Sort candidates:
        # Dense hits with high similarity come first, weighted by their doc rank
        def candidate_sort_key(item: dict[str, Any]) -> tuple[int, float, int, str]:
            # doc_rank ascending (higher ranked doc first), then score descending, chunk_index, chunk_id
            return (
                item["doc_rank"],
                -item["score"],
                item.get("chunk_index", 0),
                str(item.get("chunk_id", "")),
            )

        candidates.sort(key=candidate_sort_key)

        # 2. Select chunks respecting quotas, deduplication, and quality filters
        selected_chunks: list[dict[str, Any]] = []
        doc_chunk_counts: dict[int, int] = defaultdict(int)
        seen_selected_cids: set[str] = set()
        seen_norm_texts: list[str] = []

        chunks_skipped_cap = 0
        chunks_skipped_boilerplate = 0
        chunks_skipped_dedup = 0

        for cand in candidates:
            if len(selected_chunks) >= limit_chunks:
                break

            did = cand["doc_id"]
            cid = cand["chunk_id"]

            if cid in seen_selected_cids:
                continue
            if doc_chunk_counts[did] >= max_per_doc:
                chunks_skipped_cap += 1
                continue

            # Resolve chunk text
            chunk_meta = chunk_store.get_chunk_meta(cid)
            raw_text = chunk_meta.get("text", "") if chunk_meta else chunk_store.get_chunk_text(cid)
            if not raw_text:
                continue

            clean_text = strip_synthetic_prefixes(raw_text)
            if not clean_text:
                continue

            # Length check
            if len(clean_text) < self.config.min_chunk_chars:
                # If short, only keep if doc has no chunks selected yet
                if doc_chunk_counts[did] > 0:
                    continue

            # Boilerplate filtering
            if self.config.filter_boilerplate and is_boilerplate_chunk(clean_text):
                chunks_skipped_boilerplate += 1
                logger.debug("Filtered boilerplate chunk %s for query %d", cid, query_id)
                continue

            # Deduplication
            norm_text = normalize_text_for_dedup(clean_text)
            if self.config.deduplicate and is_duplicate_chunk(
                norm_text, seen_norm_texts, threshold=self.config.dedup_threshold
            ):
                chunks_skipped_dedup += 1
                logger.debug("Deduplicated redundant chunk %s for query %d", cid, query_id)
                continue

            # Optional passage expansion
            final_text = clean_text
            if self.config.passage_expansion.enabled:
                final_text = self._expand_passage(
                    doc_id=did,
                    chunk_id=cid,
                    chunk_text=clean_text,
                    chunk_meta=chunk_meta or {},
                    chunk_store=chunk_store,
                )

            selected_chunks.append(
                {
                    "doc_id": int(did),
                    "chunk_text": final_text,
                    "chunk_id": cid,
                }
            )
            seen_selected_cids.add(cid)
            seen_norm_texts.append(norm_text)
            doc_chunk_counts[did] += 1

        # 3. Fallback: If still under limit_chunks, ensure documents have at least one chunk
        if len(selected_chunks) < limit_chunks:
            for did in relevant_docs:
                if len(selected_chunks) >= limit_chunks:
                    break
                if doc_chunk_counts[did] >= max_per_doc:
                    continue
                if doc_chunk_counts[did] > 0:
                    continue

                # Try fallback first-chunk
                fb = chunk_store.get_fallback_chunk_for_doc(did)
                if fb:
                    fb_cid, fb_text = fb
                    if fb_cid not in seen_selected_cids:
                        clean_fb = strip_synthetic_prefixes(fb_text)
                        if clean_fb:
                            selected_chunks.append(
                                {
                                    "doc_id": int(did),
                                    "chunk_text": clean_fb,
                                    "chunk_id": fb_cid,
                                }
                            )
                            seen_selected_cids.add(fb_cid)
                            doc_chunk_counts[did] += 1
                            continue

                # Try corpus raw text
                corpus_text = chunk_store.get_text_from_corpus(did)
                if corpus_text:
                    clean_c = strip_synthetic_prefixes(corpus_text)
                    if clean_c:
                        # Truncate to reasonable chunk size
                        truncated_c = self._truncate_text_to_tokens(clean_c, max_tokens=400)
                        selected_chunks.append(
                            {
                                "doc_id": int(did),
                                "chunk_text": truncated_c,
                                "chunk_id": f"{did}_corpus_fb",
                            }
                        )
                        doc_chunk_counts[did] += 1

        # Record diagnostics for this query
        diag = ChunkSelectionDiagnostics(
            query_id=query_id,
            requested_chunks=limit_chunks,
            selected_chunks=len(selected_chunks),
            unique_docs=len(doc_chunk_counts),
            chunks_skipped_cap=chunks_skipped_cap,
            chunks_skipped_boilerplate=chunks_skipped_boilerplate,
            chunks_skipped_dedup=chunks_skipped_dedup,
            doc_chunk_distribution=dict(doc_chunk_counts),
        )
        self.last_query_diagnostics = diag
        self.all_query_diagnostics.append(diag)
        self.total_chunks_skipped_cap += chunks_skipped_cap

        return selected_chunks

    def _expand_passage(
        self,
        doc_id: int,
        chunk_id: str,
        chunk_text: str,
        chunk_meta: dict[str, Any],
        chunk_store: ChunkStore,
    ) -> str:
        """Safely expand chunk text with continuous neighboring context from the same document."""
        res = self.passage_expander.expand_chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            chunk_meta=chunk_meta,
            chunk_store=chunk_store,
        )
        return res.expanded_text

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit max_tokens budget cleanly."""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])
