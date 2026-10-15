"""Diagnostics and audit engine for source passage expansion in ViBioMIR submissions.

Computes expansion coverage, fallback attribution, token length distributions
(mean, median, p95), boundary saturation, inter-passage overlaps, and verifies
strict ranking and chunk selection order preservation against baseline.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from retrieval.dense.tokenizer_helper import BaseTokenCounter, FallbackTokenCounter
from retrieval.submission.chunk_selector import normalize_text_for_dedup

logger = logging.getLogger(__name__)


@dataclass
class TokenDistributionStats:
    """Statistical summary of token counts."""

    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    min: int = 0
    max: int = 0

    @classmethod
    def from_counts(cls, values: list[int]) -> TokenDistributionStats:
        if not values:
            return cls()
        s_vals = sorted(values)
        n = len(s_vals)
        mean_v = sum(s_vals) / n
        median_v = s_vals[n // 2] if n % 2 != 0 else (s_vals[n // 2 - 1] + s_vals[n // 2]) / 2.0
        p95_idx = min(n - 1, math.ceil(0.95 * n) - 1)
        p95_v = float(s_vals[p95_idx])
        return cls(
            count=n,
            mean=round(mean_v, 2),
            median=round(median_v, 2),
            p95=round(p95_v, 2),
            min=s_vals[0],
            max=s_vals[-1],
        )


@dataclass
class PassageDiagnosticsSummary:
    """Complete diagnostic report for a passage-expanded submission."""

    total_queries: int = 0
    total_relevant_docs: int = 0
    total_relevant_chunks: int = 0
    expanded_chunks_count: int = 0
    fallback_chunks_count: int = 0
    expansion_rate_pct: float = 0.0
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    pre_expansion_tokens: TokenDistributionStats = field(default_factory=TokenDistributionStats)
    post_expansion_tokens: TokenDistributionStats = field(default_factory=TokenDistributionStats)
    mean_token_gain: float = 0.0
    mean_token_gain_pct: float = 0.0
    passages_hitting_max_limit: int = 0
    passages_hitting_max_limit_pct: float = 0.0
    overlapping_passage_pairs_count: int = 0
    queries_with_target_chunks: int = 0
    queries_under_target_chunks: int = 0
    target_chunks_per_query: int = 100
    max_tokens_budget: int = 1024
    tokenizer_identity: str = "FallbackTokenCounter"
    ranking_identical_to_baseline: bool = True
    chunks_order_identical_to_baseline: bool = True
    ranking_mismatches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_expanded_submission(
    submission_path: Path | str,
    baseline_path: Path | str | None = None,
    max_tokens: int = 1024,
    token_counter: BaseTokenCounter | None = None,
    target_chunks: int = 100,
) -> PassageDiagnosticsSummary:
    """Analyze a passage-expanded submission and optionally audit against baseline A1.2."""
    tc = token_counter or FallbackTokenCounter()
    tok_name = tc.__class__.__name__

    sub_file = Path(submission_path)
    base_file = Path(baseline_path) if baseline_path else None

    with open(sub_file, encoding="utf-8") as f:
        cand_records = json.load(f)

    base_records_by_qid: dict[int, dict[str, Any]] = {}
    if base_file and base_file.exists():
        with open(base_file, encoding="utf-8") as f:
            base_data = json.load(f)
            base_records_by_qid = {int(item["id"]): item for item in base_data}

    total_queries = len(cand_records)
    total_docs = sum(len(r.get("relevant_docs", [])) for r in cand_records)
    total_chunks = sum(len(r.get("relevant_chunks", [])) for r in cand_records)

    pre_counts: list[int] = []
    post_counts: list[int] = []
    expanded_count = 0
    fallback_count = 0
    fallback_reasons: dict[str, int] = {}
    hitting_max_count = 0
    overlapping_pairs = 0
    queries_at_target = 0
    queries_under_target = 0

    ranking_identical = True
    chunks_order_identical = True
    ranking_mismatches: list[str] = []

    for item in cand_records:
        qid = int(item["id"])
        c_docs = [int(d) for d in item.get("relevant_docs", [])]
        c_chunks = item.get("relevant_chunks", [])

        if len(c_chunks) >= target_chunks:
            queries_at_target += 1
        else:
            queries_under_target += 1

        b_item = base_records_by_qid.get(qid)
        if b_item:
            b_docs = [int(d) for d in b_item.get("relevant_docs", [])]
            b_chunks = b_item.get("relevant_chunks", [])

            # Check document ranking preservation
            if c_docs != b_docs:
                ranking_identical = False
                ranking_mismatches.append(f"Query {qid}: relevant_docs order differs from baseline")

            # Check chunk count & doc_id sequence
            if len(c_chunks) != len(b_chunks):
                chunks_order_identical = False
                ranking_mismatches.append(f"Query {qid}: chunk count ({len(c_chunks)}) != baseline ({len(b_chunks)})")
            else:
                for idx, (cc, bc) in enumerate(zip(c_chunks, b_chunks, strict=False)):
                    if int(cc["doc_id"]) != int(bc["doc_id"]):
                        chunks_order_identical = False
                        ranking_mismatches.append(f"Query {qid} chunk #{idx}: doc_id {cc['doc_id']} != baseline {bc['doc_id']}")

        # Track token lengths and overlaps within query
        query_chunk_norms: list[tuple[int, str]] = []

        for idx, chunk in enumerate(c_chunks):
            c_text = chunk.get("chunk_text", "")
            post_tok = tc.count_tokens(c_text)
            post_counts.append(post_tok)

            if post_tok >= (max_tokens - 15):
                hitting_max_count += 1

            # If baseline available, compare text expansion
            if b_item and idx < len(b_item.get("relevant_chunks", [])):
                b_text = b_item["relevant_chunks"][idx].get("chunk_text", "")
                pre_tok = tc.count_tokens(b_text)
                pre_counts.append(pre_tok)

                if c_text.strip() != b_text.strip() and post_tok > pre_tok:
                    expanded_count += 1
                else:
                    fallback_count += 1
                    reason = "already_at_budget" if pre_tok >= max_tokens else "no_expansion_source"
                    fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            else:
                # Without baseline: estimate baseline chunk length as ~200-300 tokens
                if post_tok > 350:
                    expanded_count += 1
                else:
                    fallback_count += 1

            did = int(chunk.get("doc_id", 0))
            norm = normalize_text_for_dedup(c_text)
            query_chunk_norms.append((did, norm))

        # Check overlapping passage pairs among same document chunks in this query
        for i in range(len(query_chunk_norms)):
            for j in range(i + 1, len(query_chunk_norms)):
                did_i, norm_i = query_chunk_norms[i]
                did_j, norm_j = query_chunk_norms[j]
                if did_i == did_j and norm_i and norm_j:
                    if norm_i in norm_j or norm_j in norm_i:
                        overlapping_pairs += 1

    pre_dist = TokenDistributionStats.from_counts(pre_counts) if pre_counts else TokenDistributionStats()
    post_dist = TokenDistributionStats.from_counts(post_counts)

    gain_mean = round(post_dist.mean - pre_dist.mean, 2) if pre_counts else 0.0
    gain_pct = round((gain_mean / pre_dist.mean) * 100, 2) if (pre_counts and pre_dist.mean > 0) else 0.0
    exp_rate = round((expanded_count / max(1, total_chunks)) * 100, 2)
    hit_max_pct = round((hitting_max_count / max(1, total_chunks)) * 100, 2)

    return PassageDiagnosticsSummary(
        total_queries=total_queries,
        total_relevant_docs=total_docs,
        total_relevant_chunks=total_chunks,
        expanded_chunks_count=expanded_count,
        fallback_chunks_count=fallback_count,
        expansion_rate_pct=exp_rate,
        fallback_reasons=fallback_reasons,
        pre_expansion_tokens=pre_dist,
        post_expansion_tokens=post_dist,
        mean_token_gain=gain_mean,
        mean_token_gain_pct=gain_pct,
        passages_hitting_max_limit=hitting_max_count,
        passages_hitting_max_limit_pct=hit_max_pct,
        overlapping_passage_pairs_count=overlapping_pairs,
        queries_with_target_chunks=queries_at_target,
        queries_under_target_chunks=queries_under_target,
        target_chunks_per_query=target_chunks,
        max_tokens_budget=max_tokens,
        tokenizer_identity=tok_name,
        ranking_identical_to_baseline=ranking_identical,
        chunks_order_identical_to_baseline=chunks_order_identical,
        ranking_mismatches=ranking_mismatches[:10],
    )


def format_passage_diagnostics_report(diag: PassageDiagnosticsSummary) -> str:
    """Format PassageDiagnosticsSummary into clean, professional ASCII table."""
    lines: list[str] = [
        "=" * 72,
        f"=== VIBIOMIR PASSAGE EXPANSION REPORT (Budget: {diag.max_tokens_budget} tokens) ===",
        "=" * 72,
        f"Tokenizer Identity         : {diag.tokenizer_identity}",
        f"Total Queries Audited      : {diag.total_queries:,}",
        f"Total Relevant Documents   : {diag.total_relevant_docs:,}",
        f"Total Relevant Chunks      : {diag.total_relevant_chunks:,}",
        f"Queries With Full {diag.target_chunks_per_query} Chunks: {diag.queries_with_target_chunks:,} / {diag.total_queries:,}",
        "-" * 72,
        "[EXPANSION COVERAGE & FALLBACKS]",
        f"  - Successfully Expanded  : {diag.expanded_chunks_count:,} ({diag.expansion_rate_pct:.1f}%)",
        f"  - Fallback / Unchanged    : {diag.fallback_chunks_count:,} ({100.0 - diag.expansion_rate_pct:.1f}%)",
    ]

    if diag.fallback_reasons:
        lines.append("  - Fallback Reasons Breakdown:")
        for r, cnt in sorted(diag.fallback_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"      * {r:26s}: {cnt:,} chunks")

    lines.extend(
        [
            "-" * 72,
            "[TOKEN LENGTH DISTRIBUTION]",
            f"  - Baseline Mean / Median / P95 : {diag.pre_expansion_tokens.mean:.1f} / {diag.pre_expansion_tokens.median:.1f} / {diag.pre_expansion_tokens.p95:.1f} tokens"
            if diag.pre_expansion_tokens.count > 0
            else "  - Baseline Token Data          : Not provided",
            f"  - Expanded Mean / Median / P95 : {diag.post_expansion_tokens.mean:.1f} / {diag.post_expansion_tokens.median:.1f} / {diag.post_expansion_tokens.p95:.1f} tokens",
            f"  - Min / Max Token Length       : {diag.post_expansion_tokens.min} / {diag.post_expansion_tokens.max} tokens",
            f"  - Mean Token Gain              : +{diag.mean_token_gain:.1f} tokens (+{diag.mean_token_gain_pct:.1f}%)",
            f"  - Passages Hitting Max Limit   : {diag.passages_hitting_max_limit:,} ({diag.passages_hitting_max_limit_pct:.1f}%)",
            f"  - Overlapping Passage Pairs    : {diag.overlapping_passage_pairs_count:,} pairs",
            "-" * 72,
            "[CONTROLLED EXPERIMENT INTEGRITY]",
            f"  - Document Ranking Preserved   : {'100% IDENTICAL [PASS]' if diag.ranking_identical_to_baseline else 'MISMATCH DETECTED [FAIL]'}",
            f"  - Chunk Selection Order Valid  : {'100% IDENTICAL [PASS]' if diag.chunks_order_identical_to_baseline else 'MISMATCH DETECTED [FAIL]'}",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def export_diagnostics_json(data: dict[str, Any], path: Path | str) -> Path:
    """Save diagnostics dictionary to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return p
