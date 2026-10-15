"""Submission diagnostics, coverage statistics, and baseline comparison engine.

Provides auditing tools to evaluate chunk-to-document coverage, verify document
ranking preservation across experiments, detect cap violations, and compute
comparative statistics between Baseline A1 (cap 3) and Experiment A1.2 (cap 2).
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _compute_median(values: list[float | int]) -> float:
    """Compute exact median of a numeric list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def _load_submission_data(target_path: Path | str) -> list[dict[str, Any]]:
    """Load submission records from either a .json file or a .zip archive."""
    p = Path(target_path)
    if not p.exists():
        raise FileNotFoundError(f"Submission file does not exist: {p.resolve()}")

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            infolist = [f for f in zf.infolist() if f.filename.lower().endswith(".json")]
            if not infolist:
                raise ValueError(f"No JSON file found inside ZIP archive: {p.name}")
            with zf.open(infolist[0]) as jf:
                return json.load(jf)
    elif p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported submission format '{p.suffix}'. Expected .json or .zip")


def _load_optional_metadata(submission_path: Path | str) -> dict[str, Any] | None:
    """Load accompanying submission_metadata.json if present in the same directory."""
    p = Path(submission_path)
    meta_path = p.parent / "submission_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@dataclass
class QueryDiagnostics:
    """Metrics for an individual query's submission output."""

    query_id: int
    num_docs: int
    num_chunks: int
    num_unique_docs_in_chunks: int
    doc_coverage_ratio: float
    chunks_per_doc_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class SubmissionDiagnostics:
    """Comprehensive diagnostic metrics for a single submission artifact."""

    submission_path: str
    total_queries: int
    total_relevant_docs: int
    total_relevant_chunks: int
    target_docs: int
    target_chunks: int
    max_chunks_per_doc_limit: int

    # Chunks per query
    min_chunks: int
    max_chunks: int
    mean_chunks: float
    median_chunks: float

    # Unique docs represented in chunks per query
    min_unique_docs: int
    max_unique_docs: int
    mean_unique_docs: float
    median_unique_docs: float

    # Document coverage ratio
    mean_coverage_ratio: float
    median_coverage_ratio: float

    # Distribution of chunks per document (across all queries)
    docs_with_1_chunk: int
    docs_with_2_chunks: int
    docs_with_3_chunks: int
    docs_with_gt_2_chunks: int
    docs_with_gt_3_chunks: int
    pct_docs_with_1_chunk: float
    pct_docs_with_2_chunks: float
    pct_docs_with_3_chunks: float

    # Capacity shortfall
    queries_under_target: list[int]
    queries_under_target_count: int
    total_chunk_shortfall: int

    # Cap violations
    cap_violations: list[dict[str, Any]]
    cap_violations_count: int

    # Metadata metrics (if submission_metadata.json found)
    chunks_skipped_cap_count: int | None = None
    metadata_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonReport:
    """Detailed comparative evaluation between baseline and candidate experiments."""

    baseline_path: str
    candidate_path: str
    target_docs: int
    target_chunks: int

    # Document ranking preservation
    ranking_identical: bool
    total_queries_compared: int
    mismatched_queries: list[dict[str, Any]]

    # Coverage metrics comparison
    baseline_mean_unique_docs: float
    candidate_mean_unique_docs: float
    mean_unique_docs_delta: float
    mean_unique_docs_pct_gain: float

    baseline_median_unique_docs: float
    candidate_median_unique_docs: float
    median_unique_docs_delta: float

    # Distribution shift
    baseline_docs_1: int
    candidate_docs_1: int
    baseline_docs_2: int
    candidate_docs_2: int
    baseline_docs_3: int
    candidate_docs_3: int

    # Queries breakdown
    queries_coverage_increased: int
    queries_coverage_same: int
    queries_coverage_decreased: int

    # Diagnostic objects
    baseline_diagnostics: SubmissionDiagnostics
    candidate_diagnostics: SubmissionDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_submission(
    submission_path: Path | str,
    target_docs: int = 100,
    target_chunks: int = 100,
    max_chunks_per_doc: int = 2,
) -> SubmissionDiagnostics:
    """Analyze a submission file and compute comprehensive coverage and distribution metrics."""
    records = _load_submission_data(submission_path)
    meta = _load_optional_metadata(submission_path)

    total_queries = len(records)
    total_docs = 0
    total_chunks = 0

    chunks_per_query: list[int] = []
    unique_docs_per_query: list[int] = []
    coverage_ratios: list[float] = []

    docs_with_1 = 0
    docs_with_2 = 0
    docs_with_3 = 0
    docs_with_gt_2 = 0
    docs_with_gt_3 = 0

    queries_under_target: list[int] = []
    total_shortfall = 0
    cap_violations: list[dict[str, Any]] = []

    for item in records:
        q_id = item.get("id", -1)
        rel_docs = item.get("relevant_docs", [])
        rel_chunks = item.get("relevant_chunks", [])

        n_docs = len(rel_docs)
        n_chunks = len(rel_chunks)
        total_docs += n_docs
        total_chunks += n_chunks

        chunks_per_query.append(n_chunks)
        if n_chunks < target_chunks:
            queries_under_target.append(q_id)
            total_shortfall += (target_chunks - n_chunks)

        # Count occurrences of doc_id in chunks
        chunk_docs: dict[int, int] = {}
        for c in rel_chunks:
            did = c.get("doc_id")
            if did is not None:
                chunk_docs[did] = chunk_docs.get(did, 0) + 1

        n_unique_docs = len(chunk_docs)
        unique_docs_per_query.append(n_unique_docs)
        cov_ratio = (n_unique_docs / max(1, n_docs)) if n_docs > 0 else 0.0
        coverage_ratios.append(cov_ratio)

        for did, count in chunk_docs.items():
            if count == 1:
                docs_with_1 += 1
            elif count == 2:
                docs_with_2 += 1
            elif count == 3:
                docs_with_3 += 1

            if count > 2:
                docs_with_gt_2 += 1
            if count > 3:
                docs_with_gt_3 += 1

            if count > max_chunks_per_doc:
                cap_violations.append(
                    {
                        "query_id": q_id,
                        "doc_id": did,
                        "chunk_count": count,
                        "cap_limit": max_chunks_per_doc,
                    }
                )

    total_doc_allocations = docs_with_1 + docs_with_2 + docs_with_3 + (docs_with_gt_3)
    pct_1 = (docs_with_1 / total_doc_allocations * 100.0) if total_doc_allocations > 0 else 0.0
    pct_2 = (docs_with_2 / total_doc_allocations * 100.0) if total_doc_allocations > 0 else 0.0
    pct_3 = (docs_with_3 / total_doc_allocations * 100.0) if total_doc_allocations > 0 else 0.0

    mean_ch = sum(chunks_per_query) / total_queries if total_queries > 0 else 0.0
    mean_unq = sum(unique_docs_per_query) / total_queries if total_queries > 0 else 0.0
    mean_cov = sum(coverage_ratios) / total_queries if total_queries > 0 else 0.0

    skipped_cap = None
    if meta and "diagnostics" in meta:
        skipped_cap = meta["diagnostics"].get("chunks_skipped_cap")

    return SubmissionDiagnostics(
        submission_path=str(Path(submission_path).resolve()),
        total_queries=total_queries,
        total_relevant_docs=total_docs,
        total_relevant_chunks=total_chunks,
        target_docs=target_docs,
        target_chunks=target_chunks,
        max_chunks_per_doc_limit=max_chunks_per_doc,
        min_chunks=min(chunks_per_query) if chunks_per_query else 0,
        max_chunks=max(chunks_per_query) if chunks_per_query else 0,
        mean_chunks=round(mean_ch, 2),
        median_chunks=_compute_median(chunks_per_query),
        min_unique_docs=min(unique_docs_per_query) if unique_docs_per_query else 0,
        max_unique_docs=max(unique_docs_per_query) if unique_docs_per_query else 0,
        mean_unique_docs=round(mean_unq, 2),
        median_unique_docs=_compute_median(unique_docs_per_query),
        mean_coverage_ratio=round(mean_cov, 4),
        median_coverage_ratio=round(_compute_median(coverage_ratios), 4),
        docs_with_1_chunk=docs_with_1,
        docs_with_2_chunks=docs_with_2,
        docs_with_3_chunks=docs_with_3,
        docs_with_gt_2_chunks=docs_with_gt_2,
        docs_with_gt_3_chunks=docs_with_gt_3,
        pct_docs_with_1_chunk=round(pct_1, 2),
        pct_docs_with_2_chunks=round(pct_2, 2),
        pct_docs_with_3_chunks=round(pct_3, 2),
        queries_under_target=queries_under_target,
        queries_under_target_count=len(queries_under_target),
        total_chunk_shortfall=total_shortfall,
        cap_violations=cap_violations,
        cap_violations_count=len(cap_violations),
        chunks_skipped_cap_count=skipped_cap,
        metadata_context=meta,
    )


def verify_ranking_preservation(
    baseline_path: Path | str,
    candidate_path: Path | str,
) -> tuple[bool, list[dict[str, Any]]]:
    """Verify that candidate relevant_docs ranking strictly matches baseline relevant_docs.

    Returns:
        (is_identical, mismatches_list)
    """
    baseline_data = _load_submission_data(baseline_path)
    candidate_data = _load_submission_data(candidate_path)

    base_map = {item["id"]: item.get("relevant_docs", []) for item in baseline_data}
    cand_map = {item["id"]: item.get("relevant_docs", []) for item in candidate_data}

    mismatches: list[dict[str, Any]] = []

    # Check query IDs match
    if set(base_map.keys()) != set(cand_map.keys()):
        missing_in_cand = sorted(list(set(base_map.keys()) - set(cand_map.keys())))
        extra_in_cand = sorted(list(set(cand_map.keys()) - set(base_map.keys())))
        mismatches.append(
            {
                "type": "query_id_mismatch",
                "missing_in_candidate": missing_in_cand[:10],
                "extra_in_candidate": extra_in_cand[:10],
            }
        )

    for q_id, b_docs in base_map.items():
        if q_id not in cand_map:
            continue
        c_docs = cand_map[q_id]

        if b_docs != c_docs:
            mismatches.append(
                {
                    "query_id": q_id,
                    "baseline_count": len(b_docs),
                    "candidate_count": len(c_docs),
                    "baseline_sample": b_docs[:5],
                    "candidate_sample": c_docs[:5],
                }
            )

    return len(mismatches) == 0, mismatches


def compare_submissions(
    baseline_path: Path | str,
    candidate_path: Path | str,
    target_docs: int = 100,
    target_chunks: int = 100,
) -> ComparisonReport:
    """Perform side-by-side comparative analysis between baseline and candidate submissions."""
    ranking_ok, mismatches = verify_ranking_preservation(baseline_path, candidate_path)

    base_diag = analyze_submission(
        baseline_path,
        target_docs=target_docs,
        target_chunks=target_chunks,
        max_chunks_per_doc=3,
    )
    cand_diag = analyze_submission(
        candidate_path,
        target_docs=target_docs,
        target_chunks=target_chunks,
        max_chunks_per_doc=2,
    )

    base_data = _load_submission_data(baseline_path)
    cand_data = _load_submission_data(candidate_path)
    base_unq_map = {
        item["id"]: len({c["doc_id"] for c in item.get("relevant_chunks", []) if "doc_id" in c})
        for item in base_data
    }
    cand_unq_map = {
        item["id"]: len({c["doc_id"] for c in item.get("relevant_chunks", []) if "doc_id" in c})
        for item in cand_data
    }

    increased = 0
    same = 0
    decreased = 0
    for q_id in base_unq_map:
        if q_id in cand_unq_map:
            b_u = base_unq_map[q_id]
            c_u = cand_unq_map[q_id]
            if c_u > b_u:
                increased += 1
            elif c_u == b_u:
                same += 1
            else:
                decreased += 1

    delta_mean = round(cand_diag.mean_unique_docs - base_diag.mean_unique_docs, 2)
    pct_gain = (
        round((delta_mean / base_diag.mean_unique_docs) * 100.0, 2)
        if base_diag.mean_unique_docs > 0
        else 0.0
    )
    delta_med = round(cand_diag.median_unique_docs - base_diag.median_unique_docs, 2)

    return ComparisonReport(
        baseline_path=str(Path(baseline_path).resolve()),
        candidate_path=str(Path(candidate_path).resolve()),
        target_docs=target_docs,
        target_chunks=target_chunks,
        ranking_identical=ranking_ok,
        total_queries_compared=min(len(base_data), len(cand_data)),
        mismatched_queries=mismatches,
        baseline_mean_unique_docs=base_diag.mean_unique_docs,
        candidate_mean_unique_docs=cand_diag.mean_unique_docs,
        mean_unique_docs_delta=delta_mean,
        mean_unique_docs_pct_gain=pct_gain,
        baseline_median_unique_docs=base_diag.median_unique_docs,
        candidate_median_unique_docs=cand_diag.median_unique_docs,
        median_unique_docs_delta=delta_med,
        baseline_docs_1=base_diag.docs_with_1_chunk,
        candidate_docs_1=cand_diag.docs_with_1_chunk,
        baseline_docs_2=base_diag.docs_with_2_chunks,
        candidate_docs_2=cand_diag.docs_with_2_chunks,
        baseline_docs_3=base_diag.docs_with_3_chunks,
        candidate_docs_3=cand_diag.docs_with_3_chunks,
        queries_coverage_increased=increased,
        queries_coverage_same=same,
        queries_coverage_decreased=decreased,
        baseline_diagnostics=base_diag,
        candidate_diagnostics=cand_diag,
    )


def format_diagnostics_report(diag: SubmissionDiagnostics) -> str:
    """Format single submission diagnostics into a clean human-readable report."""
    lines: list[str] = []
    lines.append("=" * 76)
    lines.append("=== VIBIOMIR SUBMISSION CHUNK COVERAGE DIAGNOSTICS ===")
    lines.append("=" * 76)
    lines.append(f"Submission Path          : {diag.submission_path}")
    lines.append(f"Total Official Queries   : {diag.total_queries:,}")
    lines.append(f"Total Relevant Documents : {diag.total_relevant_docs:,}")
    lines.append(f"Total Relevant Chunks    : {diag.total_relevant_chunks:,}")
    lines.append(f"Configured Limits        : top_docs={diag.target_docs}, top_chunks={diag.target_chunks}, max_per_doc={diag.max_chunks_per_doc_limit}")
    lines.append("-" * 76)
    lines.append("1. DOCUMENT COVERAGE IN RELEVANT CHUNKS:")
    lines.append(f"  - Unique docs / query  : Mean={diag.mean_unique_docs:.2f} | Median={diag.median_unique_docs:.1f} | Min={diag.min_unique_docs} | Max={diag.max_unique_docs}")
    lines.append(f"  - Document coverage %  : Mean={diag.mean_coverage_ratio * 100:.2f}% | Median={diag.median_coverage_ratio * 100:.2f}%")
    lines.append("-" * 76)
    lines.append("2. CHUNK ALLOCATION PER DOCUMENT DISTRIBUTION:")
    lines.append(f"  - Documents with 1 chunk  : {diag.docs_with_1_chunk:,} ({diag.pct_docs_with_1_chunk:.1f}%)")
    lines.append(f"  - Documents with 2 chunks : {diag.docs_with_2_chunks:,} ({diag.pct_docs_with_2_chunks:.1f}%)")
    lines.append(f"  - Documents with 3 chunks : {diag.docs_with_3_chunks:,} ({diag.pct_docs_with_3_chunks:.1f}%)")
    if diag.max_chunks_per_doc_limit == 2:
        cap_status = "0 (COMPLIANT: No doc exceeds cap 2)" if diag.docs_with_gt_2_chunks == 0 else f"{diag.docs_with_gt_2_chunks} (VIOLATION: Docs with > 2 chunks!)"
        lines.append(f"  - Documents with >2 chunks: {cap_status}")
    lines.append("-" * 76)
    lines.append("3. CAPACITY & SHORTFALL AUDIT:")
    if diag.queries_under_target_count == 0:
        lines.append(f"  - Chunk pool capacity  : 100% FULL ({diag.total_queries} queries have exactly {diag.target_chunks} valid chunks)")
    else:
        lines.append(f"  - Queries < {diag.target_chunks} chunks : {diag.queries_under_target_count} queries (total shortfall: {diag.total_chunk_shortfall} chunks)")
        lines.append(f"  - Query IDs with shortfall: {diag.queries_under_target[:20]}")

    if diag.chunks_skipped_cap_count is not None:
        lines.append(f"  - Chunks skipped by cap: {diag.chunks_skipped_cap_count:,} candidate chunks skipped because doc reached cap")

    lines.append("-" * 76)
    status_str = "PASSED [100% COMPLIANT]" if diag.cap_violations_count == 0 else f"FAILED [{diag.cap_violations_count} CAP VIOLATION(S)]"
    lines.append(f"Cap Compliance Status    : {status_str}")
    lines.append("=" * 76)
    return "\n".join(lines)


def format_comparison_report(report: ComparisonReport) -> str:
    """Format side-by-side comparison into a clear report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("=== VIBIOMIR EXPERIMENT COMPARISON: BASELINE A1 vs EXPERIMENT A1.2 ===")
    lines.append("=" * 78)
    lines.append(f"Baseline Submission     : {report.baseline_path}")
    lines.append(f"Experiment A1.2 (Cap 2) : {report.candidate_path}")
    lines.append("-" * 78)

    # 1. Ranking integrity check
    rank_status = "IDENTICAL (100% PRESERVED)" if report.ranking_identical else "MISMATCH DETECTED"
    lines.append(f"1. DOCUMENT RANKING INTEGRITY : {rank_status}")
    if report.ranking_identical:
        lines.append("   [OK] relevant_docs in A1.2 matches baseline 100 docs/query in exact order.")
        lines.append("   [OK] Document F2 score is mathematically guaranteed to be preserved.")
    else:
        lines.append(f"   [ERROR] Found {len(report.mismatched_queries)} query ranking mismatches!")
        for m in report.mismatched_queries[:5]:
            lines.append(f"     - Query {m.get('query_id')}: {m}")

    lines.append("-" * 78)
    # 2. Document coverage shift
    lines.append("2. DOCUMENT COVERAGE IN 100 CHUNKS:")
    lines.append(f"{'Metric':<28} | {'Baseline (Cap 3)':<18} | {'Exp A1.2 (Cap 2)':<18} | {'Gain / Delta'}")
    lines.append("-" * 78)
    gain_sign = "+" if report.mean_unique_docs_delta >= 0 else ""
    delta_str = f"{gain_sign}{report.mean_unique_docs_delta:.2f} ({gain_sign}{report.mean_unique_docs_pct_gain:.1f}%)"
    lines.append(f"{'Mean Unique Docs / Query':<28} | {report.baseline_mean_unique_docs:<18.2f} | {report.candidate_mean_unique_docs:<18.2f} | {delta_str}")
    med_delta_str = f"{gain_sign}{report.median_unique_docs_delta:.1f}"
    lines.append(f"{'Median Unique Docs / Query':<28} | {report.baseline_median_unique_docs:<18.1f} | {report.candidate_median_unique_docs:<18.1f} | {med_delta_str}")

    lines.append("-" * 78)
    # 3. Distribution shift
    lines.append("3. CHUNKS PER DOCUMENT DISTRIBUTION SHIFT:")
    lines.append(f"{'Allocation':<28} | {'Baseline (Cap 3)':<18} | {'Exp A1.2 (Cap 2)':<18} | {'Shift'}")
    lines.append("-" * 78)
    d1_shift = report.candidate_docs_1 - report.baseline_docs_1
    lines.append(f"{'Docs with 1 chunk':<28} | {report.baseline_docs_1:<18,} | {report.candidate_docs_1:<18,} | {d1_shift:+,}")
    d2_shift = report.candidate_docs_2 - report.baseline_docs_2
    lines.append(f"{'Docs with 2 chunks':<28} | {report.baseline_docs_2:<18,} | {report.candidate_docs_2:<18,} | {d2_shift:+,}")
    d3_shift = report.candidate_docs_3 - report.baseline_docs_3
    lines.append(f"{'Docs with 3 chunks':<28} | {report.baseline_docs_3:<18,} | {report.candidate_docs_3:<18,} | {d3_shift:+,}")

    lines.append("-" * 78)
    # 4. Queries breakdown
    lines.append("4. PER-QUERY COVERAGE DYNAMICS:")
    lines.append(f"  - Queries with INCREASED document coverage : {report.queries_coverage_increased} queries ({report.queries_coverage_increased / max(1, report.total_queries_compared) * 100:.1f}%)")
    lines.append(f"  - Queries with SAME document coverage      : {report.queries_coverage_same} queries ({report.queries_coverage_same / max(1, report.total_queries_compared) * 100:.1f}%)")
    lines.append(f"  - Queries with DECREASED document coverage  : {report.queries_coverage_decreased} queries ({report.queries_coverage_decreased / max(1, report.total_queries_compared) * 100:.1f}%)")
    lines.append("=" * 78)
    return "\n".join(lines)


def export_report_json(report_data: dict[str, Any], output_path: Path | str) -> None:
    """Save diagnostics or comparison report data to JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved diagnostics report to %s", out_p.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="ViBioMIR Submission Diagnostics & Comparison")
    parser.add_argument("--candidate", "-c", required=True, help="Path to submission.json or submission.zip")
    parser.add_argument("--baseline", "-b", help="Optional baseline submission.json or submission.zip to compare against")
    parser.add_argument("--target-docs", type=int, default=100, help="Expected docs per query (default: 100)")
    parser.add_argument("--target-chunks", type=int, default=100, help="Expected chunks per query (default: 100)")
    parser.add_argument("--max-chunks-per-doc", type=int, default=2, help="Expected max chunks per document (default: 2)")
    parser.add_argument("--output", "-o", help="Optional output JSON report path")

    args = parser.parse_args()

    if args.baseline:
        rep = compare_submissions(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            target_docs=args.target_docs,
            target_chunks=args.target_chunks,
        )
        print("\n" + format_comparison_report(rep) + "\n")
        if args.output:
            export_report_json(rep.to_dict(), args.output)
    else:
        diag = analyze_submission(
            submission_path=args.candidate,
            target_docs=args.target_docs,
            target_chunks=args.target_chunks,
            max_chunks_per_doc=args.max_chunks_per_doc,
        )
        print("\n" + format_diagnostics_report(diag) + "\n")
        if args.output:
            export_report_json(diag.to_dict(), args.output)


if __name__ == "__main__":
    main()
