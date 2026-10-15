"""Convenience CLI script to run diagnostics and compare Experiment A1.2 with Baseline A1."""

import argparse
from pathlib import Path

from retrieval.submission.diagnostics import (
    analyze_submission,
    compare_submissions,
    export_report_json,
    format_comparison_report,
    format_diagnostics_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment A1.2 Diagnostics & Baseline A1 Comparison"
    )
    parser.add_argument(
        "--candidate",
        "-c",
        default="data/experiments/a1_2_max2_chunks/submission/submission.json",
        help="Path to Experiment A1.2 submission.json or submission.zip",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        default="data/experiments/baseline_a1/submission/submission.json",
        help="Path to Baseline A1 submission.json or submission.zip (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/experiments/a1_2_max2_chunks/submission/diagnostics_report.json",
        help="Path to save output JSON report",
    )
    parser.add_argument("--target-docs", type=int, default=100, help="Expected docs/query (100)")
    parser.add_argument("--target-chunks", type=int, default=100, help="Expected chunks/query (100)")
    parser.add_argument("--max-chunks-per-doc", type=int, default=2, help="Max chunks/doc (2)")

    args = parser.parse_args()

    cand_path = Path(args.candidate)
    base_path = Path(args.baseline) if args.baseline else None

    # Check if candidate file exists
    if not cand_path.exists():
        # Check if ZIP exists instead
        zip_cand = cand_path.with_suffix(".zip")
        if zip_cand.exists():
            cand_path = zip_cand
        else:
            print(f"[ERROR] Candidate submission file not found: {cand_path}")
            print("Please run the submission export command first.")
            return

    # Check if baseline file exists for comparison
    has_baseline = False
    if base_path:
        if base_path.exists():
            has_baseline = True
        else:
            base_zip = base_path.with_suffix(".zip")
            if base_zip.exists():
                base_path = base_zip
                has_baseline = True
            else:
                # Check legacy exp_a1 path
                legacy_base = Path("data/experiments/exp_a1/submission/submission.json")
                if legacy_base.exists():
                    base_path = legacy_base
                    has_baseline = True

    if has_baseline and base_path:
        print(f"\nComparing Candidate ({cand_path}) against Baseline ({base_path})...")
        report = compare_submissions(
            baseline_path=base_path,
            candidate_path=cand_path,
            target_docs=args.target_docs,
            target_chunks=args.target_chunks,
        )
        print("\n" + format_comparison_report(report) + "\n")
        if args.output:
            export_report_json(report.to_dict(), args.output)
            print(f"Comparison report saved to: {Path(args.output).resolve()}\n")
    else:
        print(f"\nAnalyzing Candidate Submission ({cand_path})...")
        diag = analyze_submission(
            submission_path=cand_path,
            target_docs=args.target_docs,
            target_chunks=args.target_chunks,
            max_chunks_per_doc=args.max_chunks_per_doc,
        )
        print("\n" + format_diagnostics_report(diag) + "\n")
        if args.output:
            export_report_json(diag.to_dict(), args.output)
            print(f"Diagnostics report saved to: {Path(args.output).resolve()}\n")


if __name__ == "__main__":
    main()
