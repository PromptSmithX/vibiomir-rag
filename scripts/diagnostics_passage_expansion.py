"""Convenience CLI script to analyze and compare source passage expansion experiments (A1.3 & A1.4)."""

import argparse
from pathlib import Path

from retrieval.dense.tokenizer_helper import FallbackTokenCounter
from retrieval.submission.passage_diagnostics import (
    analyze_expanded_submission,
    export_diagnostics_json,
    format_passage_diagnostics_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ViBioMIR Experiments A1.3 / A1.4 Passage Expansion Diagnostics & Baseline Comparison"
    )
    parser.add_argument(
        "--candidate",
        "-c",
        required=True,
        help="Path to candidate submission.json or submission.zip (A1.3 or A1.4)",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        default="data/experiments/a1_2_max2_chunks/submission/submission.json",
        help="Path to baseline A1.2 submission.json or submission.zip",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=1024,
        help="Max token limit for passage expansion (768 for A1.3, 1024 for A1.4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional destination path to save diagnostics.json",
    )
    parser.add_argument(
        "--target-chunks",
        type=int,
        default=100,
        help="Expected chunks per query (default: 100)",
    )

    args = parser.parse_args()

    cand_path = Path(args.candidate)
    base_path = Path(args.baseline) if args.baseline else None

    # Check candidate exists
    if not cand_path.exists():
        zip_cand = cand_path.with_suffix(".zip")
        if zip_cand.exists():
            cand_path = zip_cand
        else:
            print(f"[ERROR] Candidate submission file not found: {cand_path}")
            return

    # Check baseline exists
    if base_path and not base_path.exists():
        base_zip = base_path.with_suffix(".zip")
        if base_zip.exists():
            base_path = base_zip
        else:
            print(f"[WARNING] Baseline submission file not found at {base_path}. Proceeding without baseline diff.")
            base_path = None

    tc = FallbackTokenCounter()
    print(f"\nAnalyzing Passage Expansion in: {cand_path} (Budget: {args.max_tokens} tokens)...")
    if base_path:
        print(f"Comparing against Baseline: {base_path}...")

    diag = analyze_expanded_submission(
        submission_path=cand_path,
        baseline_path=base_path,
        max_tokens=args.max_tokens,
        token_counter=tc,
        target_chunks=args.target_chunks,
    )

    report_str = format_passage_diagnostics_report(diag)
    print("\n" + report_str + "\n")

    out_p = args.output
    if not out_p:
        out_p = cand_path.parent / "diagnostics.json"

    export_diagnostics_json(diag.to_dict(), out_p)
    print(f"Diagnostics JSON report saved to: {Path(out_p).resolve()}\n")


if __name__ == "__main__":
    main()
