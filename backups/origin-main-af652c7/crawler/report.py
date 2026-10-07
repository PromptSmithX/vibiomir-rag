from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from crawler.config import AppConfig
from crawler.manifest import Manifest

REVIEW_CRITERIA = (
    "review_title_correct",
    "review_body_correct",
    "review_structure_ok",
    "review_noise_ok",
    "review_medical_values_ok",
)


def summarize_review(review_file: Path) -> dict[str, object]:
    reviewed = 0
    reviewed_success = 0
    acceptable_success = 0
    invalid_decisions = 0
    criterion_failures = {criterion: 0 for criterion in REVIEW_CRITERIA}
    domains: dict[str, dict[str, int]] = {}

    with review_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            decisions = [row.get(criterion) for criterion in REVIEW_CRITERIA]
            if any(value is None for value in decisions):
                continue
            if any(type(value) is not bool for value in decisions):
                invalid_decisions += 1
                continue

            reviewed += 1
            if row.get("crawl_status") != "SUCCESS":
                continue

            reviewed_success += 1
            domain = str(row.get("domain") or "unknown")
            domain_summary = domains.setdefault(
                domain,
                {"reviewed_success": 0, "unacceptable_success": 0, "noise_failures": 0},
            )
            domain_summary["reviewed_success"] += 1
            if all(decisions):
                acceptable_success += 1
            else:
                domain_summary["unacceptable_success"] += 1
            for criterion, decision in zip(REVIEW_CRITERIA, decisions, strict=True):
                if not decision:
                    criterion_failures[criterion] += 1
            if row.get("review_noise_ok") is False:
                domain_summary["noise_failures"] += 1

    by_domain = [
        {"domain": domain, **values}
        for domain, values in sorted(
            domains.items(),
            key=lambda item: (
                -item[1]["noise_failures"],
                -item[1]["unacceptable_success"],
                item[0],
            ),
        )
    ]
    return {
        "file": str(review_file),
        "reviewed": reviewed,
        "reviewed_success": reviewed_success,
        "acceptable_success": acceptable_success,
        "acceptable_success_rate": (
            acceptable_success / reviewed_success if reviewed_success else None
        ),
        "invalid_decisions": invalid_decisions,
        "criterion_failures": criterion_failures,
        "by_domain": by_domain,
    }


def build_report(
    config: AppConfig,
    manifest: Manifest,
    output: Path,
    review_file: Path | None = None,
) -> dict[str, object]:
    counts = manifest.counts()
    http = {
        str(row[0]) if row[0] is not None else "null": int(row[1])
        for row in manifest.connection.execute(
            "SELECT http_status, COUNT(*) FROM crawl_tasks GROUP BY http_status"
        )
    }
    content_types = {
        str(row[0]) if row[0] is not None else "null": int(row[1])
        for row in manifest.connection.execute(
            "SELECT content_type, COUNT(*) FROM crawl_tasks GROUP BY content_type"
        )
    }
    top_errors = [
        {"domain": row[0], "status": row[1], "count": int(row[2])}
        for row in manifest.connection.execute(
            """
            SELECT domain, status, COUNT(*) AS n FROM crawl_tasks
            WHERE status != 'SUCCESS'
            GROUP BY domain, status ORDER BY n DESC LIMIT 50
            """
        )
    ]
    shards = sorted(config.paths.corpus_dir.glob("part-*.parquet"))
    corpus_bytes = sum(path.stat().st_size for path in shards)
    disk = shutil.disk_usage(config.paths.corpus_dir)
    terminal = sum(
        count
        for status, count in counts.items()
        if status not in {"TOTAL", "PENDING", "FETCHING", "RETRY_WAIT"}
    )
    avg_disk = corpus_bytes / terminal if terminal else None
    analysis_path = config.paths.source_dir / "analysis.json"
    source_rows = None
    if analysis_path.exists():
        try:
            source_rows = int(json.loads(analysis_path.read_text(encoding="utf-8"))["total_rows"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            source_rows = None
    review_summary = None
    if review_file and review_file.exists():
        review_summary = summarize_review(review_file)

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": str(config.paths.manifest),
        "counts": counts,
        "http_statuses": http,
        "content_types": content_types,
        "shards": len(shards),
        "corpus_bytes": corpus_bytes,
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "average_disk_bytes_per_terminal_document": avg_disk,
        "estimated_100k_bytes": avg_disk * 100_000 if avg_disk else None,
        "source_rows": source_rows,
        "estimated_full_corpus_bytes": avg_disk * source_rows if avg_disk and source_rows else None,
        "estimated_full_with_10_percent_headroom_bytes": (
            avg_disk * source_rows * 1.10 if avg_disk and source_rows else None
        ),
        "top_errors": top_errors,
        "manual_review": review_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    lines = [
        "# ViBioMIR crawl report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"- Terminal documents: {terminal:,}",
        f"- Shards: {len(shards):,}",
        f"- Corpus bytes: {corpus_bytes:,}",
    ]
    lines.append(
        f"- Average bytes/document: {avg_disk:,.1f}"
        if avg_disk
        else "- No terminal documents yet."
    )
    if review_summary:
        rate = review_summary["acceptable_success_rate"]
        lines.append(f"- Reviewed documents: {review_summary['reviewed']}")
        lines.append(
            f"- Acceptable SUCCESS review rate: {rate:.1%}"
            if rate is not None
            else "- SUCCESS review incomplete."
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
