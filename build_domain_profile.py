from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from crawler.models import TERMINAL_STATUSES
from crawler.source import iter_source_rows
from crawler.utils.files import replace_with_retry
from crawler.utils.hashing import sha256_file, stable_int
from crawler.utils.urls import domain_from_url, normalize_fetch_url

PRIORITY_DOMAINS = {
    "www.cnkang.com",
    "www.120ask.com",
    "ask.39.net",
    "www.familydoctor.com.cn",
    "zysjonline.com",
}
PROFILE_STATUSES = (
    "SUCCESS",
    "ROBOTS_DENIED",
    "FAILED",
    "EMPTY_CONTENT",
    "NEEDS_JS",
    "TOO_LARGE",
    "UNSUPPORTED",
    "NEEDS_OCR",
)
COST_CATEGORIES = ("SUCCESS", "FAILED", "EMPTY_CONTENT", "NEEDS_JS", "OTHER")
TIMING_CATEGORIES = (*COST_CATEGORIES, "ROBOTS_DENIED")
COST_MODEL_VERSION = "expected-service-time-v1"
ROBOTS_COST_SECONDS = 0.02
PRIOR_SAMPLE_SIZE = 20.0


def _readonly_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve().as_posix()
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    replace_with_retry(temporary, path)


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile from an empty sample")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def _latency_cost(values: list[float]) -> float:
    return 0.7 * float(median(values)) + 0.3 * _nearest_rank(values, 0.95)


def _cost_category(status: str) -> str:
    return status if status in TIMING_CATEGORIES else "OTHER"


def create_calibration_sample(
    input_path: Path,
    output_path: Path,
    *,
    per_domain: int = 100,
    priority_per_domain: int = 500,
    seed: int = 20261005,
) -> dict[str, Any]:
    if per_domain < 1 or priority_per_domain < per_domain:
        raise ValueError("sample sizes must satisfy priority_per_domain >= per_domain >= 1")
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    heaps: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    domain_counts: Counter[str] = Counter()
    input_rows = 0
    for doc_id, url in iter_source_rows(input_path):
        domain = domain_from_url(normalize_fetch_url(url))
        domain_counts[domain] += 1
        input_rows += 1
        limit = priority_per_domain if domain in PRIORITY_DOMAINS else per_domain
        score = stable_int(str(doc_id), seed)
        candidate = (-score, -doc_id, doc_id, url)
        heap = heaps[domain]
        if len(heap) < limit:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)

    rows: list[dict[str, Any]] = []
    for domain in sorted(domain_counts):
        selected = sorted(heaps[domain], key=lambda item: (-item[0], item[2]))
        for _negative_score, _negative_doc_id, doc_id, url in selected:
            rows.append({"doc_id": doc_id, "url": url, "domain": domain})
    rows.sort(key=lambda row: int(row["doc_id"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    replace_with_retry(temporary, output_path)
    return {
        "input": str(input_path),
        "input_rows": input_rows,
        "domains": len(domain_counts),
        "output": str(output_path),
        "sample_rows": len(rows),
        "per_domain": per_domain,
        "priority_per_domain": priority_per_domain,
        "priority_domains": sorted(PRIORITY_DOMAINS),
        "seed": seed,
    }


def _source_domain_counts(input_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _doc_id, url in iter_source_rows(input_path):
        counts[domain_from_url(normalize_fetch_url(url))] += 1
    return counts


def _status_counts(manifest_path: Path) -> tuple[dict[str, Counter[str]], int]:
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    with _readonly_connection(manifest_path) as connection:
        rows = connection.execute(
            "SELECT domain, status, COUNT(*) AS n FROM crawl_tasks GROUP BY domain, status"
        )
        total = 0
        for row in rows:
            count = int(row["n"])
            by_domain[str(row["domain"])][str(row["status"])] = count
            total += count
    return by_domain, total


def _timing_rows(path: Path) -> dict[str, dict[str, Any]]:
    attempts: dict[tuple[str, int], tuple[str, float, str, bool]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                fetch_key = str(row["fetch_key"])
                attempt = int(row["attempt"])
                domain = str(row["domain"])
                elapsed = float(row["elapsed_seconds"])
                status = str(row["status"])
                retryable = bool(row["retryable"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid timing row {path}:{line_number}: {exc}") from exc
            if not fetch_key or attempt < 1 or elapsed < 0 or not math.isfinite(elapsed):
                raise ValueError(f"invalid timing values at {path}:{line_number}")
            key = (fetch_key, attempt)
            value = (domain, elapsed, status, retryable)
            previous = attempts.get(key)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting duplicate timing attempt {key}")
            attempts[key] = value

    targets: dict[str, dict[str, Any]] = {}
    for (fetch_key, _attempt), (domain, elapsed, _status, _retryable) in attempts.items():
        target = targets.setdefault(fetch_key, {"domain": domain, "elapsed_seconds": 0.0})
        if target["domain"] != domain:
            raise ValueError(f"fetch key {fetch_key} appears under multiple domains")
        target["elapsed_seconds"] += elapsed
    return targets


def _calibration_latencies(
    manifest_path: Path, timing_path: Path
) -> tuple[dict[str, list[float]], dict[tuple[str, str], list[float]], int]:
    timings = _timing_rows(timing_path)
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_domain_status: dict[tuple[str, str], list[float]] = defaultdict(list)
    terminal_values = {status.value for status in TERMINAL_STATUSES}
    with _readonly_connection(manifest_path) as connection:
        unfinished = int(
            connection.execute(
                "SELECT COUNT(*) FROM fetch_targets "
                "WHERE status IN ('PENDING','FETCHING','RETRY_WAIT')"
            ).fetchone()[0]
        )
        if unfinished:
            raise RuntimeError(f"calibration manifest still has {unfinished} unfinished targets")
        manifest_rows = {
            str(row["fetch_key"]): (str(row["domain"]), str(row["status"]))
            for row in connection.execute("SELECT fetch_key, domain, status FROM fetch_targets")
        }
    missing = sorted(set(manifest_rows) - set(timings))
    extra = sorted(set(timings) - set(manifest_rows))
    if missing or extra:
        raise RuntimeError(
            "calibration timing/manifest mismatch: "
            f"missing_timings={len(missing)} extra_timings={len(extra)}"
        )
    for fetch_key, (domain, status) in manifest_rows.items():
        if status not in terminal_values:
            raise RuntimeError(f"non-terminal calibration status for {fetch_key}: {status}")
        timing = timings[fetch_key]
        if timing["domain"] != domain:
            raise RuntimeError(f"calibration domain mismatch for {fetch_key}")
        elapsed = float(timing["elapsed_seconds"])
        by_domain[domain].append(elapsed)
        by_domain_status[(domain, _cost_category(status))].append(elapsed)
    return by_domain, by_domain_status, len(manifest_rows)


def _write_status_summary(
    output_path: Path,
    source_counts: Counter[str],
    status_counts: dict[str, Counter[str]],
) -> None:
    fields = ["domain", *[status.lower() for status in PROFILE_STATUSES], "sample_count"]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for domain in sorted(source_counts):
            counts = status_counts.get(domain, Counter())
            sample_count = sum(counts.get(status, 0) for status in PROFILE_STATUSES)
            row: dict[str, Any] = {"domain": domain, "sample_count": sample_count}
            row.update({status.lower(): counts.get(status, 0) for status in PROFILE_STATUSES})
            writer.writerow(row)
    replace_with_retry(temporary, output_path)


def build_status_summary(
    input_path: Path, status_manifest: Path, output_path: Path
) -> dict[str, Any]:
    input_path = input_path.resolve()
    status_manifest = status_manifest.resolve()
    output_path = output_path.resolve()
    source_counts = _source_domain_counts(input_path)
    statuses, manifest_total = _status_counts(status_manifest)
    source_total = sum(source_counts.values())
    if manifest_total != source_total or set(statuses) != set(source_counts):
        raise RuntimeError("status manifest does not match the source corpus")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_status_summary(output_path, source_counts, statuses)
    return {
        "output": str(output_path),
        "domains": len(source_counts),
        "source_rows": source_total,
        "status_totals": {
            status: sum(counts.get(status, 0) for counts in statuses.values())
            for status in PROFILE_STATUSES
        },
    }


def build_domain_profile(
    input_path: Path,
    status_manifest: Path,
    timing_manifest: Path,
    timing_log: Path,
    output_dir: Path,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    status_manifest = status_manifest.resolve()
    timing_manifest = timing_manifest.resolve()
    timing_log = timing_log.resolve()
    output_dir = output_dir.resolve()
    for path in (input_path, status_manifest, timing_manifest, timing_log):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_counts = _source_domain_counts(input_path)
    statuses, manifest_total = _status_counts(status_manifest)
    if manifest_total != sum(source_counts.values()):
        raise RuntimeError(
            f"status manifest has {manifest_total} rows; source has {sum(source_counts.values())}"
        )
    if set(statuses) != set(source_counts):
        raise RuntimeError("status manifest domains do not match the source corpus")
    domain_timings, status_timings, timing_targets = _calibration_latencies(
        timing_manifest, timing_log
    )

    all_nonrobots = [
        value
        for (domain, category), values in status_timings.items()
        if category != "ROBOTS_DENIED"
        for value in values
    ]
    if not all_nonrobots:
        raise RuntimeError("calibration contains no non-robots timing samples")
    global_fallback = _latency_cost(all_nonrobots)
    global_costs: dict[str, float] = {}
    for category in COST_CATEGORIES:
        values = [
            value
            for (domain, observed), samples in status_timings.items()
            if observed == category
            for value in samples
        ]
        global_costs[category] = _latency_cost(values) if values else global_fallback

    profile_rows: list[dict[str, Any]] = []
    for domain in sorted(source_counts):
        counts = statuses[domain]
        sample_count = sum(counts.get(status, 0) for status in PROFILE_STATUSES)
        if sample_count == 0:
            raise RuntimeError(f"domain has no terminal status sample: {domain}")
        rates = {status: counts.get(status, 0) / sample_count for status in PROFILE_STATUSES}
        other_rate = sum(rates[status] for status in ("TOO_LARGE", "UNSUPPORTED", "NEEDS_OCR"))
        category_rates = {
            "SUCCESS": rates["SUCCESS"],
            "FAILED": rates["FAILED"],
            "EMPTY_CONTENT": rates["EMPTY_CONTENT"],
            "NEEDS_JS": rates["NEEDS_JS"],
            "OTHER": other_rate,
        }
        estimated_seconds = rates["ROBOTS_DENIED"] * ROBOTS_COST_SECONDS
        for category, rate in category_rates.items():
            samples = status_timings.get((domain, category), [])
            observed = _latency_cost(samples) if samples else global_costs[category]
            weight = len(samples) / (len(samples) + PRIOR_SAMPLE_SIZE)
            adjusted = weight * observed + (1.0 - weight) * global_costs[category]
            estimated_seconds += rate * adjusted

        timing_values = domain_timings.get(domain, [])
        if not timing_values:
            raise RuntimeError(f"domain has no calibration timing sample: {domain}")
        profile_rows.append(
            {
                "domain": domain,
                "full_url_count": source_counts[domain],
                "sample_count": sample_count,
                "success_rate": rates["SUCCESS"],
                "robots_rate": rates["ROBOTS_DENIED"],
                "failed_rate": rates["FAILED"],
                "empty_rate": rates["EMPTY_CONTENT"],
                "needs_js_rate": rates["NEEDS_JS"],
                "other_rate": other_rate,
                "timing_sample_count": len(timing_values),
                "median_fetch_seconds": float(median(timing_values)),
                "p95_fetch_seconds": _nearest_rank(timing_values, 0.95),
                "estimated_seconds_per_url": estimated_seconds,
                "estimated_work": source_counts[domain] * estimated_seconds,
            }
        )

    status_output = output_dir / "domain_status_summary.csv"
    profile_output = output_dir / "domain_profile.csv"
    metadata_output = output_dir / "domain_profile_metadata.json"
    _write_status_summary(status_output, source_counts, statuses)
    fields = list(profile_rows[0])
    temporary = profile_output.with_suffix(profile_output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in profile_rows:
            writer.writerow(
                {
                    key: f"{value:.9f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    replace_with_retry(temporary, profile_output)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "cost_model_version": COST_MODEL_VERSION,
        "robots_cost_seconds": ROBOTS_COST_SECONDS,
        "prior_sample_size": PRIOR_SAMPLE_SIZE,
        "latency_blend": {"median": 0.7, "p95": 0.3},
        "timing_definition": (
            "active robots/network/read time summed across attempts; "
            "scheduler semaphore and crawl-delay queue waits excluded"
        ),
        "quantile_method": "nearest-rank",
        "global_status_cost_seconds": global_costs,
        "inputs": {
            "source": {"path": str(input_path), "sha256": sha256_file(input_path)},
            "status_manifest": {
                "path": str(status_manifest),
                "sha256": sha256_file(status_manifest),
            },
            "timing_manifest": {
                "path": str(timing_manifest),
                "sha256": sha256_file(timing_manifest),
            },
            "timing_log": {"path": str(timing_log), "sha256": sha256_file(timing_log)},
        },
        "domains": len(profile_rows),
        "source_rows": sum(source_counts.values()),
        "timing_targets": timing_targets,
        "priority_domains": [row for row in profile_rows if str(row["domain"]) in PRIORITY_DOMAINS],
    }
    _atomic_json(metadata_output, metadata)
    return {
        "domain_status_summary": str(status_output),
        "domain_profile": str(profile_output),
        "metadata": str(metadata_output),
        "domains": len(profile_rows),
        "source_rows": sum(source_counts.values()),
        "timing_targets": timing_targets,
        "status_totals": {
            status: sum(counts.get(status, 0) for counts in statuses.values())
            for status in PROFILE_STATUSES
        },
        "priority_domains": [row for row in profile_rows if str(row["domain"]) in PRIORITY_DOMAINS],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ViBioMIR domain workload profiles")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="create a deterministic calibration sample")
    sample.add_argument("--input", required=True, type=Path)
    sample.add_argument("--output", required=True, type=Path)
    sample.add_argument("--per-domain", default=100, type=int)
    sample.add_argument("--priority-per-domain", default=500, type=int)
    sample.add_argument("--seed", default=20261005, type=int)
    status = commands.add_parser("status", help="write an expanded domain status report")
    status.add_argument("--input", required=True, type=Path)
    status.add_argument("--status-manifest", required=True, type=Path)
    status.add_argument("--output", required=True, type=Path)
    profile = commands.add_parser("build", help="build status and workload profiles")
    profile.add_argument("--input", required=True, type=Path)
    profile.add_argument("--status-manifest", required=True, type=Path)
    profile.add_argument("--timing-manifest", required=True, type=Path)
    profile.add_argument("--timing-log", required=True, type=Path)
    profile.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "sample":
        result = create_calibration_sample(
            args.input,
            args.output,
            per_domain=args.per_domain,
            priority_per_domain=args.priority_per_domain,
            seed=args.seed,
        )
    elif args.command == "status":
        result = build_status_summary(args.input, args.status_manifest, args.output)
    else:
        result = build_domain_profile(
            args.input,
            args.status_manifest,
            args.timing_manifest,
            args.timing_log,
            args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
