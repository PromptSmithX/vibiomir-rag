from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from crawler.utils.urls import domain_from_url, normalize_fetch_url


def validate_worker_partition(
    input_file: Path, assignment_file: Path, worker_id: int
) -> tuple[int, dict[str, int]]:
    if not assignment_file.is_file():
        raise FileNotFoundError(assignment_file)
    table = pq.read_table(assignment_file)
    names = set(table.column_names)
    required = {"domain", "worker_id", "url_count"}
    if not required.issubset(names):
        raise ValueError(f"assignment is missing columns: {sorted(required - names)}")

    owners: dict[str, set[int]] = defaultdict(set)
    expected_counts: dict[tuple[str, int], int] = {}
    overrides: dict[str, int] = {}
    split_parts_by_domain: dict[str, int] = {}
    concurrency_totals: Counter[str] = Counter()
    is_v2 = "split_parts" in names
    for row in table.to_pylist():
        domain = str(row["domain"])
        owner = int(row["worker_id"])
        count = int(row["url_count"])
        split_parts = int(row.get("split_parts") or 1) if is_v2 else 1
        limit_value = row.get("per_worker_domain_concurrency") if is_v2 else None
        limit = int(limit_value) if limit_value is not None else None
        key = (domain, owner)
        if key in expected_counts:
            raise ValueError(f"duplicate domain/worker assignment: {domain}/{owner}")
        if split_parts not in {1, 2} or count < 1:
            raise ValueError(f"invalid assignment for domain {domain}")
        previous_parts = split_parts_by_domain.setdefault(domain, split_parts)
        if previous_parts != split_parts:
            raise ValueError(f"inconsistent split_parts for domain {domain}")
        owners[domain].add(owner)
        expected_counts[key] = count
        if split_parts > 1:
            if limit != 1:
                raise ValueError(f"split domain {domain} must use per-worker concurrency 1")
            concurrency_totals[domain] += limit
            if owner == worker_id:
                overrides[domain] = limit
        elif limit is not None:
            raise ValueError(f"unsplit domain {domain} must not define a concurrency override")

    for domain, domain_owners in owners.items():
        expected_parts = split_parts_by_domain[domain]
        if len(domain_owners) != expected_parts:
            raise ValueError(f"domain {domain} has an invalid number of worker assignments")
        if expected_parts > 1 and concurrency_totals[domain] > 2:
            raise ValueError(f"domain {domain} exceeds aggregate concurrency 2")

    observed: Counter[str] = Counter()
    rows = 0
    parquet = pq.ParquetFile(input_file)
    for batch in parquet.iter_batches(columns=["doc_id", "url", "domain"], batch_size=65_536):
        for row in batch.to_pylist():
            rows += 1
            domain = str(row["domain"])
            parsed_domain = domain_from_url(normalize_fetch_url(str(row["url"])))
            if domain != parsed_domain or worker_id not in owners.get(domain, set()):
                raise ValueError(
                    f"worker {worker_id} partition row {row['doc_id']} "
                    f"has unassigned or mismatched domain {domain!r}"
                )
            observed[domain] += 1
    if rows == 0:
        raise ValueError(f"worker {worker_id} partition is empty")
    expected = {
        domain: count for (domain, owner), count in expected_counts.items() if owner == worker_id
    }
    if dict(observed) != expected:
        raise ValueError(f"worker {worker_id} domain counts do not match assignment")
    return rows, overrides
