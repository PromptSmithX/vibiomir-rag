from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crawler.source import iter_source_rows
from crawler.utils.files import replace_with_retry
from crawler.utils.urls import domain_from_url, normalize_fetch_url

WORKER_COUNT = 6
SPLIT_IMPROVEMENT_THRESHOLD = 0.10
MAX_SPLIT_PARTS = 2
PARTITION_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.int64(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class AssignmentPart:
    domain: str
    split_part: int
    split_parts: int
    url_count: int
    estimated_work: float
    worker_id: int = -1


def _worker_filename(worker_id: int) -> str:
    suffix = "local" if worker_id == 0 else "kaggle"
    return f"worker_{worker_id}_{suffix}.parquet"


def _insert_id_batch(connection: sqlite3.Connection, values: list[tuple[int]]) -> None:
    connection.execute("SAVEPOINT id_batch")
    try:
        connection.executemany("INSERT INTO input_ids(doc_id) VALUES (?)", values)
    except sqlite3.IntegrityError as batch_error:
        connection.execute("ROLLBACK TO id_batch")
        connection.execute("RELEASE id_batch")
        for (doc_id,) in values:
            try:
                connection.execute("INSERT INTO input_ids(doc_id) VALUES (?)", (doc_id,))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate doc_id in input: {doc_id}") from exc
        raise RuntimeError("failed to identify duplicate doc_id") from batch_error
    else:
        connection.execute("RELEASE id_batch")


def _scan_input(input_path: Path, connection: sqlite3.Connection) -> tuple[int, Counter[str]]:
    domain_counts: Counter[str] = Counter()
    pending_ids: list[tuple[int]] = []
    total = 0
    for doc_id, url in iter_source_rows(input_path):
        try:
            domain = domain_from_url(normalize_fetch_url(url))
        except ValueError as exc:
            raise ValueError(f"invalid URL for doc_id {doc_id}: {url!r}: {exc}") from exc
        if not domain:
            raise ValueError(f"URL has no hostname for doc_id {doc_id}: {url!r}")
        domain_counts[domain] += 1
        pending_ids.append((doc_id,))
        total += 1
        if len(pending_ids) >= 50_000:
            _insert_id_batch(connection, pending_ids)
            connection.commit()
            pending_ids.clear()
    if pending_ids:
        _insert_id_batch(connection, pending_ids)
        connection.commit()
    if total == 0:
        raise ValueError("input parquet is empty")
    return total, domain_counts


def _load_workloads(profile_path: Path | None, domain_counts: Counter[str]) -> dict[str, float]:
    if profile_path is None:
        return {domain: float(count) for domain, count in domain_counts.items()}
    workloads: dict[str, float] = {}
    profile_counts: dict[str, int] = {}
    with profile_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"domain", "full_url_count", "estimated_work"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"domain profile must contain {sorted(required)}")
        for row in reader:
            domain = str(row["domain"])
            if domain in workloads:
                raise ValueError(f"duplicate domain in profile: {domain}")
            count = int(row["full_url_count"])
            work = float(row["estimated_work"])
            if count < 1 or not math.isfinite(work) or work <= 0:
                raise ValueError(f"invalid profile values for domain {domain}")
            profile_counts[domain] = count
            workloads[domain] = work
    if set(workloads) != set(domain_counts):
        missing = sorted(set(domain_counts) - set(workloads))
        extra = sorted(set(workloads) - set(domain_counts))
        raise ValueError(f"profile domains mismatch: missing={missing[:5]} extra={extra[:5]}")
    mismatched = [
        domain for domain, count in domain_counts.items() if profile_counts[domain] != count
    ]
    if mismatched:
        domain = mismatched[0]
        raise ValueError(
            f"profile count mismatch for {domain}: "
            f"profile={profile_counts[domain]} input={domain_counts[domain]}"
        )
    return workloads


def _parts_for_domains(
    domain_counts: Counter[str], workloads: dict[str, float], split_domains: set[str]
) -> list[AssignmentPart]:
    parts: list[AssignmentPart] = []
    for domain in sorted(domain_counts):
        count = domain_counts[domain]
        work = workloads[domain]
        split_parts = MAX_SPLIT_PARTS if domain in split_domains else 1
        for split_part in range(split_parts):
            part_count = count // split_parts + int(split_part < count % split_parts)
            part_work = work * part_count / count
            parts.append(AssignmentPart(domain, split_part, split_parts, part_count, part_work))
    return parts


def _greedy_parts(parts: list[AssignmentPart]) -> tuple[list[AssignmentPart], list[float]]:
    loads = [0.0] * WORKER_COUNT
    used_workers: dict[str, set[int]] = defaultdict(set)
    assigned: list[AssignmentPart] = []
    for part in sorted(
        parts,
        key=lambda value: (-value.estimated_work, value.domain, value.split_part),
    ):
        candidates = [
            worker_id
            for worker_id in range(WORKER_COUNT)
            if worker_id not in used_workers[part.domain]
        ]
        worker_id = min(candidates, key=lambda value: (loads[value], value))
        assigned.append(
            AssignmentPart(
                part.domain,
                part.split_part,
                part.split_parts,
                part.url_count,
                part.estimated_work,
                worker_id,
            )
        )
        used_workers[part.domain].add(worker_id)
        loads[worker_id] += part.estimated_work
    return assigned, loads


def _assign_by_work(
    domain_counts: Counter[str],
    workloads: dict[str, float],
    *,
    allow_splits: bool,
) -> tuple[list[AssignmentPart], list[float], dict[str, object]]:
    split_domains: set[str] = set()
    current, loads = _greedy_parts(_parts_for_domains(domain_counts, workloads, split_domains))
    baseline_makespan = max(loads)
    total_work = sum(workloads.values())
    ideal = total_work / WORKER_COUNT
    if allow_splits:
        while True:
            current_makespan = max(loads)
            bottleneck_workers = {
                worker_id
                for worker_id, load in enumerate(loads)
                if math.isclose(load, current_makespan, rel_tol=1e-12, abs_tol=1e-9)
            }
            current_owner = {
                part.domain: part.worker_id for part in current if part.split_parts == 1
            }
            best: tuple[float, str, list[AssignmentPart], list[float]] | None = None
            for domain, work in workloads.items():
                if (
                    domain in split_domains
                    or work <= ideal
                    or current_owner.get(domain) not in bottleneck_workers
                    or domain_counts[domain] < 2
                ):
                    continue
                candidate_splits = {*split_domains, domain}
                candidate, candidate_loads = _greedy_parts(
                    _parts_for_domains(domain_counts, workloads, candidate_splits)
                )
                improvement = (current_makespan - max(candidate_loads)) / current_makespan
                value = (improvement, domain, candidate, candidate_loads)
                if best is None or (value[0], value[1]) > (best[0], best[1]):
                    best = value
            if best is None or best[0] + 1e-12 < SPLIT_IMPROVEMENT_THRESHOLD:
                break
            _improvement, domain, current, loads = best
            split_domains.add(domain)
    return (
        current,
        loads,
        {
            "total_estimated_work": total_work,
            "ideal_worker_work": ideal,
            "baseline_makespan": baseline_makespan,
            "final_makespan": max(loads),
            "split_domains": sorted(split_domains),
        },
    )


def _write_worker_partitions(
    input_path: Path, output_dir: Path, assignments: list[AssignmentPart]
) -> list[Path]:
    assignment_workers = {(part.domain, part.split_part): part.worker_id for part in assignments}
    split_counts = {part.domain: part.split_parts for part in assignments}
    domain_seen: Counter[str] = Counter()
    temporary_paths = [
        output_dir / f"{_worker_filename(worker_id)}.tmp" for worker_id in range(WORKER_COUNT)
    ]
    writers = [
        pq.ParquetWriter(path, PARTITION_SCHEMA, compression="zstd", use_dictionary=True)
        for path in temporary_paths
    ]
    buffers: list[list[dict[str, object]]] = [[] for _ in range(WORKER_COUNT)]

    def flush(worker_id: int) -> None:
        if not buffers[worker_id]:
            return
        table = pa.Table.from_pylist(buffers[worker_id], schema=PARTITION_SCHEMA)
        writers[worker_id].write_table(table, row_group_size=16_384)
        buffers[worker_id].clear()

    try:
        for doc_id, url in iter_source_rows(input_path):
            domain = domain_from_url(normalize_fetch_url(url))
            split_part = domain_seen[domain] % split_counts[domain]
            domain_seen[domain] += 1
            worker_id = assignment_workers[(domain, split_part)]
            buffers[worker_id].append({"doc_id": doc_id, "url": url, "domain": domain})
            if len(buffers[worker_id]) >= 16_384:
                flush(worker_id)
        for worker_id in range(WORKER_COUNT):
            flush(worker_id)
    finally:
        for writer in writers:
            writer.close()

    final_paths = [output_dir / _worker_filename(i) for i in range(WORKER_COUNT)]
    for temporary, final in zip(temporary_paths, final_paths, strict=True):
        replace_with_retry(temporary, final)
    return final_paths


def _write_assignment(output_dir: Path, assignments: list[AssignmentPart]) -> Path:
    schema = pa.schema(
        [
            pa.field("domain", pa.string(), nullable=False),
            pa.field("worker_id", pa.int32(), nullable=False),
            pa.field("split_part", pa.int16(), nullable=False),
            pa.field("split_parts", pa.int16(), nullable=False),
            pa.field("url_count", pa.int64(), nullable=False),
            pa.field("estimated_work", pa.float64(), nullable=False),
            pa.field("per_worker_domain_concurrency", pa.int16(), nullable=True),
        ]
    )
    rows = [
        {
            "domain": part.domain,
            "worker_id": part.worker_id,
            "split_part": part.split_part,
            "split_parts": part.split_parts,
            "url_count": part.url_count,
            "estimated_work": part.estimated_work,
            "per_worker_domain_concurrency": 1 if part.split_parts > 1 else None,
        }
        for part in sorted(assignments, key=lambda value: (value.domain, value.split_part))
    ]
    final = output_dir / "domain_assignment.parquet"
    temporary = final.with_suffix(final.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    replace_with_retry(temporary, final)
    return final


def _write_summary(
    output_dir: Path, assignments: list[AssignmentPart], work_loads: list[float]
) -> Path:
    url_loads = [0] * WORKER_COUNT
    worker_domains: list[set[str]] = [set() for _ in range(WORKER_COUNT)]
    split_domain_count = [0] * WORKER_COUNT
    for part in assignments:
        url_loads[part.worker_id] += part.url_count
        worker_domains[part.worker_id].add(part.domain)
        if part.split_parts > 1:
            split_domain_count[part.worker_id] += 1
    total_urls = sum(url_loads)
    total_work = sum(work_loads)
    final = output_dir / "partition_summary.csv"
    temporary = final.with_suffix(final.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "worker_id",
                "domain_count",
                "url_count",
                "percentage_of_total",
                "estimated_work",
                "percentage_of_total_work",
                "split_domain_count",
            ),
        )
        writer.writeheader()
        for worker_id in range(WORKER_COUNT):
            writer.writerow(
                {
                    "worker_id": worker_id,
                    "domain_count": len(worker_domains[worker_id]),
                    "url_count": url_loads[worker_id],
                    "percentage_of_total": f"{100 * url_loads[worker_id] / total_urls:.6f}",
                    "estimated_work": f"{work_loads[worker_id]:.9f}",
                    "percentage_of_total_work": f"{100 * work_loads[worker_id] / total_work:.6f}",
                    "split_domain_count": split_domain_count[worker_id],
                }
            )
    replace_with_retry(temporary, final)
    return final


def _validate_outputs(
    paths: list[Path],
    total: int,
    connection: sqlite3.Connection,
    assignments: list[AssignmentPart],
) -> dict[str, int]:
    connection.executescript(
        """
        DROP TABLE IF EXISTS output_ids;
        DROP TABLE IF EXISTS output_domains;
        CREATE TABLE output_ids(doc_id INTEGER PRIMARY KEY);
        CREATE TABLE output_domains(domain TEXT NOT NULL, worker_id INTEGER NOT NULL,
                                    row_count INTEGER NOT NULL DEFAULT 0,
                                    PRIMARY KEY(domain, worker_id));
        """
    )
    output_rows = 0
    for worker_id, path in enumerate(paths):
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != PARTITION_SCHEMA:
            raise RuntimeError(f"partition schema mismatch: {path}")
        for batch in parquet.iter_batches(columns=["doc_id", "domain"], batch_size=65_536):
            ids = [int(value) for value in batch.column("doc_id").to_pylist()]
            domains = Counter(str(value) for value in batch.column("domain").to_pylist())
            try:
                connection.executemany(
                    "INSERT INTO output_ids(doc_id) VALUES (?)", ((value,) for value in ids)
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError(f"duplicate doc_id found while validating {path}") from exc
            for domain, count in domains.items():
                connection.execute(
                    """
                    INSERT INTO output_domains(domain, worker_id, row_count) VALUES (?, ?, ?)
                    ON CONFLICT(domain, worker_id) DO UPDATE
                    SET row_count=row_count+excluded.row_count
                    """,
                    (domain, worker_id, count),
                )
            output_rows += len(ids)
        connection.commit()
    if output_rows != total:
        raise RuntimeError(f"partition row total {output_rows} does not match input {total}")
    unique_ids = int(connection.execute("SELECT COUNT(*) FROM output_ids").fetchone()[0])
    if unique_ids != total:
        raise RuntimeError(f"unique partition IDs {unique_ids} do not match input {total}")
    missing = connection.execute(
        "SELECT doc_id FROM input_ids EXCEPT SELECT doc_id FROM output_ids LIMIT 1"
    ).fetchone()
    extra = connection.execute(
        "SELECT doc_id FROM output_ids EXCEPT SELECT doc_id FROM input_ids LIMIT 1"
    ).fetchone()
    if missing or extra:
        raise RuntimeError(
            f"partition doc_id set differs from input: missing={missing}, extra={extra}"
        )

    expected = {(part.domain, part.worker_id): part.url_count for part in assignments}
    observed = {
        (str(row[0]), int(row[1])): int(row[2])
        for row in connection.execute("SELECT domain, worker_id, row_count FROM output_domains")
    }
    if observed != expected:
        raise RuntimeError("partition domain/worker counts do not match the assignment")
    per_domain: dict[str, list[AssignmentPart]] = defaultdict(list)
    for part in assignments:
        per_domain[part.domain].append(part)
    for domain, parts in per_domain.items():
        if len(parts) not in {1, 2} or len({part.worker_id for part in parts}) != len(parts):
            raise RuntimeError(f"invalid worker affinity for domain {domain}")
        if any(part.split_parts != len(parts) for part in parts):
            raise RuntimeError(f"inconsistent split metadata for domain {domain}")
    return {
        "input_rows": total,
        "partition_rows": output_rows,
        "unique_doc_ids": unique_ids,
        "assigned_domains": len(per_domain),
        "split_domains": sum(len(parts) > 1 for parts in per_domain.values()),
    }


def prepare_partitions(
    input_path: Path, output_dir: Path, domain_profile: Path | None = None
) -> dict[str, object]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    domain_profile = domain_profile.resolve() if domain_profile else None
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if domain_profile is not None and not domain_profile.exists():
        raise FileNotFoundError(domain_profile)
    if output_dir.exists():
        raise FileExistsError(f"partition output already exists: {output_dir}")
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(
        prefix=".partition-index-", suffix=".sqlite", dir=stage, delete=False
    ) as handle:
        database = Path(handle.name)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE input_ids(doc_id INTEGER PRIMARY KEY)")
    try:
        total, domain_counts = _scan_input(input_path, connection)
        workloads = _load_workloads(domain_profile, domain_counts)
        assignments, work_loads, balance = _assign_by_work(
            domain_counts,
            workloads,
            allow_splits=domain_profile is not None,
        )
        worker_paths = _write_worker_partitions(input_path, stage, assignments)
        assignment_path = _write_assignment(stage, assignments)
        summary_path = _write_summary(stage, assignments, work_loads)
        validation = _validate_outputs(worker_paths, total, connection, assignments)
        (stage / "balance_metadata.json").write_text(
            json.dumps(balance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        connection.close()
        database.unlink(missing_ok=True)
    stage.rename(output_dir)
    return {
        "input": str(input_path),
        "domain_profile": str(domain_profile) if domain_profile else None,
        "output_dir": str(output_dir),
        "workers": [str(output_dir / path.name) for path in worker_paths],
        "domain_assignment": str(output_dir / assignment_path.name),
        "partition_summary": str(output_dir / summary_path.name),
        "worker_url_counts": [
            sum(part.url_count for part in assignments if part.worker_id == worker_id)
            for worker_id in range(WORKER_COUNT)
        ],
        "worker_estimated_work": work_loads,
        "balance": balance,
        **validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Partition ViBioMIR URLs by domain workload")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("partitions"), type=Path)
    parser.add_argument(
        "--domain-profile",
        type=Path,
        help="balance by estimated_work and allow guarded two-way domain splits",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            prepare_partitions(args.input, args.output_dir, args.domain_profile),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
