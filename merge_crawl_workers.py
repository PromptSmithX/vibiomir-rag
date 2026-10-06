from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from crawler.utils.files import replace_with_retry
from crawler.writer import corpus_schema

STATUS_RANK = {
    "ROBOTS_DENIED": 0,
    "FAILED": 1,
    "UNSUPPORTED": 2,
    "TOO_LARGE": 3,
    "NEEDS_JS": 4,
    "NEEDS_OCR": 5,
    "EMPTY_CONTENT": 6,
    "SUCCESS": 7,
}
SHARD_PATTERN = re.compile(r"part-(\d+)\.parquet")
ROWS_PER_SHARD = 10_000


@dataclass(frozen=True)
class SourceShard:
    worker_id: int
    root: Path
    path: Path
    source_index: int
    sequence: int


def _parse_inputs(values: list[str]) -> dict[int, Path]:
    inputs: dict[int, Path] = {}
    for value in values:
        worker_text, separator, path_text = value.partition("=")
        if not separator or not path_text:
            raise ValueError(f"input must have form WORKER_ID=PATH: {value!r}")
        try:
            worker_id = int(worker_text)
        except ValueError as exc:
            raise ValueError(f"invalid worker ID: {worker_text!r}") from exc
        if worker_id in inputs:
            raise ValueError(f"worker {worker_id} was supplied more than once")
        inputs[worker_id] = Path(path_text).resolve()
    if set(inputs) != set(range(6)):
        raise ValueError("exactly one input for each worker ID 0 through 5 is required")
    return inputs


def _open_manifests(inputs: dict[int, Path]) -> dict[int, sqlite3.Connection]:
    connections: dict[int, sqlite3.Connection] = {}
    try:
        for worker_id, root in inputs.items():
            manifest_path = root / "manifest.sqlite"
            if worker_id == 0 and not manifest_path.exists():
                manifest_path = root / "crawl_manifest.sqlite"
            if not manifest_path.exists():
                raise FileNotFoundError(f"worker {worker_id} manifest missing: {root}")
            connection = sqlite3.connect(f"{manifest_path.as_uri()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            connections[worker_id] = connection
        return connections
    except BaseException:
        for connection in connections.values():
            connection.close()
        raise


def _source_shards(inputs: dict[int, Path]) -> list[SourceShard]:
    shards: list[SourceShard] = []
    for worker_id in range(6):
        root = inputs[worker_id]
        corpus = root / "corpus"
        if not corpus.is_dir():
            raise FileNotFoundError(f"worker {worker_id} corpus missing: {corpus}")
        paths = sorted(corpus.glob("part-*.parquet"))
        for path in paths:
            match = SHARD_PATTERN.fullmatch(path.name)
            if match is None:
                raise ValueError(f"invalid shard name: {path}")
            shards.append(SourceShard(worker_id, root, path, len(shards), int(match.group(1))))
    return shards


def _validate_committed_shards(
    inputs: dict[int, Path], manifests: dict[int, sqlite3.Connection]
) -> None:
    for worker_id, manifest in manifests.items():
        corpus = inputs[worker_id] / "corpus"
        for name, expected_rows in manifest.execute(
            "SELECT shard_name,row_count FROM shards WHERE status='COMMITTED'"
        ):
            path = corpus / str(name)
            if not path.is_file():
                raise FileNotFoundError(f"worker {worker_id} committed shard missing: {path}")
            actual_rows = pq.ParquetFile(path).metadata.num_rows
            if actual_rows != int(expected_rows):
                raise RuntimeError(
                    f"worker {worker_id} shard {name} has {actual_rows} rows, "
                    f"manifest expects {expected_rows}"
                )


def _iter_rows(shard: SourceShard, columns: list[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(shard.path)
    if parquet.schema_arrow != corpus_schema():
        raise ValueError(f"corpus schema mismatch: {shard.path}")
    row_number = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=16_384):
        for row in batch.to_pylist():
            yield row_number, row
            row_number += 1


def _candidate_key(row: dict[str, Any], shard: SourceShard) -> tuple[int, int, int, int, int]:
    status = str(row["crawl_status"])
    if status not in STATUS_RANK:
        raise ValueError(f"unexpected terminal status {status!r} in {shard.path}")
    return (
        STATUS_RANK[status],
        int(bool(row.get("content_hash"))),
        int(row.get("text_chars") or 0),
        shard.sequence,
        -shard.source_index,
    )


def _create_index(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        """CREATE TABLE winners (
            doc_id INTEGER PRIMARY KEY,
            url TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            row_number INTEGER NOT NULL,
            status TEXT NOT NULL,
            status_rank INTEGER NOT NULL,
            has_hash INTEGER NOT NULL,
            text_chars INTEGER NOT NULL,
            sequence INTEGER NOT NULL,
            emitted INTEGER NOT NULL DEFAULT 0
        )"""
    )
    return connection


def _index_winners(
    shards: list[SourceShard],
    manifests: dict[int, sqlite3.Connection],
    index: sqlite3.Connection,
    conflict_path: Path,
) -> tuple[int, int]:
    scanned = 0
    conflicts = 0
    columns = ["doc_id", "url", "crawl_status", "content_hash", "text_chars"]
    with conflict_path.open("w", encoding="utf-8", newline="\n") as conflict_log:
        for shard in shards:
            manifest = manifests[shard.worker_id]
            for row_number, row in _iter_rows(shard, columns):
                scanned += 1
                doc_id = int(row["doc_id"])
                url = str(row["url"])
                source = manifest.execute(
                    "SELECT url FROM crawl_tasks WHERE doc_id=?", (doc_id,)
                ).fetchone()
                if source is None or str(source["url"]) != url:
                    raise ValueError(
                        f"worker {shard.worker_id} doc_id {doc_id} does not match original URL"
                    )
                key = _candidate_key(row, shard)
                current = index.execute(
                    "SELECT * FROM winners WHERE doc_id=?", (doc_id,)
                ).fetchone()
                if current is None:
                    index.execute(
                        """INSERT INTO winners(
                            doc_id,url,source_index,row_number,status,status_rank,
                            has_hash,text_chars,sequence
                        ) VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            doc_id,
                            url,
                            shard.source_index,
                            row_number,
                            row["crawl_status"],
                            key[0],
                            key[1],
                            key[2],
                            key[3],
                        ),
                    )
                else:
                    if str(current["url"]) != url:
                        raise ValueError(
                            f"doc_id {doc_id} has conflicting original URLs: "
                            f"{current['url']!r} vs {url!r}"
                        )
                    previous_shard = shards[int(current["source_index"])]
                    previous_key = (
                        int(current["status_rank"]),
                        int(current["has_hash"]),
                        int(current["text_chars"]),
                        int(current["sequence"]),
                        -int(current["source_index"]),
                    )
                    candidate_wins = key > previous_key
                    conflict_log.write(
                        json.dumps(
                            {
                                "doc_id": doc_id,
                                "url": url,
                                "previous": {
                                    "worker_id": previous_shard.worker_id,
                                    "shard": previous_shard.path.name,
                                    "row_number": int(current["row_number"]),
                                    "status": current["status"],
                                },
                                "candidate": {
                                    "worker_id": shard.worker_id,
                                    "shard": shard.path.name,
                                    "row_number": row_number,
                                    "status": row["crawl_status"],
                                },
                                "selected": "candidate" if candidate_wins else "previous",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    conflicts += 1
                    if candidate_wins:
                        index.execute(
                            """UPDATE winners SET
                                source_index=?,row_number=?,status=?,status_rank=?,
                                has_hash=?,text_chars=?,sequence=? WHERE doc_id=?""",
                            (
                                shard.source_index,
                                row_number,
                                row["crawl_status"],
                                key[0],
                                key[1],
                                key[2],
                                key[3],
                                doc_id,
                            ),
                        )
                if scanned % 25_000 == 0:
                    index.commit()
            index.commit()
    return scanned, conflicts


def _write_merged_corpus(
    shards: list[SourceShard], index: sqlite3.Connection, corpus_dir: Path
) -> tuple[int, int, Counter[str]]:
    corpus_dir.mkdir(parents=True, exist_ok=False)
    buffer: list[dict[str, Any]] = []
    output_rows = 0
    output_shards = 0
    statuses: Counter[str] = Counter()

    def flush() -> None:
        nonlocal output_shards
        if not buffer:
            return
        final = corpus_dir / f"part-{output_shards:06d}.parquet"
        temporary = final.with_suffix(final.suffix + ".tmp")
        table = pa.Table.from_pylist(buffer, schema=corpus_schema())
        pq.write_table(table, temporary, compression="zstd", row_group_size=2048)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        check = pq.ParquetFile(temporary)
        try:
            if check.metadata.num_rows != len(buffer) or check.schema_arrow != corpus_schema():
                raise RuntimeError(f"temporary merged shard failed validation: {temporary}")
        finally:
            check.close()
        replace_with_retry(temporary, final)
        output_shards += 1
        buffer.clear()

    columns = corpus_schema().names
    for shard in shards:
        for row_number, row in _iter_rows(shard, columns):
            doc_id = int(row["doc_id"])
            winner = index.execute(
                "SELECT source_index,row_number,emitted FROM winners WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if winner is None:
                raise RuntimeError(f"merge index lost doc_id {doc_id}")
            if int(winner[0]) != shard.source_index or int(winner[1]) != row_number:
                continue
            if int(winner[2]):
                raise RuntimeError(f"duplicate selected doc_id {doc_id}")
            index.execute("UPDATE winners SET emitted=1 WHERE doc_id=?", (doc_id,))
            buffer.append(row)
            statuses[str(row["crawl_status"])] += 1
            output_rows += 1
            if len(buffer) >= ROWS_PER_SHARD:
                flush()
            if output_rows % 25_000 == 0:
                index.commit()
        index.commit()
    flush()
    index.commit()
    return output_rows, output_shards, statuses


def merge_workers(
    inputs: dict[int, Path], output_dir: Path, *, allow_partial: bool = False
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"merge output already exists: {output_dir}")
    if any(root == output_dir or root in output_dir.parents for root in inputs.values()):
        raise ValueError("merge output must be outside worker input directories")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    manifests = _open_manifests(inputs)
    index = _create_index(stage / "merge_index.sqlite")
    try:
        manifest_totals: dict[str, dict[str, int]] = {}
        for worker_id, manifest in manifests.items():
            total, committed = manifest.execute(
                "SELECT COUNT(*), COUNT(output_shard) FROM crawl_tasks"
            ).fetchone()
            manifest_totals[str(worker_id)] = {
                "total": int(total),
                "committed": int(committed),
            }
            if not allow_partial and committed != total:
                raise RuntimeError(
                    f"worker {worker_id} is incomplete: {committed}/{total} "
                    "documents committed; use --allow-partial for an interim merge"
                )
        _validate_committed_shards(inputs, manifests)
        shards = _source_shards(inputs)
        scanned, conflicts = _index_winners(
            shards, manifests, index, stage / "merge_conflicts.jsonl"
        )
        unique_docs = int(index.execute("SELECT COUNT(*) FROM winners").fetchone()[0])
        output_rows, output_shards, statuses = _write_merged_corpus(shards, index, stage / "corpus")
        emitted = int(index.execute("SELECT COUNT(*) FROM winners WHERE emitted=1").fetchone()[0])
        if output_rows != unique_docs or emitted != unique_docs:
            raise RuntimeError(
                f"merge output mismatch: rows={output_rows}, emitted={emitted}, "
                f"unique_doc_ids={unique_docs}"
            )
        checked = sum(
            pq.ParquetFile(path).metadata.num_rows
            for path in sorted((stage / "corpus").glob("part-*.parquet"))
        )
        if checked != unique_docs:
            raise RuntimeError(f"output Parquet row count {checked} != {unique_docs}")
        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "inputs": {str(worker_id): str(inputs[worker_id]) for worker_id in range(6)},
            "source_shards": len(shards),
            "source_rows": scanned,
            "conflicts": conflicts,
            "unique_doc_ids": unique_docs,
            "output_rows": output_rows,
            "output_shards": output_shards,
            "status_counts": dict(statuses),
            "manifest_rows": manifest_totals,
            "partial": any(
                value["committed"] != value["total"] for value in manifest_totals.values()
            ),
        }
        (stage / "merge_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        index.close()
        for manifest in manifests.values():
            manifest.close()
    (stage / "merge_index.sqlite").unlink(missing_ok=True)
    stage.rename(output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge six ViBioMIR crawler outputs")
    parser.add_argument("--input", action="append", required=True, metavar="WORKER_ID=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = _parse_inputs(args.input)
    print(
        json.dumps(
            merge_workers(inputs, args.output_dir, allow_partial=args.allow_partial),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
