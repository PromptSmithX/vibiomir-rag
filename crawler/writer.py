from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from crawler.config import AppConfig
from crawler.manifest import Manifest
from crawler.models import DocumentBatch, DocumentRecord
from crawler.utils.files import replace_with_retry

LOGGER = logging.getLogger(__name__)


def _arrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required; run `uv sync`") from exc
    return pa, pq


def corpus_schema():
    pa, _ = _arrow_modules()
    return pa.schema(
        [
            pa.field("doc_id", pa.int64(), nullable=False),
            pa.field("url", pa.string(), nullable=False),
            pa.field("final_url", pa.string()),
            pa.field("domain", pa.string(), nullable=False),
            pa.field("content_type", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("language", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("http_status", pa.int32()),
            pa.field("crawl_status", pa.string(), nullable=False),
            pa.field("text_chars", pa.int32(), nullable=False),
            pa.field("bytes_downloaded", pa.int64(), nullable=False),
        ]
    )


class ParquetShardWriter:
    def __init__(
        self,
        config: AppConfig,
        manifest: Manifest,
        on_commit: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.on_commit = on_commit
        self.records: list[DocumentRecord] = []
        self.buffer_bytes = 0

    def recover(
        self,
        recover_adapters: bool = True,
        recover_network_errors: bool = True,
    ) -> dict[str, int]:
        _, pq = _arrow_modules()
        committed = 0
        reset = 0
        for row in self.manifest.writing_shards():
            shard_name = row["shard_name"]
            temp_path = Path(row["temp_path"])
            final_path = Path(row["final_path"])
            if final_path.exists():
                try:
                    table = pq.read_table(final_path, columns=["doc_id"])
                except Exception as exc:
                    raise RuntimeError(f"recovery found corrupt shard {final_path}: {exc}") from exc
                actual = sorted(
                    int(value) for value in table.column("doc_id").to_pylist()
                )
                expected = self.manifest.staged_doc_ids(shard_name)
                if actual != expected:
                    raise RuntimeError(
                        f"recovery mismatch for {shard_name}: parquet IDs differ from staging"
                    )
                self.manifest.commit_staged_shard(shard_name)
                temp_path.unlink(missing_ok=True)
                committed += 1
            else:
                temp_path.unlink(missing_ok=True)
                self.manifest.abort_staged_shard(shard_name)
                reset += 1
        reset += self.manifest.recover_unstaged_fetches()
        if recover_adapters:
            from crawler.adapters import get_all_adapters

            adapter_resets = self.manifest.recover_adapter_targets(get_all_adapters())
            reset += sum(adapter_resets.values())
        if recover_network_errors:
            reset += self.manifest.recover_transient_network_errors()
        return {"committed_shards": committed, "reset_targets": reset}

    async def run(self, queue) -> None:
        while True:
            batch = await queue.get()
            try:
                if batch is None:
                    break
                if not isinstance(batch, DocumentBatch):
                    raise TypeError(f"unexpected writer item: {type(batch)!r}")
                self.records.extend(batch.records)
                self.buffer_bytes += batch.approx_bytes
                if (
                    len(self.records) >= self.config.writer.rows_per_shard
                    or self.buffer_bytes >= self.config.writer.max_buffer_bytes
                ):
                    self.flush()
            finally:
                queue.task_done()
        if self.records:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        pa, pq = _arrow_modules()
        records = self.records
        self.records = []
        buffer_bytes = self.buffer_bytes
        self.buffer_bytes = 0

        shard_name = self.manifest.next_shard_name()
        final_path = self.config.paths.corpus_dir / shard_name
        temp_path = self.config.paths.temp_dir / f"{shard_name}.tmp"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        self.manifest.stage_shard(shard_name, temp_path, final_path, records)
        try:
            table = pa.Table.from_pylist(
                [record.parquet_dict() for record in records], schema=corpus_schema()
            )
            pq.write_table(
                table,
                temp_path,
                compression=self.config.writer.compression,
                use_dictionary=True,
                row_group_size=self.config.writer.row_group_size,
                write_statistics=True,
            )
            # Windows' CRT requires a writable descriptor for os.fsync/_commit.
            with temp_path.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            check = pq.ParquetFile(temp_path)
            try:
                if (
                    check.metadata.num_rows != len(records)
                    or check.schema_arrow != corpus_schema()
                ):
                    raise RuntimeError(f"validation failed for temporary shard {temp_path}")
            finally:
                check.close()
            replace_with_retry(temp_path, final_path)
            if os.name == "posix":
                directory_fd = os.open(final_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            self.manifest.commit_staged_shard(shard_name)
            if self.on_commit:
                self.on_commit(len(records), buffer_bytes)
            LOGGER.info(
                "committed shard %s with %d rows",
                shard_name,
                len(records),
                extra={"event": "shard"},
            )
        except Exception:
            LOGGER.exception("failed to write shard %s", shard_name)
            raise


def verify_corpus(config: AppConfig, manifest: Manifest) -> dict[str, object]:
    _, pq = _arrow_modules()
    expected_schema = corpus_schema()
    shard_rows = {
        row["shard_name"]: int(row["row_count"])
        for row in manifest.connection.execute(
            "SELECT shard_name, row_count FROM shards WHERE status='COMMITTED'"
        )
    }
    seen: set[int] = set()
    problems: list[str] = []
    total_rows = 0
    for shard_name, expected_rows in sorted(shard_rows.items()):
        path = config.paths.corpus_dir / shard_name
        if not path.exists():
            problems.append(f"missing shard: {shard_name}")
            continue
        try:
            parquet = pq.ParquetFile(path)
            if parquet.schema_arrow != expected_schema:
                problems.append(f"schema mismatch: {shard_name}")
            if parquet.metadata.num_rows != expected_rows:
                problems.append(
                    f"row count mismatch: {shard_name} expected={expected_rows} "
                    f"actual={parquet.metadata.num_rows}"
                )
            for batch in parquet.iter_batches(columns=["doc_id"], batch_size=65_536):
                for doc_id in batch.column(0).to_pylist():
                    value = int(doc_id)
                    if value in seen:
                        problems.append(f"duplicate doc_id in corpus: {value}")
                    seen.add(value)
                    total_rows += 1
        except Exception as exc:
            problems.append(f"cannot read {shard_name}: {type(exc).__name__}: {exc}")

    terminal_with_shard = {
        int(row[0])
        for row in manifest.connection.execute(
            "SELECT doc_id FROM crawl_tasks WHERE output_shard IS NOT NULL"
        )
    }
    missing = sorted(terminal_with_shard - seen)[:100]
    extra = sorted(seen - terminal_with_shard)[:100]
    if missing:
        problems.append(f"manifest IDs missing from corpus (first 100): {missing}")
    if extra:
        problems.append(f"corpus IDs missing from manifest (first 100): {extra}")
    return {
        "ok": not problems,
        "shards": len(shard_rows),
        "rows": total_rows,
        "manifest_rows_with_shard": len(terminal_with_shard),
        "problems": problems,
    }
