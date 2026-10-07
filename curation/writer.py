import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from curation.models import CuratedDoc


class ParquetCorpusWriter:
    """Atomic writer for saving curated documents into Parquet shards."""

    SCHEMA = pa.schema([
        ("doc_id", pa.int64()),
        ("url", pa.string()),
        ("final_url", pa.string()),
        ("domain", pa.string()),
        ("content_type", pa.string()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("language", pa.string()),
        ("content_hash", pa.string()),
        ("http_status", pa.int32()),
        ("crawl_status", pa.string()),
        ("text_chars", pa.int32()),
        ("raw_text_chars", pa.int32()),
        ("removed_chars", pa.int32()),
        ("curation_status", pa.string()),
    ])

    def __init__(self, output_dir: Path | str, compression: str = "zstd") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def write_shard(self, shard_name: str, docs: list[CuratedDoc]) -> Path:
        """Atomically write a list of CuratedDoc objects to a Parquet file."""
        target_path = self.output_dir / shard_name
        tmp_path = self.output_dir / f"{shard_name}.tmp.{os.getpid()}"

        data = {
            "doc_id": [d.doc_id for d in docs],
            "url": [d.url for d in docs],
            "final_url": [d.final_url or "" for d in docs],
            "domain": [d.domain for d in docs],
            "content_type": [d.content_type for d in docs],
            "title": [d.title for d in docs],
            "text": [d.text for d in docs],
            "language": [d.language for d in docs],
            "content_hash": [d.content_hash for d in docs],
            "http_status": [d.http_status for d in docs],
            "crawl_status": [d.crawl_status for d in docs],
            "text_chars": [d.text_chars for d in docs],
            "raw_text_chars": [d.raw_text_chars for d in docs],
            "removed_chars": [d.removed_chars for d in docs],
            "curation_status": [d.curation_status for d in docs],
        }

        table = pa.Table.from_pydict(data, schema=self.SCHEMA)

        # Write to temporary file first, then atomically rename
        pq.write_table(table, tmp_path, compression=self.compression)
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)

        return target_path
