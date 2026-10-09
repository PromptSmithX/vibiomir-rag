"""Streaming reader for Parquet corpus shards across worker directories."""

import logging
from collections.abc import Generator
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class CorpusReader:
    """Streams documents from Parquet files across one or more corpus directories."""

    def __init__(
        self,
        corpus_dirs: list[Path | str],
        only_success: bool = True,
        min_text_chars: int = 1,
    ) -> None:
        self.corpus_dirs = [Path(p) for p in corpus_dirs]
        self.only_success = only_success
        self.min_text_chars = min_text_chars

    def get_shard_paths(self) -> list[Path]:
        """Find all parquet shard files across configured corpus directories."""
        all_shards: list[Path] = []
        for cdir in self.corpus_dirs:
            if not cdir.exists():
                logger.warning("Corpus directory does not exist: %s", cdir)
                continue
            shards = sorted(cdir.glob("*.parquet"))
            all_shards.extend(shards)
        return all_shards

    def count_total_rows(self) -> int:
        """Rapidly count total rows using Parquet metadata without reading data."""
        total = 0
        for shard in self.get_shard_paths():
            try:
                meta = pq.read_metadata(shard)
                total += meta.num_rows
            except Exception as e:
                logger.error("Failed to read metadata for %s: %s", shard, e)
        return total

    def iter_docs(self, batch_size: int = 4096) -> Generator[tuple[int, str, str], None, None]:
        """Stream documents yielding (doc_id, title, text).

        Yields:
            doc_id (int): Official doc_id from competition dataset.
            title (str): Title of document or empty string.
            text (str): Clean text content.
        """
        shards = self.get_shard_paths()
        logger.info("Found %d shards across %d directories", len(shards), len(self.corpus_dirs))

        for _shard_idx, shard_path in enumerate(shards, 1):
            try:
                # Inspect schema columns
                schema = pq.read_schema(shard_path)
                has_crawl_status = "crawl_status" in schema.names
                has_curation_status = "curation_status" in schema.names

                columns = ["doc_id", "title", "text"]
                if self.only_success and has_crawl_status:
                    columns.append("crawl_status")
                if has_curation_status:
                    columns.append("curation_status")

                tbl = pq.read_table(shard_path, columns=columns)

                ids = tbl.column("doc_id").to_pylist()
                titles = tbl.column("title").to_pylist()
                texts = tbl.column("text").to_pylist()

                crawl_statuses = (
                    tbl.column("crawl_status").to_pylist()
                    if (self.only_success and has_crawl_status)
                    else None
                )
                curation_statuses = (
                    tbl.column("curation_status").to_pylist() if has_curation_status else None
                )

                for i in range(len(ids)):
                    doc_id = ids[i]
                    if doc_id is None:
                        continue

                    # Status filters
                    if crawl_statuses and crawl_statuses[i] != "SUCCESS":
                        continue
                    if curation_statuses and curation_statuses[i] == "LOW_QUALITY":
                        # Note: we only skip empty/corrupt if explicitly marked
                        continue

                    title = (titles[i] or "").strip()
                    text = (texts[i] or "").strip()

                    # Valid document check: must have at least some text or title
                    if len(text) < self.min_text_chars and not title:
                        continue

                    yield int(doc_id), title, text

            except Exception as e:
                logger.error("Error streaming shard %s: %s", shard_path, e)
                continue
