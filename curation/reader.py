from collections.abc import Generator
from pathlib import Path

import pyarrow.parquet as pq

from curation.models import RawDoc


class ParquetCorpusReader:
    """Reader for streaming documents from raw Parquet corpus shards."""

    def __init__(self, corpus_dir: Path | str, only_success: bool = True) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.only_success = only_success

    def get_shard_paths(self) -> list[Path]:
        """Return sorted list of parquet shards in corpus dir."""
        if not self.corpus_dir.exists():
            return []
        return sorted(self.corpus_dir.glob("*.parquet"))

    def read_shard(self, shard_path: Path) -> list[RawDoc]:
        """Read a single parquet shard into a list of RawDoc objects."""
        tbl = pq.read_table(shard_path)
        docs: list[RawDoc] = []

        cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
        num_rows = tbl.num_rows

        for i in range(num_rows):
            status = cols.get("crawl_status", [None])[i]
            if self.only_success and status != "SUCCESS":
                continue

            doc = RawDoc(
                doc_id=cols["doc_id"][i],
                url=cols["url"][i],
                final_url=cols.get("final_url", [None])[i],
                domain=cols.get("domain", [None])[i],
                content_type=cols.get("content_type", [None])[i],
                title=cols.get("title", [None])[i],
                text=cols.get("text", [None])[i],
                language=cols.get("language", [None])[i],
                content_hash=cols.get("content_hash", [None])[i],
                http_status=cols.get("http_status", [None])[i],
                crawl_status=status,
                text_chars=cols.get("text_chars", [None])[i],
                bytes_downloaded=cols.get("bytes_downloaded", [None])[i],
            )
            docs.append(doc)

        return docs

    def iter_docs(self) -> Generator[tuple[Path, list[RawDoc]], None, None]:
        """Yield (shard_path, list_of_raw_docs) shard by shard."""
        for shard_path in self.get_shard_paths():
            yield shard_path, self.read_shard(shard_path)
