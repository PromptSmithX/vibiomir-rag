from pathlib import Path

import pyarrow.parquet as pq

from curation.audit.sampler import AuditSamplePair


class ParquetBinaryLookup:
    """Zero-conversion binary Parquet lookup by URL or doc_id."""

    def __init__(self, raw_corpus_dir: Path | str, clean_corpus_dir: Path | str) -> None:
        self.raw_dir = Path(raw_corpus_dir)
        self.clean_dir = Path(clean_corpus_dir)

    def find_by_url(self, target_url: str) -> AuditSamplePair | None:
        """Search clean corpus for target URL and return paired raw vs clean."""
        clean_shards = sorted(self.clean_dir.glob("*.parquet"))
        clean_url_norm = target_url.strip()

        for clean_shard in clean_shards:
            # Read only 'url' and 'doc_id' for fast binary scanning
            tbl_urls = pq.read_table(clean_shard, columns=["doc_id", "url"])
            urls = tbl_urls.column("url").to_pylist()
            if clean_url_norm in urls:
                row_idx = urls.index(clean_url_norm)
                doc_id = tbl_urls.column("doc_id")[row_idx].as_py()
                return self._load_paired(clean_shard, row_idx, doc_id)

        return None

    def find_by_doc_id(self, target_doc_id: int) -> AuditSamplePair | None:
        """Search clean corpus for target doc_id and return paired raw vs clean."""
        clean_shards = sorted(self.clean_dir.glob("*.parquet"))

        for clean_shard in clean_shards:
            tbl_ids = pq.read_table(clean_shard, columns=["doc_id"])
            doc_ids = tbl_ids.column("doc_id").to_pylist()
            if target_doc_id in doc_ids:
                row_idx = doc_ids.index(target_doc_id)
                return self._load_paired(clean_shard, row_idx, target_doc_id)

        return None

    def _load_paired(self, clean_shard: Path, row_idx: int, doc_id: int) -> AuditSamplePair:
        """Read full clean row and matching raw row."""
        tbl_clean = pq.read_table(clean_shard)
        clean_row = {col: tbl_clean.column(col)[row_idx].as_py() for col in tbl_clean.column_names}

        raw_shard = self.raw_dir / clean_shard.name
        raw_text = clean_row.get("text", "")
        if raw_shard.exists():
            tbl_raw = pq.read_table(raw_shard, columns=["doc_id", "text"])
            r_ids = tbl_raw.column("doc_id").to_pylist()
            if doc_id in r_ids:
                r_idx = r_ids.index(doc_id)
                raw_text = tbl_raw.column("text")[r_idx].as_py() or ""

        clean_text = clean_row.get("text", "")
        clean_c = clean_row.get("text_chars", len(clean_text))
        raw_c = clean_row.get("raw_text_chars", len(raw_text))
        rem_c = clean_row.get("removed_chars", max(0, raw_c - clean_c))
        rem_pct = round((rem_c / raw_c * 100), 2) if raw_c > 0 else 0.0

        return AuditSamplePair(
            doc_id=doc_id,
            domain=clean_row.get("domain", ""),
            url=clean_row.get("url", ""),
            title=clean_row.get("title", ""),
            raw_text=raw_text,
            clean_text=clean_text,
            raw_chars=raw_c,
            clean_chars=clean_c,
            removed_chars=rem_c,
            removed_pct=rem_pct,
        )
