from dataclasses import dataclass, field


@dataclass
class RawDoc:
    """Document as read from the raw crawl parquet shards."""

    doc_id: int
    url: str
    final_url: str | None = None
    domain: str | None = None
    content_type: str | None = None
    title: str | None = None
    text: str | None = None
    language: str | None = None
    content_hash: str | None = None
    http_status: int | None = None
    crawl_status: str | None = None
    text_chars: int | None = None
    bytes_downloaded: int | None = None


@dataclass
class CuratedDoc:
    """Document after going through the curation pipeline."""

    doc_id: int
    url: str
    final_url: str | None
    domain: str
    content_type: str
    title: str
    text: str
    language: str
    content_hash: str
    http_status: int
    crawl_status: str
    text_chars: int
    raw_text_chars: int
    removed_chars: int
    curation_status: str  # "CLEAN", "LOW_QUALITY", "EMPTY"


@dataclass
class CurationStats:
    """Summary metrics of a curation execution."""

    shards_processed: int = 0
    docs_read: int = 0
    docs_curated: int = 0
    docs_low_quality: int = 0
    docs_skipped_non_success: int = 0
    total_raw_chars: int = 0
    total_clean_chars: int = 0
    total_removed_chars: int = 0
    domain_counts: dict[str, int] = field(default_factory=dict)
    domain_removed_chars: dict[str, int] = field(default_factory=dict)
