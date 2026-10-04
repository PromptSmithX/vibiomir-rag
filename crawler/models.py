from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class CrawlStatus(StrEnum):
    PENDING = "PENDING"
    FETCHING = "FETCHING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TOO_LARGE = "TOO_LARGE"
    UNSUPPORTED = "UNSUPPORTED"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    ROBOTS_DENIED = "ROBOTS_DENIED"
    NEEDS_JS = "NEEDS_JS"
    NEEDS_OCR = "NEEDS_OCR"


TERMINAL_STATUSES = {
    CrawlStatus.SUCCESS,
    CrawlStatus.FAILED,
    CrawlStatus.TOO_LARGE,
    CrawlStatus.UNSUPPORTED,
    CrawlStatus.EMPTY_CONTENT,
    CrawlStatus.ROBOTS_DENIED,
    CrawlStatus.NEEDS_JS,
    CrawlStatus.NEEDS_OCR,
}


@dataclass(slots=True)
class FetchTarget:
    fetch_key: str
    request_url: str
    domain: str
    attempts: int


@dataclass(slots=True)
class FetchOutcome:
    target: FetchTarget
    status: CrawlStatus
    final_url: str | None = None
    content_type: str | None = None
    content: bytes | None = None
    http_status: int | None = None
    bytes_downloaded: int = 0
    retryable: bool = False
    retry_at: float | None = None
    error_type: str | None = None
    error: str | None = None


@dataclass(slots=True)
class ExtractedContent:
    status: CrawlStatus
    title: str | None = None
    text: str | None = None
    language: str | None = None
    content_hash: str | None = None
    extractor: str | None = None
    error_type: str | None = None
    error: str | None = None


@dataclass(slots=True)
class SourceDocument:
    doc_id: int
    url: str
    domain: str


@dataclass(slots=True)
class DocumentRecord:
    doc_id: int
    url: str
    final_url: str | None
    domain: str
    content_type: str | None
    title: str | None
    text: str | None
    language: str | None
    content_hash: str | None
    http_status: int | None
    crawl_status: str
    text_chars: int
    bytes_downloaded: int
    fetch_key: str = field(repr=False, compare=False)
    error_type: str | None = field(default=None, repr=False, compare=False)
    error: str | None = field(default=None, repr=False, compare=False)

    def parquet_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("fetch_key")
        value.pop("error_type")
        value.pop("error")
        return value


@dataclass(slots=True)
class DocumentBatch:
    records: list[DocumentRecord]

    @property
    def approx_bytes(self) -> int:
        return sum(
            len(record.text.encode("utf-8")) if record.text else 0 for record in self.records
        )

