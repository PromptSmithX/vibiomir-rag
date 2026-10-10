from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryItem:
    """Official query from BTC."""

    query_id: int
    query_text: str


@dataclass(frozen=True)
class RetrievalHit:
    """Single retrieval result item for a query."""

    query_id: int
    doc_id: int
    rank: int
    bm25_score: float


@dataclass
class UnionCandidate:
    """Unique document candidate across queries with attribution."""

    doc_id: int
    query_ids: list[int] = field(default_factory=list)
    best_rank: int = 0
    best_score: float = 0.0
    query_matches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateLevelStats:
    """Summary statistics for candidates at a given top-K cutoff."""

    k: int
    total_candidates: int
    unique_docs: int
    unique_ratio: float
    avg_queries_per_doc: float


@dataclass
class IndexStats:
    """Metadata and execution stats of the BM25 index."""

    total_docs: int = 0
    total_shards: int = 0
    index_size_bytes: int = 0
    build_time_seconds: float = 0.0
    title_boost: float = 2.0
    corpus_sources: list[str] = field(default_factory=list)
    created_at: str = ""
