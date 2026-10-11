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


@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk from a candidate document with full attribution."""

    chunk_id: str
    doc_id: int
    chunk_index: int
    text: str
    embed_text: str
    title: str | None = None
    heading: str | None = None
    language: str | None = None
    token_count: int = 0
    chunking_strategy: str = "sentence_aware"
    chunking_version: str = "a1"


@dataclass(frozen=True)
class ChunkHit:
    """Dense retrieval match at the chunk level."""

    chunk_id: str
    doc_id: int
    rank: int
    score: float


@dataclass(frozen=True)
class DenseDocHit:
    """Document-level hit after aggregating chunk scores."""

    query_id: int
    doc_id: int
    rank: int
    score: float
    best_chunk_id: str = ""
    best_chunk_score: float = 0.0


@dataclass(frozen=True)
class FusedRetrievalHit:
    """Final merged result from BM25 and Dense RRF fusion."""

    query_id: int
    doc_id: int
    final_rank: int
    fused_score: float
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    dense_best_chunk_id: str | None = None


@dataclass
class ChunkStats:
    """Execution statistics for chunking phase."""

    total_docs_processed: int = 0
    total_chunks_created: int = 0
    total_tokens: int = 0
    avg_tokens_per_chunk: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class EmbeddingStats:
    """Execution statistics for embedding generation."""

    total_chunks_embedded: int = 0
    embedding_dimension: int = 0
    elapsed_seconds: float = 0.0
    throughput_chunks_per_sec: float = 0.0


@dataclass
class DenseRetrievalStats:
    """Execution statistics for dense retrieval."""

    num_queries: int = 0
    top_k_chunks: int = 500
    search_time_seconds: float = 0.0
    queries_per_second: float = 0.0


@dataclass
class FusionStats:
    """Execution statistics for BM25 + Dense RRF fusion."""

    num_queries: int = 0
    rrf_k: int = 60
    total_fused_hits: int = 0
    unique_docs: int = 0
    bm25_only_docs: int = 0
    dense_only_docs: int = 0
    both_systems_docs: int = 0

