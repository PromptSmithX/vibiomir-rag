"""Configuration management for ViBioMIR Dense Retrieval & Fusion experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    strategy: str = "sentence_aware"
    target_tokens: int = 400
    max_total_tokens: int = 512
    overlap_tokens: int = 48
    include_document_title: bool = True
    include_heading_if_available: bool = True
    max_title_tokens: int = 64
    special_tokens_reserve: int = 4
    chunking_version: str = "a1"

    def fingerprint(self) -> str:
        data = f"{self.strategy}_{self.target_tokens}_{self.max_total_tokens}_{self.overlap_tokens}_{self.include_document_title}_{self.chunking_version}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]


@dataclass
class ModelConfig:
    """Configuration for dense embedding model and runtime."""

    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    local_model_path: str | None = None
    offline_mode: bool = False
    device: str = "cuda"
    precision: str = "float16"
    batch_size: int = 16
    max_length: int = 512
    query_instruction: str = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    )
    pooling_method: str = "last_token"
    normalize: bool = True

    def fingerprint(self) -> str:
        data = f"{self.model_id}_{self.pooling_method}_{self.normalize}_{self.max_length}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]


@dataclass
class IndexConfig:
    """Configuration for dense vector indexing."""

    index_type: str = "flat_ip"
    use_faiss: bool = True
    metric: str = "cosine"


@dataclass
class DenseRetrievalConfig:
    """Configuration for dense search and document aggregation."""

    dense_chunk_top_k: int = 500
    document_aggregation: str = "max"
    top_k_docs: int = 1000


@dataclass
class FusionConfig:
    """Configuration for Reciprocal Rank Fusion (RRF)."""

    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    top_k: int = 1000


@dataclass
class SubmissionConfig:
    """Configuration for official ViBioMIR submission exporting."""

    top_docs: int = 10
    top_chunks: int = 10
    chunks_per_doc: int = 1
    include_chunk_order: bool = False
    doc_id_type: str = "int"
    output_filename: str = "submission.json"
    zip_filename: str = "submission.zip"
    output_dir: str = "data/experiments/exp_a1/submission"
    corpus_path: str | None = None


@dataclass
class ExperimentPaths:
    """Default or configured file paths for experiment execution."""

    candidate_doc_ids: str = "data/retrieval/candidates/candidate_doc_ids.parquet"
    bm25_hits: str = "data/retrieval/candidates/retrieval_top_1000.parquet"
    queries: str = "data/source/query.parquet"
    output_dir: str = "data/experiments/exp_a1"


@dataclass
class ExperimentConfig:
    """Top-level configuration tying together all experiment stages."""

    experiment_id: str = "exp_a1"
    description: str = (
        "Experiment A1: Sentence-aware chunking + Qwen3-Embedding-0.6B + Dense retrieval + RRF fusion"
    )
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: DenseRetrievalConfig = field(default_factory=DenseRetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    paths: ExperimentPaths = field(default_factory=ExperimentPaths)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        chunking_data = data.get("chunking", {})
        model_data = data.get("model", {})
        index_data = data.get("index", {})
        retrieval_data = data.get("retrieval", {})
        fusion_data = data.get("fusion", {})
        submission_data = data.get("submission", {})
        paths_data = data.get("paths", {})

        return cls(
            experiment_id=data.get("experiment_id", "exp_a1"),
            description=data.get("description", ""),
            chunking=ChunkingConfig(**chunking_data) if isinstance(chunking_data, dict) else ChunkingConfig(),
            model=ModelConfig(**model_data) if isinstance(model_data, dict) else ModelConfig(),
            index=IndexConfig(**index_data) if isinstance(index_data, dict) else IndexConfig(),
            retrieval=DenseRetrievalConfig(**retrieval_data)
            if isinstance(retrieval_data, dict)
            else DenseRetrievalConfig(),
            fusion=FusionConfig(**fusion_data) if isinstance(fusion_data, dict) else FusionConfig(),
            submission=SubmissionConfig(**submission_data)
            if isinstance(submission_data, dict)
            else SubmissionConfig(),
            paths=ExperimentPaths(**paths_data) if isinstance(paths_data, dict) else ExperimentPaths(),
        )

    @classmethod
    def load_yaml(cls, path: Path | str) -> ExperimentConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {p}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def fingerprint(self) -> str:
        """Deterministic fingerprint representing chunking and model identity."""
        raw = f"{self.experiment_id}:{self.chunking.fingerprint()}:{self.model.fingerprint()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
