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
    export_chunk_hits: bool = True


@dataclass
class FusionConfig:
    """Configuration for Reciprocal Rank Fusion (RRF)."""

    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    top_k: int = 1000


@dataclass
class PassageExpansionConfig:
    """Configuration for optional continuous source passage expansion."""

    enabled: bool = False
    max_tokens: int = 1024
    anchor: str = "retrieved_chunk"
    tokenizer_name: str | None = None


@dataclass
class ChunkSelectionConfig:
    """Configuration for relevant chunk selection strategy."""

    top_k: int = 50
    max_chunks_per_document: int = 3
    deduplicate: bool = True
    dedup_threshold: float = 0.85
    preserve_original_text: bool = True
    filter_boilerplate: bool = True
    min_chunk_chars: int = 30
    passage_expansion: PassageExpansionConfig = field(default_factory=PassageExpansionConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkSelectionConfig:
        pe_data = data.get("passage_expansion", {})
        if isinstance(pe_data, dict):
            pe_cfg = PassageExpansionConfig(
                enabled=bool(pe_data.get("enabled", False)),
                max_tokens=int(pe_data.get("max_tokens", 1024)),
                anchor=str(pe_data.get("anchor", "retrieved_chunk")),
                tokenizer_name=pe_data.get("tokenizer_name"),
            )
        else:
            pe_cfg = PassageExpansionConfig()
        return cls(
            top_k=data.get("top_k", 50),
            max_chunks_per_document=data.get("max_chunks_per_document", 3),
            deduplicate=data.get("deduplicate", True),
            dedup_threshold=data.get("dedup_threshold", 0.85),
            preserve_original_text=data.get("preserve_original_text", True),
            filter_boilerplate=data.get("filter_boilerplate", True),
            min_chunk_chars=data.get("min_chunk_chars", 30),
            passage_expansion=pe_cfg,
        )


@dataclass
class SubmissionConfig:
    """Configuration for official ViBioMIR submission exporting."""

    top_docs: int = 50
    top_chunks: int = 50
    chunks_per_doc: int = 3
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
    index_dir: str | None = None
    chunks_dir: str | None = None
    fusion_hits: str | None = None
    dense_hits: str | None = None
    cached_query_embeddings: str | None = None


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
    chunk_selection: ChunkSelectionConfig = field(default_factory=ChunkSelectionConfig)
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
        chunking_data = dict(data.get("chunking", {}))
        model_data = dict(data.get("model", {}))
        index_data = dict(data.get("index", {}))
        retrieval_data = dict(data.get("retrieval", {}))
        fusion_data = dict(data.get("fusion", {}))
        submission_data = dict(data.get("submission", {}))
        chunk_sel_data = dict(data.get("chunk_selection", {}))
        paths_data = dict(data.get("paths", {}))

        # Support both experiment: {name: ...} and experiment_id: ...
        exp_id = data.get("experiment_id")
        if not exp_id and "experiment" in data:
            exp_val = data["experiment"]
            if isinstance(exp_val, dict):
                exp_id = exp_val.get("name") or exp_val.get("experiment_id")
            elif isinstance(exp_val, str):
                exp_id = exp_val
        if not exp_id:
            exp_id = "exp_a1"

        # Align chunk_selection.max_chunks_per_document and submission.chunks_per_doc
        if "max_chunks_per_document" in chunk_sel_data:
            if "chunks_per_doc" not in submission_data:
                submission_data["chunks_per_doc"] = chunk_sel_data["max_chunks_per_document"]
        elif "chunks_per_doc" in submission_data:
            chunk_sel_data["max_chunks_per_document"] = submission_data["chunks_per_doc"]

        # Align submission.top_chunks and chunk_selection.top_k
        if "top_chunks" in submission_data and "top_k" not in chunk_sel_data:
            chunk_sel_data["top_k"] = submission_data["top_chunks"]

        return cls(
            experiment_id=exp_id,
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
            chunk_selection=ChunkSelectionConfig.from_dict(chunk_sel_data)
            if isinstance(chunk_sel_data, dict)
            else ChunkSelectionConfig(),
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
