"""ViBioMIR Dense Retrieval Package for Experiment A and future extensions."""

from retrieval.dense.candidate_loader import CandidateChunkingPipeline, load_candidate_doc_ids
from retrieval.dense.chunker import BaseChunker, ChunkerRegistry, SentenceAwareChunker
from retrieval.dense.config import (
    ChunkingConfig,
    DenseRetrievalConfig,
    ExperimentConfig,
    FusionConfig,
    IndexConfig,
    ModelConfig,
)
from retrieval.dense.embedder import DenseEmbedder
from retrieval.dense.fusion import ReciprocalRankFusion
from retrieval.dense.index import DenseIndex, FaissVectorIndex, NumpyVectorIndex
from retrieval.dense.model_loader import DenseModelLoader, ModelMetadata
from retrieval.dense.pipeline import ExperimentAPipeline

__all__ = [
    "BaseChunker",
    "CandidateChunkingPipeline",
    "ChunkerRegistry",
    "ChunkingConfig",
    "DenseEmbedder",
    "DenseIndex",
    "DenseModelLoader",
    "DenseRetrievalConfig",
    "ExperimentAPipeline",
    "ExperimentConfig",
    "FaissVectorIndex",
    "FusionConfig",
    "IndexConfig",
    "ModelConfig",
    "ModelMetadata",
    "NumpyVectorIndex",
    "ReciprocalRankFusion",
    "SentenceAwareChunker",
    "load_candidate_doc_ids",
]
