"""Unit tests for DenseIndex, NumPy vector search fallback, and document-level score aggregation."""

from pathlib import Path
import numpy as np

from retrieval.dense.config import DenseRetrievalConfig
from retrieval.dense.index import DenseIndex, NumpyVectorIndex


def test_numpy_vector_index_search():
    """Verify that pure NumPy vector search returns correct cosine similarity and rankings."""
    dimension = 4
    index = NumpyVectorIndex(dimension=dimension)

    # 3 vectors
    v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)
    vectors = np.vstack([v0, v1, v2])
    index.add(vectors)

    # Query identical to v0
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    scores, indices = index.search(q, top_k=3)

    assert len(scores) == 1
    assert len(indices) == 1
    # v0 has score 1.0 (rank 1), v2 has score ~0.7071 (rank 2), v1 has score 0.0 (rank 3)
    assert indices[0][0] == 0
    assert indices[0][1] == 2
    assert indices[0][2] == 1
    assert np.isclose(scores[0][0], 1.0, atol=1e-4)
    assert np.isclose(scores[0][1], 0.7071, atol=1e-3)


def test_dense_index_document_aggregation_max():
    """Verify document-level aggregation collapses multiple chunks of the same doc_id to max score."""
    dimension = 4
    backend = NumpyVectorIndex(dimension=dimension)

    # 4 chunks:
    # chunk 0: doc 101, score ~ 0.9
    # chunk 1: doc 101, score ~ 0.5
    # chunk 2: doc 202, score ~ 0.8
    # chunk 3: doc 303, score ~ 0.2
    v0 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.2, 0.1, 0.0, 0.0], dtype=np.float32)
    backend.add(np.vstack([v0, v1, v2, v3]))

    chunk_ids = ["101_c0", "101_c1", "202_c0", "303_c0"]
    doc_ids = [101, 101, 202, 303]

    dense_index = DenseIndex(backend, chunk_ids=chunk_ids, doc_ids=doc_ids)

    query_vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    query_ids = [1]
    ret_config = DenseRetrievalConfig(dense_chunk_top_k=4, document_aggregation="max", top_k_docs=10)

    doc_hits, stats = dense_index.search_and_aggregate_documents(query_ids, query_vectors, ret_config)

    # Total 3 unique documents (doc 101, doc 202, doc 303)
    assert len(doc_hits) == 3
    # Check deduplicated doc_ids
    returned_doc_ids = [h.doc_id for h in doc_hits]
    assert returned_doc_ids == [101, 202, 303]

    # Rank 1 must be doc 101 with max score from chunk 0
    assert doc_hits[0].rank == 1
    assert doc_hits[0].doc_id == 101
    assert np.isclose(doc_hits[0].score, 0.9, atol=1e-3)
    assert doc_hits[0].best_chunk_id == "101_c0"

    # Rank 2 must be doc 202
    assert doc_hits[1].rank == 2
    assert doc_hits[1].doc_id == 202
    assert np.isclose(doc_hits[1].score, 0.8, atol=1e-3)
    assert doc_hits[1].best_chunk_id == "202_c0"


def test_dense_index_save_and_load(tmp_path: Path):
    """Verify index can be persisted to disk and loaded with intact ID mappings."""
    dimension = 2
    backend = NumpyVectorIndex(dimension=dimension)
    backend.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    chunk_ids = ["doc1_c0", "doc2_c0"]
    doc_ids = [1001, 1002]

    index = DenseIndex(backend, chunk_ids=chunk_ids, doc_ids=doc_ids)
    index.save(tmp_path)

    # Reload from disk
    loaded_index = DenseIndex.load(tmp_path)
    assert loaded_index.chunk_ids == chunk_ids
    assert loaded_index.doc_ids == doc_ids

    # Query loaded index
    q = np.array([[0.0, 1.0]], dtype=np.float32)
    chunk_hits = loaded_index.search_chunks(q, top_k_chunks=1)
    assert len(chunk_hits[0]) == 1
    assert chunk_hits[0][0].chunk_id == "doc2_c0"
    assert chunk_hits[0][0].doc_id == 1002
