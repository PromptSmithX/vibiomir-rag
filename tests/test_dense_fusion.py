"""Unit tests for ReciprocalRankFusion (BM25 + Dense RRF).

Tests formula correctness, missing ranks, single-system candidates,
ranking stability, and dual Parquet/CSV export.
"""

from pathlib import Path

from retrieval.dense.config import FusionConfig
from retrieval.dense.fusion import ReciprocalRankFusion
from retrieval.models import DenseDocHit, RetrievalHit


def test_rrf_formula_and_merging():
    """Verify RRF formula calculation with both systems."""
    cfg = FusionConfig(rrf_k=60, bm25_weight=1.0, dense_weight=1.0)
    fusion = ReciprocalRankFusion(cfg)

    # Query 1:
    # Doc 1: BM25 rank 1, Dense rank 2
    # Doc 2: BM25 rank 2, Dense rank 1
    # Doc 3: BM25 rank 3, not in Dense (Dense-missing)
    # Doc 4: not in BM25, Dense rank 3 (BM25-missing)
    bm25_hits = [
        RetrievalHit(query_id=1, doc_id=10, rank=1, bm25_score=15.0),
        RetrievalHit(query_id=1, doc_id=20, rank=2, bm25_score=12.0),
        RetrievalHit(query_id=1, doc_id=30, rank=3, bm25_score=10.0),
    ]
    dense_hits = [
        DenseDocHit(query_id=1, doc_id=20, rank=1, score=0.95, best_chunk_id="20_c0"),
        DenseDocHit(query_id=1, doc_id=10, rank=2, score=0.90, best_chunk_id="10_c0"),
        DenseDocHit(query_id=1, doc_id=40, rank=3, score=0.85, best_chunk_id="40_c0"),
    ]

    fused_hits, stats = fusion.fuse(bm25_hits=bm25_hits, dense_hits=dense_hits)

    # All 4 documents must be present in fused result
    doc_ids_in_result = [h.doc_id for h in fused_hits]
    assert set(doc_ids_in_result) == {10, 20, 30, 40}

    # Expected scores:
    # doc 10: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # doc 20: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.032522
    # doc 30 (BM25 only): 1/(60+3) = 1/63 = 0.015873
    # doc 40 (Dense only): 1/(60+3) = 1/63 = 0.015873
    score_10 = next(h.fused_score for h in fused_hits if h.doc_id == 10)
    score_20 = next(h.fused_score for h in fused_hits if h.doc_id == 20)
    score_30 = next(h.fused_score for h in fused_hits if h.doc_id == 30)
    score_40 = next(h.fused_score for h in fused_hits if h.doc_id == 40)

    assert abs(score_10 - (1 / 61 + 1 / 62)) < 1e-6
    assert abs(score_20 - (1 / 62 + 1 / 61)) < 1e-6
    assert abs(score_30 - (1 / 63)) < 1e-6
    assert abs(score_40 - (1 / 63)) < 1e-6

    # Both doc 10 and 20 must rank higher than single-system docs 30 and 40
    assert score_10 > score_30
    assert score_20 > score_40

    # Check stats accounting
    assert stats.num_queries == 1
    assert stats.bm25_only_docs == 1
    assert stats.dense_only_docs == 1
    assert stats.both_systems_docs == 2


def test_rrf_preserves_official_doc_id():
    """Verify official doc_ids and ranking indices are preserved without truncation."""
    cfg = FusionConfig(rrf_k=60)
    fusion = ReciprocalRankFusion(cfg)

    official_id_1 = 9223372036854775807  # max int64
    official_id_2 = 1234567890123

    bm25 = [RetrievalHit(query_id=101, doc_id=official_id_1, rank=1, bm25_score=20.0)]
    dense = [DenseDocHit(query_id=101, doc_id=official_id_2, rank=1, score=0.99)]

    fused, stats = fusion.fuse(bm25_hits=bm25, dense_hits=dense)

    assert len(fused) == 2
    fused_doc_ids = {h.doc_id for h in fused}
    assert official_id_1 in fused_doc_ids
    assert official_id_2 in fused_doc_ids


def test_export_fused_results(tmp_path: Path):
    """Verify that fused results export cleanly to both Parquet and CSV."""
    cfg = FusionConfig(rrf_k=60)
    fusion = ReciprocalRankFusion(cfg)

    bm25 = [RetrievalHit(query_id=1, doc_id=100, rank=1, bm25_score=10.0)]
    dense = [DenseDocHit(query_id=1, doc_id=100, rank=1, score=0.9, best_chunk_id="100_c0")]

    fused, _ = fusion.fuse(bm25_hits=bm25, dense_hits=dense)

    pq_path, csv_path = ReciprocalRankFusion.export_fused_results(
        fused, output_dir=tmp_path, filename_prefix="test_fused"
    )

    assert pq_path.exists()
    assert csv_path.exists()
    assert pq_path.stat().st_size > 0
    assert csv_path.stat().st_size > 0

    # Verify CSV content has header and rows
    content = csv_path.read_text(encoding="utf-8")
    assert "query_id,doc_id,final_rank,fused_score" in content
    assert "1,100,1," in content
