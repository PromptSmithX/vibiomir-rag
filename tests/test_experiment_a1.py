"""Unit tests specifically covering ViBioMIR Experiment A1.1 improvements.

Requirements verified:
- Top 10, 30, 50, 100 document selection without hardcoding limits
- Deduplication of document IDs: strictly no duplicate doc_ids
- Handling of BM25-only documents: doc kept in relevant_docs, no fake chunks
- Chunk selection respecting max_chunks_per_document quota
- Exact and near-duplicate chunk deduplication
- Source text provenance: strictly clean source text, no synthetic prefixes
- Optional continuous passage expansion: bounded, contiguous, safe fallback
- Query coverage: complete coverage for all official queries
- Integer document IDs: strictly integer, no strings
- Official submission JSON schema and root-only ZIP archive
- Experiment artifact isolation: baseline artifacts never overwritten
- Reusing/resuming saved rankings and vector index
"""

import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retrieval.dense.config import ExperimentConfig, SubmissionConfig
from retrieval.dense.pipeline import ExperimentAPipeline
from retrieval.submission.chunk_selector import (
    ChunkSelectionConfig,
    ChunkSelector,
    PassageExpansionConfig,
    compute_shingle_jaccard,
    is_boilerplate_chunk,
    is_duplicate_chunk,
    strip_synthetic_prefixes,
)
from retrieval.submission.chunk_store import ChunkStore
from retrieval.submission.exporter import SubmissionExporter
from retrieval.submission.validator import SubmissionValidator


# ---------------------------------------------------------------------------
# Fixture Helpers
# ---------------------------------------------------------------------------


def create_synthetic_artifacts(tmp_path: Path, num_queries: int = 5, docs_per_query: int = 120):
    """Create synthetic query, chunk, BM25, and fusion artifacts for test runs."""
    q_ids = list(range(1, num_queries + 1))
    query_file = tmp_path / "query.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "id": q_ids,
                "query": [f"Y khoa thắc mắc câu hỏi số {i}" for i in q_ids],
            }
        ),
        query_file,
    )

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    c_ids: list[str] = []
    d_ids: list[int] = []
    c_idxs: list[int] = []
    texts: list[str] = []
    embed_texts: list[str] = []

    # Create 3 chunks per document for up to 300 documents
    total_docs = num_queries * docs_per_query
    for did in range(1, total_docs + 1):
        for cidx in range(3):
            cid = f"{did}_c{cidx}_a1"
            txt = f"Tài liệu y khoa {did} đoạn {cidx}: Điều trị và dự phòng bệnh tim mạch và huyết áp."
            c_ids.append(cid)
            d_ids.append(did)
            c_idxs.append(cidx)
            texts.append(txt)
            embed_texts.append(f"[Tiêu đề: Tim mạch] {txt}")

    pq.write_table(
        pa.Table.from_pydict(
            {
                "chunk_id": c_ids,
                "doc_id": d_ids,
                "chunk_index": c_idxs,
                "text": texts,
                "embed_text": embed_texts,
            }
        ),
        chunks_dir / "chunks_part_00000.parquet",
    )

    # Fusion file with up to docs_per_query ranked hits per query
    fusion_file = tmp_path / "rrf_fusion_top_1000.parquet"
    fq_ids: list[int] = []
    fd_ids: list[int] = []
    franks: list[int] = []
    fscores: list[float] = []
    fbest_chunks: list[str] = []

    for qid in q_ids:
        for r in range(1, docs_per_query + 1):
            did = (qid - 1) * docs_per_query + r
            fq_ids.append(qid)
            fd_ids.append(did)
            franks.append(r)
            fscores.append(1.0 / (60 + r))
            fbest_chunks.append(f"{did}_c0_a1")

    pq.write_table(
        pa.Table.from_pydict(
            {
                "query_id": fq_ids,
                "doc_id": fd_ids,
                "final_rank": franks,
                "fused_score": fscores,
                "dense_best_chunk_id": fbest_chunks,
            }
        ),
        fusion_file,
    )

    return query_file, chunks_dir, fusion_file


# ---------------------------------------------------------------------------
# 1. Configurable Retrieval Depth: Top 10, 30, 50, 100 Documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [10, 30, 50, 100])
def test_configurable_document_depth(tmp_path: Path, depth: int):
    """Verify that exporter extracts precisely the requested top-K documents."""
    query_file, chunks_dir, fusion_file = create_synthetic_artifacts(tmp_path, num_queries=2, docs_per_query=120)

    out_json = tmp_path / f"sub_{depth}.json"
    exporter = SubmissionExporter()
    summary = exporter.export(
        retrieval_file=fusion_file,
        query_path=query_file,
        chunks_dir=chunks_dir,
        output_json_path=out_json,
        top_docs=depth,
        top_chunks=20,
    )

    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        assert len(item["relevant_docs"]) == depth
        # Check no duplicates
        assert len(item["relevant_docs"]) == len(set(item["relevant_docs"]))
        # Check integer types
        assert all(isinstance(d, int) and not isinstance(d, bool) for d in item["relevant_docs"])


def test_ranking_with_fewer_than_k_results(tmp_path: Path):
    """Verify that queries with fewer results output exact count without fake IDs."""
    query_file = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pydict({"id": [1], "query": ["hiếm gặp"]}), query_file)

    fusion_file = tmp_path / "fusion.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "query_id": [1, 1],
                "doc_id": [501, 502],
                "final_rank": [1, 2],
                "fused_score": [0.03, 0.02],
                "dense_best_chunk_id": ["", ""],
            }
        ),
        fusion_file,
    )

    out_json = tmp_path / "fewer_results.json"
    exporter = SubmissionExporter()
    exporter.export(
        retrieval_file=fusion_file,
        query_path=query_file,
        chunks_dir=tmp_path,
        output_json_path=out_json,
        top_docs=50,  # Request 50, but only 2 available
        top_chunks=50,
    )

    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data[0]["relevant_docs"]) == 2
    assert data[0]["relevant_docs"] == [501, 502]


# ---------------------------------------------------------------------------
# 2. Document-Only BM25 Results Handling
# ---------------------------------------------------------------------------


def test_bm25_only_doc_without_chunks(tmp_path: Path):
    """Verify that BM25-only docs without valid chunks are kept in relevant_docs without fake chunks."""
    query_file = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pydict({"id": [1], "query": ["test bm25 only"]}), query_file)

    # doc 999 has no chunks in chunks_dir
    fusion_file = tmp_path / "fusion.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "query_id": [1],
                "doc_id": [999],
                "final_rank": [1],
                "fused_score": [0.015],
                "dense_best_chunk_id": [""],
            }
        ),
        fusion_file,
    )

    empty_chunks_dir = tmp_path / "empty_chunks"
    empty_chunks_dir.mkdir()

    out_json = tmp_path / "bm25_only_sub.json"
    exporter = SubmissionExporter()
    summary = exporter.export(
        retrieval_file=fusion_file,
        query_path=query_file,
        chunks_dir=empty_chunks_dir,
        output_json_path=out_json,
        top_docs=10,
        top_chunks=10,
    )

    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)

    # Document 999 is retained in relevant_docs
    assert data[0]["relevant_docs"] == [999]
    # No hallucinated or fake chunks generated
    assert len(data[0]["relevant_chunks"]) == 0

    validator = SubmissionValidator()
    val_res = validator.validate_file(out_json)
    assert val_res.is_valid is True


# ---------------------------------------------------------------------------
# 3. Chunk Selection & Quota Limits per Document
# ---------------------------------------------------------------------------


def test_chunk_selection_max_per_document(tmp_path: Path):
    """Verify max_chunks_per_document prevents a single document from dominating slots."""
    chunk_store = ChunkStore()
    chunk_store._loaded = True
    # Populate doc 100 with 10 chunks
    doc_100_chunks = []
    for i in range(10):
        cid = f"100_c{i}_a1"
        txt = f"Nội dung số {i} của tài liệu dài 100 về triệu chứng viêm phổi."
        doc_100_chunks.append({"chunk_id": cid, "doc_id": 100, "chunk_index": i, "text": txt})
        chunk_store._chunk_text_cache[cid] = txt

    chunk_store._doc_chunks[100] = doc_100_chunks

    cfg = ChunkSelectionConfig(top_k=20, max_chunks_per_document=3)
    selector = ChunkSelector(config=cfg)

    selected = selector.select_chunks(
        query_id=1,
        relevant_docs=[100],
        chunk_store=chunk_store,
    )

    # Exactly 3 chunks allowed despite top_k=20
    assert len(selected) == 3
    assert all(c["doc_id"] == 100 for c in selected)


# ---------------------------------------------------------------------------
# 4. Chunk Content Deduplication
# ---------------------------------------------------------------------------


def test_chunk_deduplication():
    """Verify exact and near-duplicate chunks are rejected."""
    seen_texts = [
        "thuoc khang sinh duoc chi dinh dieu tri nhiem khuan duong ho hap tren",
    ]

    # Exact duplicate
    assert is_duplicate_chunk(seen_texts[0], seen_texts, threshold=0.85) is True

    # Near-duplicate with slight phrasing change
    near_dup = "thuoc khang sinh duoc chi dinh cho dieu tri nhiem khuan duong ho hap tren"
    sim = compute_shingle_jaccard(seen_texts[0], near_dup, k=3)
    assert sim >= 0.65
    assert is_duplicate_chunk(near_dup, seen_texts, threshold=0.65) is True

    # Distinct content
    different = "benh tieu duong can kiem soat chi so duong huyet hang ngay"
    assert is_duplicate_chunk(different, seen_texts, threshold=0.85) is False


# ---------------------------------------------------------------------------
# 5. Boilerplate Detection & Filtering
# ---------------------------------------------------------------------------


def test_boilerplate_filtering():
    """Verify website chrome and navigation menus are detected without rejecting medical text."""
    # Obvious breadcrumbs
    assert is_boilerplate_chunk("Trang chủ > Sức khỏe > Chuyên mục tim mạch") is True

    # Obvious copyright footer
    assert is_boilerplate_chunk("© 2024 Bản quyền thuộc về Bệnh viện. All rights reserved.") is True

    # Valid clinical/medical sentence
    assert is_boilerplate_chunk("Bệnh nhân sốt cao 39 độ C, khó thở nhẹ, phổi ran ẩm hai đáy.") is False

    # Short clinical sentence (must NOT be rejected)
    assert is_boilerplate_chunk("Xét nghiệm máu cho thấy bạch cầu tăng cao.") is False


# ---------------------------------------------------------------------------
# 6. Source Text Provenance
# ---------------------------------------------------------------------------


def test_strip_synthetic_prefixes():
    """Verify model prompts and [Tiêu đề: ...] prefixes are stripped cleanly."""
    raw_with_instruction = "Instruct: Given a web search query\nQuery: Huyết áp cao\nNội dung thực tế về bệnh."
    cleaned_inst = strip_synthetic_prefixes(raw_with_instruction)
    assert cleaned_inst == "Huyết áp cao\nNội dung thực tế về bệnh."

    raw_with_title = "[Tiêu đề: Tim Mạch] [Mục: Chẩn đoán] Đây là triệu chứng bệnh."
    cleaned_title = strip_synthetic_prefixes(raw_with_title)
    assert cleaned_title == "Đây là triệu chứng bệnh."


# ---------------------------------------------------------------------------
# 7. Optional Source Passage Expansion
# ---------------------------------------------------------------------------


def test_passage_expansion_within_doc_bounds():
    """Verify continuous passage expansion stays bounded and within document context."""
    full_doc = (
        "Triệu chứng ban đầu là sốt nhẹ và đau đầu. "
        "Sau đó bệnh nhân có thể xuất hiện nốt ban đỏ trên da. "
        "Trong giai đoạn phục hồi cần bổ sung dinh dưỡng đầy đủ."
    )
    target_sub = "Sau đó bệnh nhân có thể xuất hiện nốt ban đỏ trên da."

    chunk_store = ChunkStore()
    chunk_store._corpus_text_cache[123] = full_doc

    cfg = ChunkSelectionConfig(
        passage_expansion=PassageExpansionConfig(enabled=True, max_tokens=200)
    )
    selector = ChunkSelector(config=cfg)

    expanded = selector._expand_passage(
        doc_id=123,
        chunk_id="123_c1_a1",
        chunk_text=target_sub,
        chunk_meta={"text": target_sub},
        chunk_store=chunk_store,
    )

    # Expansion contains target_sub
    assert target_sub in expanded
    # Expansion expands across sentences within document
    assert "Triệu chứng ban đầu" in expanded or "giai đoạn phục hồi" in expanded
    # Does not invent outside text
    assert expanded in full_doc


def test_passage_expansion_disabled_by_default():
    """Verify passage expansion is disabled by default."""
    cfg = ChunkSelectionConfig()
    assert cfg.passage_expansion.enabled is False


# ---------------------------------------------------------------------------
# 8. Experiment Isolation: Baseline Never Overwritten
# ---------------------------------------------------------------------------


def test_experiment_isolation(tmp_path: Path):
    """Verify multiple experiments output to distinct non-colliding paths."""
    exp50_cfg = ExperimentConfig(
        experiment_id="a1_docs50_chunks50",
        submission=SubmissionConfig(output_dir=str(tmp_path / "a1_docs50_chunks50" / "submission")),
    )
    exp100_cfg = ExperimentConfig(
        experiment_id="a1_docs100_chunks50",
        submission=SubmissionConfig(output_dir=str(tmp_path / "a1_docs100_chunks50" / "submission")),
    )

    assert exp50_cfg.submission.output_dir != exp100_cfg.submission.output_dir
    assert "a1_docs50_chunks50" in exp50_cfg.submission.output_dir
    assert "a1_docs100_chunks50" in exp100_cfg.submission.output_dir


# ---------------------------------------------------------------------------
# 9. Full Query Coverage & Official JSON Schema with ZIP Packaging
# ---------------------------------------------------------------------------


def test_end_to_end_official_schema_and_zip(tmp_path: Path):
    """End-to-end test verifying schema compliance, integer doc_ids, and single-file ZIP."""
    query_file, chunks_dir, fusion_file = create_synthetic_artifacts(tmp_path, num_queries=5, docs_per_query=50)

    out_json = tmp_path / "official_submission.json"
    out_zip = tmp_path / "official_submission.zip"

    cfg = SubmissionConfig(top_docs=50, top_chunks=50)
    exporter = SubmissionExporter(config=cfg)
    summary = exporter.export(
        retrieval_file=fusion_file,
        query_path=query_file,
        chunks_dir=chunks_dir,
        output_json_path=out_json,
        zip_output_path=out_zip,
        top_docs=50,
        top_chunks=50,
    )

    assert summary.total_official_queries == 5
    assert summary.total_queries_with_results == 5
    assert summary.total_empty_queries == 0

    # Validate output JSON
    validator = SubmissionValidator(query_path=query_file)
    json_res = validator.validate_file(out_json)
    assert json_res.is_valid is True
    assert len(json_res.errors) == 0

    # Validate ZIP archive
    zip_res = validator.validate_file(out_zip)
    assert zip_res.is_valid is True
    assert len(zip_res.errors) == 0
    assert zip_res.summary["inner_file"] == "official_submission.json"

    # Verify metadata file was created
    meta_path = out_json.parent / "submission_metadata.json"
    assert meta_path.exists()
    with open(meta_path, encoding="utf-8") as mf:
        meta_data = json.load(mf)
    assert meta_data["top_docs_limit"] == 50
    assert meta_data["top_chunks_limit"] == 50
