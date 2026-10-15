"""Unit tests for Experiment A1.2 — Optimize Chunk Selection (Max 2 Chunks per Document).

Verifies all 10 experiment requirements:
1. No document in submission has more than 2 chunks when max_chunks_per_document = 2.
2. Advance to subsequent documents when per-document cap of 2 is reached.
3. Baseline behavior (cap 3) is reproducible when max_chunks_per_document = 3.
4. Top 100 relevant_docs are strictly preserved (100% identical ranking and order).
5. Chunk selection is deterministic across multiple runs.
6. No duplicate or near-duplicate chunks are selected.
7. Correct chunk-to-doc mapping and provenance.
8. Candidate pool under capacity: returns valid candidates without inventing chunks, reports deficit.
9. Integer document IDs strictly enforced in submission schema.
10. Experiment isolation: candidate outputs do not overwrite baseline artifacts.
11. Diagnostic coverage calculation and baseline vs A1.2 comparison accuracy.
12. Validator catches cap violations when max_chunks_per_doc is specified.
13. pubmed: prefix stripping and rejection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retrieval.dense.config import (
    ChunkSelectionConfig,
    ExperimentConfig,
    SubmissionConfig,
)
from retrieval.submission.chunk_selector import (
    ChunkSelector,
    strip_synthetic_prefixes,
)
from retrieval.submission.chunk_store import ChunkStore
from retrieval.submission.diagnostics import (
    analyze_submission,
    compare_submissions,
    verify_ranking_preservation,
)
from retrieval.submission.exporter import SubmissionExporter
from retrieval.submission.validator import SubmissionValidator


# ---------------------------------------------------------------------------
# Test Fixtures & Synthetic Data Helpers
# ---------------------------------------------------------------------------


def create_synthetic_query_file(file_path: Path, query_ids: list[int]) -> Path:
    """Create query.parquet for specified query IDs."""
    tbl = pa.Table.from_pydict(
        {
            "id": query_ids,
            "query": [f"Truy vấn y khoa số {qid}" for qid in query_ids],
        }
    )
    pq.write_table(tbl, file_path)
    return file_path


def create_synthetic_chunk_store(
    chunks_dir: Path,
    doc_chunk_map: dict[int, list[str]],
) -> ChunkStore:
    """Create chunk parquet shards with specified text per document."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_cids: list[str] = []
    all_dids: list[int] = []
    all_idxs: list[int] = []
    all_texts: list[str] = []

    for did, texts in doc_chunk_map.items():
        for idx, txt in enumerate(texts):
            cid = f"{did}_c{idx}_a1"
            all_cids.append(cid)
            all_dids.append(did)
            all_idxs.append(idx)
            all_texts.append(txt)

    tbl = pa.Table.from_pydict(
        {
            "chunk_id": all_cids,
            "doc_id": all_dids,
            "chunk_index": all_idxs,
            "text": all_texts,
        }
    )
    pq.write_table(tbl, chunks_dir / "chunks_part_00000.parquet")

    store = ChunkStore(chunks_dir=chunks_dir)
    store.preload_for_retrieval(
        needed_chunk_ids=set(all_cids),
        needed_doc_ids=set(all_dids),
    )
    return store


def create_synthetic_fusion_file(
    file_path: Path,
    query_id: int,
    doc_ids: list[int],
) -> Path:
    """Create rrf_fusion parquet file for a query with specified doc_ids."""
    n = len(doc_ids)
    tbl = pa.Table.from_pydict(
        {
            "query_id": [query_id] * n,
            "doc_id": doc_ids,
            "final_rank": list(range(1, n + 1)),
            "fused_score": [1.0 / (60 + r) for r in range(1, n + 1)],
            "dense_best_chunk_id": [f"{did}_c0_a1" for did in doc_ids],
        }
    )
    pq.write_table(tbl, file_path)
    return file_path


# ---------------------------------------------------------------------------
# 1. No Document Exceeds 2 Chunks When Cap = 2
# ---------------------------------------------------------------------------


def test_no_document_has_more_than_2_chunks(tmp_path: Path):
    """Verify that under max_chunks_per_document = 2, no document receives > 2 chunks."""
    chunks_dir = tmp_path / "chunks"
    # Create 5 chunks per document for 10 documents
    doc_chunk_map = {
        did: [f"Nội dung chunk {i} của tài liệu y khoa {did}." for i in range(5)]
        for did in range(1001, 1011)
    }
    store = create_synthetic_chunk_store(chunks_dir, doc_chunk_map)

    cfg = ChunkSelectionConfig(top_k=20, max_chunks_per_document=2)
    selector = ChunkSelector(cfg)

    rel_docs = list(range(1001, 1011))
    selected = selector.select_chunks(
        query_id=1,
        relevant_docs=rel_docs,
        chunk_store=store,
    )

    assert len(selected) == 20  # 10 docs * 2 chunks each = 20

    # Verify per-document counts
    doc_counts: dict[int, int] = {}
    for c in selected:
        did = c["doc_id"]
        doc_counts[did] = doc_counts.get(did, 0) + 1

    for did, count in doc_counts.items():
        assert count <= 2, f"Document {did} received {count} chunks, exceeding cap of 2!"

    # Verify diagnostics recorded cap skips
    diag = selector.last_query_diagnostics
    assert diag is not None
    assert diag.selected_chunks == 20
    assert diag.unique_docs == 10
    # Docs 1001..1009 each had 3 excess chunks skipped (9 * 3 = 27). Doc 1010 hit quota 20 on chunk 1.
    assert diag.chunks_skipped_cap == 27


# ---------------------------------------------------------------------------
# 2. Advance to Next Document When Cap Reached
# ---------------------------------------------------------------------------


def test_advance_to_next_document_when_cap_reached(tmp_path: Path):
    """Verify that when a document reaches 2 chunks, selector moves to candidates of next docs."""
    chunks_dir = tmp_path / "chunks"
    doc_chunk_map = {
        100: [
            f"Đoạn văn bản y khoa chi tiết thứ {i} thuộc tài liệu 100 với đầy đủ nội dung chuyên sâu."
            for i in range(10)
        ],
        200: [
            f"Đoạn văn bản y khoa chi tiết thứ {i} thuộc tài liệu 200 với đầy đủ nội dung chuyên sâu."
            for i in range(10)
        ],
        300: [
            f"Đoạn văn bản y khoa chi tiết thứ {i} thuộc tài liệu 300 với đầy đủ nội dung chuyên sâu."
            for i in range(10)
        ],
    }
    store = create_synthetic_chunk_store(chunks_dir, doc_chunk_map)

    cfg = ChunkSelectionConfig(top_k=6, max_chunks_per_document=2)
    selector = ChunkSelector(cfg)

    selected = selector.select_chunks(
        query_id=10,
        relevant_docs=[100, 200, 300],
        chunk_store=store,
    )

    assert len(selected) == 6
    doc_ids_sequence = [c["doc_id"] for c in selected]
    # Doc 100 takes 2, then Doc 200 takes 2, then Doc 300 takes 2
    assert doc_ids_sequence == [100, 100, 200, 200, 300, 300]


# ---------------------------------------------------------------------------
# 3. Cap 3 Reproduces Baseline Behavior
# ---------------------------------------------------------------------------


def test_cap_3_reproduces_baseline_behavior(tmp_path: Path):
    """Verify that setting max_chunks_per_document = 3 reproduces baseline allocation."""
    chunks_dir = tmp_path / "chunks"
    doc_chunk_map = {
        100: [
            f"Đoạn văn bản nghiên cứu lâm sàng số {i} của tài liệu 100 với đầy đủ thông tin chi tiết."
            for i in range(10)
        ],
        200: [
            f"Đoạn văn bản nghiên cứu lâm sàng số {i} của tài liệu 200 với đầy đủ thông tin chi tiết."
            for i in range(10)
        ],
    }
    store = create_synthetic_chunk_store(chunks_dir, doc_chunk_map)

    cfg = ChunkSelectionConfig(top_k=6, max_chunks_per_document=3)
    selector = ChunkSelector(cfg)

    selected = selector.select_chunks(
        query_id=10,
        relevant_docs=[100, 200],
        chunk_store=store,
    )

    assert len(selected) == 6
    doc_ids_sequence = [c["doc_id"] for c in selected]
    # Doc 100 takes 3, Doc 200 takes 3
    assert doc_ids_sequence == [100, 100, 100, 200, 200, 200]


# ---------------------------------------------------------------------------
# 4. Top 100 Relevant Docs Strictly Preserved
# ---------------------------------------------------------------------------


def test_top_100_relevant_docs_strictly_preserved(tmp_path: Path):
    """Verify that relevant_docs is 100% identical in order and content between baseline & A1.2."""
    q_file = create_synthetic_query_file(tmp_path / "query.parquet", [1, 2])

    # 100 documents per query
    doc_ids_q1 = list(range(1001, 1101))
    doc_ids_q2 = list(range(2001, 2101))

    fusion_file = tmp_path / "rrf_fusion_top_1000.parquet"
    tbl = pa.Table.from_pydict(
        {
            "query_id": [1] * 100 + [2] * 100,
            "doc_id": doc_ids_q1 + doc_ids_q2,
            "final_rank": list(range(1, 101)) + list(range(1, 101)),
            "fused_score": [1.0 / (60 + r) for r in range(1, 101)] * 2,
            "dense_best_chunk_id": [f"{did}_c0_a1" for did in (doc_ids_q1 + doc_ids_q2)],
        }
    )
    pq.write_table(tbl, fusion_file)

    # Chunks for all docs
    chunks_dir = tmp_path / "chunks"
    doc_chunk_map = {did: [f"Nội dung chunk cho doc {did}."] for did in (doc_ids_q1 + doc_ids_q2)}
    create_synthetic_chunk_store(chunks_dir, doc_chunk_map)

    # Run Baseline (Cap 3)
    base_out = tmp_path / "baseline_a1" / "submission.json"
    exporter_base = SubmissionExporter()
    exporter_base.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        output_json_path=base_out,
        top_docs=100,
        top_chunks=100,
        max_chunks_per_doc=3,
    )

    # Run Exp A1.2 (Cap 2)
    cand_out = tmp_path / "a1_2_max2_chunks" / "submission.json"
    exporter_cand = SubmissionExporter()
    exporter_cand.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        output_json_path=cand_out,
        top_docs=100,
        top_chunks=100,
        max_chunks_per_doc=2,
    )

    # Verify ranking preservation
    ranking_ok, mismatches = verify_ranking_preservation(base_out, cand_out)
    assert ranking_ok is True
    assert len(mismatches) == 0

    with open(base_out, encoding="utf-8") as f:
        base_records = json.load(f)
    with open(cand_out, encoding="utf-8") as f:
        cand_records = json.load(f)

    for b, c in zip(base_records, cand_records, strict=True):
        assert b["id"] == c["id"]
        assert len(b["relevant_docs"]) == 100
        assert len(c["relevant_docs"]) == 100
        assert b["relevant_docs"] == c["relevant_docs"]  # 100% identical in order and content!


# ---------------------------------------------------------------------------
# 5. Chunk Selection Deterministic
# ---------------------------------------------------------------------------


def test_chunk_selection_deterministic(tmp_path: Path):
    """Verify that repeated chunk selection on the same inputs produces identical results."""
    chunks_dir = tmp_path / "chunks"
    doc_chunk_map = {
        did: [f"Nội dung y khoa biến thể {i} của tài liệu {did}" for i in range(4)]
        for did in range(500, 520)
    }
    store = create_synthetic_chunk_store(chunks_dir, doc_chunk_map)

    cfg = ChunkSelectionConfig(top_k=30, max_chunks_per_document=2)
    selector_1 = ChunkSelector(cfg)
    selector_2 = ChunkSelector(cfg)

    rel_docs = list(range(500, 520))
    res_1 = selector_1.select_chunks(query_id=1, relevant_docs=rel_docs, chunk_store=store)
    res_2 = selector_2.select_chunks(query_id=1, relevant_docs=rel_docs, chunk_store=store)

    assert len(res_1) == len(res_2)
    for c1, c2 in zip(res_1, res_2, strict=True):
        assert c1["chunk_id"] == c2["chunk_id"]
        assert c1["doc_id"] == c2["doc_id"]
        assert c1["chunk_text"] == c2["chunk_text"]


# ---------------------------------------------------------------------------
# 6. No Duplicate Chunks
# ---------------------------------------------------------------------------


def test_no_duplicate_chunks(tmp_path: Path):
    """Verify that duplicate or near-duplicate chunks are rejected."""
    chunks_dir = tmp_path / "chunks"
    doc_chunk_map = {
        10: [
            "Bệnh nhân được chỉ định uống kháng sinh theo phác đồ điều trị của bác sĩ.",
            "Bệnh nhân được chỉ định uống kháng sinh theo phác đồ điều trị của bác sĩ.",  # Exact dup
            "Bệnh nhân được chỉ định uống kháng sinh theo phác đồ điều trị của bác sĩ chuyên khoa.",  # Near dup
            "Triệu chứng lâm sàng khác bao gồm sốt cao và co giật cơ bắp toàn thân.",  # Distinct
        ],
    }
    store = create_synthetic_chunk_store(chunks_dir, doc_chunk_map)

    cfg = ChunkSelectionConfig(top_k=5, max_chunks_per_document=3, deduplicate=True)
    selector = ChunkSelector(cfg)

    selected = selector.select_chunks(query_id=1, relevant_docs=[10], chunk_store=store)

    # Out of 4 candidates: 1 exact dup and 1 near dup are filtered. Exactly 2 distinct chunks remain.
    assert len(selected) == 2
    assert selected[0]["chunk_text"] != selected[1]["chunk_text"]
    assert selector.last_query_diagnostics is not None
    assert selector.last_query_diagnostics.chunks_skipped_dedup >= 2


# ---------------------------------------------------------------------------
# 7. Correct Chunk-to-Doc Mapping
# ---------------------------------------------------------------------------


def test_correct_chunk_to_doc_mapping(tmp_path: Path):
    """Verify that every selected chunk preserves authentic doc_id and matches relevant_docs."""
    chunks_dir = tmp_path / "chunks"
    doc_chunk_map = {
        777: [
            "Đoạn văn bản y khoa số 1 của doc 777 cung cấp thông tin điều trị chuẩn xác.",
            "Đoạn văn bản y khoa số 2 của doc 777 giải thích cơ chế bệnh học chi tiết.",
        ],
        888: [
            "Đoạn văn bản y khoa số 1 của doc 888 cung cấp thông tin điều trị chuẩn xác.",
            "Đoạn văn bản y khoa số 2 của doc 888 giải thích cơ chế bệnh học chi tiết.",
        ],
    }
    store = create_synthetic_chunk_store(chunks_dir, doc_chunk_map)

    cfg = ChunkSelectionConfig(top_k=4, max_chunks_per_document=2)
    selector = ChunkSelector(cfg)

    selected = selector.select_chunks(query_id=1, relevant_docs=[777, 888], chunk_store=store)

    assert len(selected) == 4
    for c in selected:
        did = c["doc_id"]
        assert did in (777, 888)
        assert f"doc {did}" in c["chunk_text"]


# ---------------------------------------------------------------------------
# 8. Candidate Pool Under Capacity Reports Deficit
# ---------------------------------------------------------------------------


def test_candidate_pool_under_capacity_reports_deficit(tmp_path: Path):
    """Verify that when candidates < requested chunks, system does not hallucinate and reports deficit."""
    q_file = create_synthetic_query_file(tmp_path / "query.parquet", [101])

    chunks_dir = tmp_path / "chunks"
    # Only 5 docs, each with 2 chunks = 10 chunks total
    doc_chunk_map = {
        did: [
            f"Đoạn văn bản y tế số 0 của tài liệu doc {did} với độ dài đầy đủ vượt ngưỡng quy định.",
            f"Đoạn văn bản y tế số 1 của tài liệu doc {did} với độ dài đầy đủ vượt ngưỡng quy định.",
        ]
        for did in range(1, 6)
    }
    create_synthetic_chunk_store(chunks_dir, doc_chunk_map)

    fusion_file = create_synthetic_fusion_file(tmp_path / "fusion.parquet", 101, list(range(1, 6)))

    out_json = tmp_path / "sub_deficit.json"
    exporter = SubmissionExporter()
    summary = exporter.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        output_json_path=out_json,
        top_docs=100,
        top_chunks=100,  # Request 100 chunks
        max_chunks_per_doc=2,
    )

    # Only 10 valid chunks exist
    assert summary.total_relevant_chunks == 10
    assert summary.queries_under_target_chunks == 1
    assert summary.total_chunk_deficit == 90  # 100 - 10 = 90 missing

    with open(out_json, encoding="utf-8") as f:
        records = json.load(f)

    assert len(records[0]["relevant_chunks"]) == 10
    # No fake chunks created
    doc_ids_in_chunks = {c["doc_id"] for c in records[0]["relevant_chunks"]}
    assert doc_ids_in_chunks == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# 9. Integer Document IDs Strictly Enforced in Submission
# ---------------------------------------------------------------------------


def test_integer_doc_ids_in_submission(tmp_path: Path):
    """Verify that query ID, relevant_docs, and chunk doc_ids are strictly integer."""
    q_file = create_synthetic_query_file(tmp_path / "query.parquet", [99])
    chunks_dir = tmp_path / "chunks"
    doc_chunk_map = {12345: ["Nội dung y tế chuẩn xác."]}
    create_synthetic_chunk_store(chunks_dir, doc_chunk_map)
    fusion_file = create_synthetic_fusion_file(tmp_path / "fusion.parquet", 99, [12345])

    out_json = tmp_path / "submission_int.json"
    exporter = SubmissionExporter()
    exporter.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        output_json_path=out_json,
        top_docs=10,
        top_chunks=10,
        max_chunks_per_doc=2,
    )

    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)

    item = data[0]
    assert isinstance(item["id"], int) and not isinstance(item["id"], bool)
    assert isinstance(item["relevant_docs"][0], int) and not isinstance(item["relevant_docs"][0], bool)
    assert isinstance(item["relevant_chunks"][0]["doc_id"], int) and not isinstance(item["relevant_chunks"][0]["doc_id"], bool)

    validator = SubmissionValidator()
    res = validator.validate_json(data, max_chunks_per_doc=2)
    assert res.is_valid is True
    assert "Strictly integer" in res.summary["doc_id_format"]


# ---------------------------------------------------------------------------
# 10. Experiment Isolation: No Overwriting of Baseline
# ---------------------------------------------------------------------------


def test_experiment_isolation_no_baseline_overwrite(tmp_path: Path):
    """Verify that Experiment A1.2 writes to its own isolated directory without touching baseline."""
    base_dir = tmp_path / "baseline_a1" / "submission"
    base_dir.mkdir(parents=True, exist_ok=True)
    sentinel_file = base_dir / "submission.json"
    sentinel_content = json.dumps([{"id": 1, "relevant_docs": [100], "relevant_chunks": []}])
    sentinel_file.write_text(sentinel_content, encoding="utf-8")

    q_file = create_synthetic_query_file(tmp_path / "query.parquet", [1])
    chunks_dir = tmp_path / "chunks"
    create_synthetic_chunk_store(chunks_dir, {200: ["Chunk 200"]})
    fusion_file = create_synthetic_fusion_file(tmp_path / "fusion.parquet", 1, [200])

    # Run A1.2 to independent target
    cand_out = tmp_path / "a1_2_max2_chunks" / "submission" / "submission.json"
    exporter = SubmissionExporter()
    exporter.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        output_json_path=cand_out,
        top_docs=10,
        top_chunks=10,
        max_chunks_per_doc=2,
    )

    # Baseline file must be completely untouched
    assert sentinel_file.exists()
    assert sentinel_file.read_text(encoding="utf-8") == sentinel_content
    # Candidate file must be written to its own path
    assert cand_out.exists()
    assert cand_out.read_text(encoding="utf-8") != sentinel_content


# ---------------------------------------------------------------------------
# 11. Diagnostic Coverage Shift & Comparison
# ---------------------------------------------------------------------------


def test_diagnostics_coverage_shift_and_comparison(tmp_path: Path):
    """Verify analyze_submission and compare_submissions accurately compute coverage gain."""
    # Build synthetic Baseline (3 chunks per doc for docs 1..3, total 9 chunks, 3 unique docs)
    base_data = [
        {
            "id": 1,
            "relevant_docs": list(range(1, 11)),
            "relevant_chunks": [
                {"doc_id": 1, "chunk_text": "Chunk 1a"},
                {"doc_id": 1, "chunk_text": "Chunk 1b"},
                {"doc_id": 1, "chunk_text": "Chunk 1c"},
                {"doc_id": 2, "chunk_text": "Chunk 2a"},
                {"doc_id": 2, "chunk_text": "Chunk 2b"},
                {"doc_id": 2, "chunk_text": "Chunk 2c"},
                {"doc_id": 3, "chunk_text": "Chunk 3a"},
                {"doc_id": 3, "chunk_text": "Chunk 3b"},
                {"doc_id": 3, "chunk_text": "Chunk 3c"},
            ],
        }
    ]
    base_file = tmp_path / "base.json"
    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(base_data, f)

    # Build synthetic A1.2 (2 chunks per doc for docs 1..4, plus 1 chunk for doc 5 = 9 chunks, 5 unique docs)
    cand_data = [
        {
            "id": 1,
            "relevant_docs": list(range(1, 11)),
            "relevant_chunks": [
                {"doc_id": 1, "chunk_text": "Chunk 1a"},
                {"doc_id": 1, "chunk_text": "Chunk 1b"},
                {"doc_id": 2, "chunk_text": "Chunk 2a"},
                {"doc_id": 2, "chunk_text": "Chunk 2b"},
                {"doc_id": 3, "chunk_text": "Chunk 3a"},
                {"doc_id": 3, "chunk_text": "Chunk 3b"},
                {"doc_id": 4, "chunk_text": "Chunk 4a"},
                {"doc_id": 4, "chunk_text": "Chunk 4b"},
                {"doc_id": 5, "chunk_text": "Chunk 5a"},
            ],
        }
    ]
    cand_file = tmp_path / "cand.json"
    with open(cand_file, "w", encoding="utf-8") as f:
        json.dump(cand_data, f)

    rep = compare_submissions(
        baseline_path=base_file,
        candidate_path=cand_file,
        target_docs=10,
        target_chunks=9,
    )

    assert rep.ranking_identical is True
    assert rep.baseline_mean_unique_docs == 3.0
    assert rep.candidate_mean_unique_docs == 5.0
    assert rep.mean_unique_docs_delta == 2.0  # +2 unique documents gained!
    assert rep.mean_unique_docs_pct_gain > 60.0  # +66.7% gain
    assert rep.candidate_diagnostics.docs_with_gt_2_chunks == 0  # No doc has > 2 chunks in candidate!
    assert rep.baseline_diagnostics.docs_with_3_chunks == 3  # 3 docs had 3 chunks in baseline


# ---------------------------------------------------------------------------
# 12. Validator Catches Cap Violation
# ---------------------------------------------------------------------------


def test_validator_detects_cap_violation():
    """Verify that SubmissionValidator detects and reports cap violations."""
    validator = SubmissionValidator(max_chunks_per_doc=2)

    invalid_data = [
        {
            "id": 1,
            "relevant_docs": [10],
            "relevant_chunks": [
                {"doc_id": 10, "chunk_text": "Chunk 1"},
                {"doc_id": 10, "chunk_text": "Chunk 2"},
                {"doc_id": 10, "chunk_text": "Chunk 3"},  # 3rd chunk violates cap 2
            ],
        }
    ]

    res = validator.validate_json(invalid_data, max_chunks_per_doc=2)
    assert res.is_valid is False
    assert any("exceeding max limit of 2 chunks/doc" in err for err in res.errors)


# ---------------------------------------------------------------------------
# 13. pubmed: Prefix Stripped and Forbidden
# ---------------------------------------------------------------------------


def test_pubmed_prefix_stripped_and_forbidden():
    """Verify that pubmed: prefix is cleanly stripped and flagged as forbidden if present."""
    raw_text = "pubmed:123456 Triệu chứng hen suyễn ở trẻ em."
    clean = strip_synthetic_prefixes(raw_text)
    assert not clean.lower().startswith("pubmed:")
    assert clean == "123456 Triệu chứng hen suyễn ở trẻ em."

    # If unstripped prefix remains in submission, validator must reject it
    validator = SubmissionValidator()
    res = validator.validate_json(
        [
            {
                "id": 1,
                "relevant_docs": [10],
                "relevant_chunks": [{"doc_id": 10, "chunk_text": "pubmed:999 Dữ liệu y khoa"}],
            }
        ]
    )
    assert res.is_valid is False
    assert any("leaked prefix 'pubmed:'" in err for err in res.errors)
