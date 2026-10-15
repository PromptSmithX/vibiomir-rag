"""Unit tests for ViBioMIR Experiments A1.3 & A1.4 — Source Passage Expansion.

Covers all 15 required verification areas:
1. Expansion up to 768 tokens (Experiment A1.3).
2. Expansion up to 1024 tokens (Experiment A1.4).
3. Chunk at beginning of document (budget reallocated to succeeding context).
4. Chunk at end of document (budget reallocated to preceding context).
5. Document shorter than budget uses entire valid source document.
6. Missing source offsets: verified exact matching without requiring offsets.
7. Ambiguous exact matches: detects multiple occurrences, refuses to guess, falls back safely.
8. Single continuous passage from authentic source document (no disjoint joins).
9. Strict enforcement: token length never exceeds configured max_tokens budget.
10. Authentic source text: no rewriting, translation, or prompt instruction leakage.
11. Preserves selected chunk IDs and chunk allocation identically to baseline.
12. Preserves top 100 relevant_docs ranking 100% identically to baseline.
13. Strict integer document IDs enforced across all schema fields.
14. Submission JSON and ZIP package validity (single root JSON).
15. Experiment isolation: candidate outputs do not overwrite baseline A1.2 artifacts.
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
    PassageExpansionConfig,
    SubmissionConfig,
)
from retrieval.dense.tokenizer_helper import FallbackTokenCounter
from retrieval.submission.chunk_selector import ChunkSelector
from retrieval.submission.chunk_store import ChunkStore
from retrieval.submission.exporter import SubmissionExporter
from retrieval.submission.passage_diagnostics import (
    analyze_expanded_submission,
    format_passage_diagnostics_report,
)
from retrieval.submission.passage_expander import (
    PassageExpansionResult,
    SourcePassageExpander,
)
from retrieval.submission.validator import SubmissionValidator


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def build_synthetic_corpus_doc(num_sentences: int, did: int = 100) -> tuple[str, str]:
    """Build a synthetic multi-sentence document and return (doc_text, middle_chunk_text)."""
    sentences = [
        f"Câu số {i} trong tài liệu y khoa {did} mô tả chi tiết về triệu chứng và phương pháp điều trị lâm sàng."
        for i in range(num_sentences)
    ]
    doc_text = " ".join(sentences)
    # Pick sentence in the middle as the retrieved chunk
    mid_idx = num_sentences // 2
    middle_chunk = sentences[mid_idx]
    return doc_text, middle_chunk


# ---------------------------------------------------------------------------
# 1. Expansion to 768 Tokens (Experiment A1.3)
# ---------------------------------------------------------------------------


def test_expansion_768_tokens(tmp_path: Path):
    """Verify expansion to 768 tokens on a long document with center anchor."""
    full_text, chunk_text = build_synthetic_corpus_doc(num_sentences=40, did=768)
    store = ChunkStore()
    store.set_corpus_text(768, full_text)

    tc = FallbackTokenCounter()
    cfg = PassageExpansionConfig(enabled=True, max_tokens=768)
    expander = SourcePassageExpander(config=cfg, token_counter=tc)

    res = expander.expand_chunk(doc_id=768, chunk_id="768_c0_a1", chunk_text=chunk_text, chunk_store=store)

    assert res.is_expanded is True
    assert res.expanded_tokens <= 768
    assert res.expanded_tokens > res.original_tokens
    # Must contain original chunk text inside
    assert chunk_text in res.expanded_text
    # Must be a continuous slice of full_text
    assert res.expanded_text in full_text


# ---------------------------------------------------------------------------
# 2. Expansion to 1024 Tokens (Experiment A1.4)
# ---------------------------------------------------------------------------


def test_expansion_1024_tokens(tmp_path: Path):
    """Verify expansion to 1024 tokens expands further than 768 on the same document."""
    full_text, chunk_text = build_synthetic_corpus_doc(num_sentences=60, did=1024)
    store = ChunkStore()
    store.set_corpus_text(1024, full_text)

    tc = FallbackTokenCounter()

    # Expand to 768
    exp_768 = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=768), token_counter=tc)
    res_768 = exp_768.expand_chunk(doc_id=1024, chunk_id="1024_c0_a1", chunk_text=chunk_text, chunk_store=store)

    # Expand to 1024
    exp_1024 = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=1024), token_counter=tc)
    res_1024 = exp_1024.expand_chunk(doc_id=1024, chunk_id="1024_c0_a1", chunk_text=chunk_text, chunk_store=store)

    assert res_1024.is_expanded is True
    assert res_1024.expanded_tokens <= 1024
    assert res_1024.expanded_tokens >= res_768.expanded_tokens
    assert chunk_text in res_1024.expanded_text
    assert res_1024.expanded_text in full_text


# ---------------------------------------------------------------------------
# 3. Chunk at Document Start Reallocates Budget
# ---------------------------------------------------------------------------


def test_chunk_at_document_start_reallocates_budget(tmp_path: Path):
    """When chunk is at index 0, unused left budget is transferred to succeeding text."""
    sentences = [
        f"Câu số {i} về bệnh học truyền nhiễm mô tả các biến chứng nguy hiểm đối với hệ tim mạch."
        for i in range(30)
    ]
    full_text = " ".join(sentences)
    first_chunk = sentences[0]

    store = ChunkStore()
    store.set_corpus_text(501, full_text)

    expander = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=500), token_counter=FallbackTokenCounter())
    res = expander.expand_chunk(doc_id=501, chunk_id="501_c0_a1", chunk_text=first_chunk, chunk_store=store)

    assert res.is_expanded is True
    assert res.expanded_tokens <= 500
    assert res.start_offset == 0  # Starts exactly at beginning of document
    assert sentences[1] in res.expanded_text  # Expanded significantly into succeeding sentences
    assert res.expanded_text in full_text


# ---------------------------------------------------------------------------
# 4. Chunk at Document End Reallocates Budget
# ---------------------------------------------------------------------------


def test_chunk_at_document_end_reallocates_budget(tmp_path: Path):
    """When chunk is at the end, unused right budget is transferred to preceding text."""
    sentences = [
        f"Câu số {i} về dược lý lâm sàng hướng dẫn phác đồ điều trị suy gan và suy thận mạn tính."
        for i in range(30)
    ]
    full_text = " ".join(sentences)
    last_chunk = sentences[-1]

    store = ChunkStore()
    store.set_corpus_text(502, full_text)

    expander = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=500), token_counter=FallbackTokenCounter())
    res = expander.expand_chunk(doc_id=502, chunk_id="502_c0_a1", chunk_text=last_chunk, chunk_store=store)

    assert res.is_expanded is True
    assert res.expanded_tokens <= 500
    assert res.end_offset == len(full_text)  # Ends at end of document
    assert sentences[-2] in res.expanded_text  # Expanded significantly into preceding sentences
    assert res.expanded_text in full_text


# ---------------------------------------------------------------------------
# 5. Document Shorter Than Limit Uses Entire Document
# ---------------------------------------------------------------------------


def test_document_shorter_than_limit_uses_entire_document(tmp_path: Path):
    """When full document is shorter than max_tokens, entire document text is returned."""
    short_doc = "Bệnh cúm A lây qua giọt bắn. Bệnh nhân cần nghỉ ngơi và uống thuốc hạ sốt."
    chunk = "Bệnh cúm A lây qua giọt bắn."

    store = ChunkStore()
    store.set_corpus_text(301, short_doc)

    expander = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=768), token_counter=FallbackTokenCounter())
    res = expander.expand_chunk(doc_id=301, chunk_id="301_c0_a1", chunk_text=chunk, chunk_store=store)

    assert res.is_expanded is True
    assert res.expanded_text == short_doc
    assert res.strategy_used == "full_document"


# ---------------------------------------------------------------------------
# 6. Missing Source Offsets: Verified Exact Matching
# ---------------------------------------------------------------------------


def test_missing_source_offsets_uses_exact_matching(tmp_path: Path):
    """When chunk lacks start_char/end_char offsets, exact matching locates and expands it."""
    full_text, chunk_text = build_synthetic_corpus_doc(num_sentences=25, did=601)
    store = ChunkStore()
    store.set_corpus_text(601, full_text)

    # Pass empty chunk_meta (no offsets)
    expander = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=768), token_counter=FallbackTokenCounter())
    res = expander.expand_chunk(
        doc_id=601,
        chunk_id="601_c0_a1",
        chunk_text=chunk_text,
        chunk_meta={},
        chunk_store=store,
    )

    assert res.is_expanded is True
    assert chunk_text in res.expanded_text
    assert res.start_offset is not None
    assert res.end_offset is not None


# ---------------------------------------------------------------------------
# 7. Ambiguous Exact Matches: Refuse to Guess and Fall Back
# ---------------------------------------------------------------------------


def test_ambiguous_exact_matches_falls_back_safely(tmp_path: Path):
    """When chunk text appears identically at multiple locations, system falls back safely."""
    repeated_sentence = "Triệu chứng sốt cao xuất hiện liên tục trong ngày đầu tiên của bệnh nhân."
    full_text = f"{repeated_sentence} Đoạn văn bản giữa. {repeated_sentence} Đoạn văn bản kết thúc."

    store = ChunkStore()
    store.set_corpus_text(701, full_text)

    expander = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=768), token_counter=FallbackTokenCounter())
    res = expander.expand_chunk(
        doc_id=701,
        chunk_id="701_c0_a1",
        chunk_text=repeated_sentence,
        chunk_store=store,
    )

    # Refuses to guess: returns authentic original chunk text
    assert res.is_expanded is False
    assert res.expanded_text == repeated_sentence
    assert res.fallback_reason == "ambiguous_multiple_matches"


# ---------------------------------------------------------------------------
# 8. Single Continuous Passage from Authentic Source Document
# ---------------------------------------------------------------------------


def test_continuous_passage_from_same_source_document(tmp_path: Path):
    """Verify that expanded text is a single continuous slice without disjoint joins."""
    full_text, chunk_text = build_synthetic_corpus_doc(num_sentences=35, did=801)
    store = ChunkStore()
    store.set_corpus_text(801, full_text)

    expander = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=768), token_counter=FallbackTokenCounter())
    res = expander.expand_chunk(doc_id=801, chunk_id="801_c0_a1", chunk_text=chunk_text, chunk_store=store)

    assert res.is_expanded is True
    # Mathematical proof of continuity: res.expanded_text must be an exact substring of full_text
    assert res.expanded_text in full_text
    assert full_text[res.start_offset : res.end_offset] == res.expanded_text


# ---------------------------------------------------------------------------
# 9. Strict Enforcement: Does Not Exceed Token Budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_budget", [256, 512, 768, 1024])
def test_does_not_exceed_token_budget(max_budget: int):
    """Verify that expansion never violates token budget for any limit."""
    full_text, chunk_text = build_synthetic_corpus_doc(num_sentences=50, did=901)
    store = ChunkStore()
    store.set_corpus_text(901, full_text)

    tc = FallbackTokenCounter()
    expander = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=max_budget), token_counter=tc)
    res = expander.expand_chunk(doc_id=901, chunk_id="901_c0_a1", chunk_text=chunk_text, chunk_store=store)

    tok_len = tc.count_tokens(res.expanded_text)
    assert tok_len <= max_budget


# ---------------------------------------------------------------------------
# 10. Authentic Source Text: No Mutation or Prompt Instruction Leakage
# ---------------------------------------------------------------------------


def test_does_not_mutate_source_text():
    """Verify that no prompt instructions or pubmed: prefixes leak into expanded text."""
    raw_chunk = "pubmed:12345 Instruct: \nQuery: Triệu chứng viêm phổi do vi khuẩn phế cầu."
    full_text = "Triệu chứng viêm phổi do vi khuẩn phế cầu. Bệnh nhân có thể sốt rét run và khó thở dữ dội."

    store = ChunkStore()
    store.set_corpus_text(999, full_text)

    expander = SourcePassageExpander(config=PassageExpansionConfig(max_tokens=768), token_counter=FallbackTokenCounter())
    res = expander.expand_chunk(doc_id=999, chunk_id="999_c0_a1", chunk_text=raw_chunk, chunk_store=store)

    assert "pubmed:" not in res.expanded_text
    assert "Instruct:" not in res.expanded_text
    assert "Query:" not in res.expanded_text
    assert res.expanded_text in full_text


# ---------------------------------------------------------------------------
# 11. Preserves Selected Chunk IDs and Selection Order
# ---------------------------------------------------------------------------


def test_preserves_selected_chunk_ids_and_order(tmp_path: Path):
    """Verify that enabling passage expansion preserves identical chunk selection and ordering."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    doc_chunk_map = {
        did: [
            f"Đoạn văn bản y tế số 0 của tài liệu {did} với nội dung chuyên sâu về dược lực học.",
            f"Đoạn văn bản y tế số 1 của tài liệu {did} với nội dung chuyên sâu về dược động học.",
        ]
        for did in range(101, 111)
    }

    all_cids, all_dids, all_idxs, all_texts = [], [], [], []
    for did, texts in doc_chunk_map.items():
        for idx, txt in enumerate(texts):
            cid = f"{did}_c{idx}_a1"
            all_cids.append(cid)
            all_dids.append(did)
            all_idxs.append(idx)
            all_texts.append(txt)

    pq.write_table(
        pa.Table.from_pydict({"chunk_id": all_cids, "doc_id": all_dids, "chunk_index": all_idxs, "text": all_texts}),
        chunks_dir / "chunks_part_00000.parquet",
    )
    store = ChunkStore(chunks_dir=chunks_dir)
    store.preload_for_retrieval(set(all_cids), set(all_dids))

    rel_docs = list(range(101, 111))

    # Run without expansion
    cfg_unexpanded = ChunkSelectionConfig(top_k=20, max_chunks_per_document=2, passage_expansion=PassageExpansionConfig(enabled=False))
    sel_unexp = ChunkSelector(cfg_unexpanded).select_chunks(query_id=1, relevant_docs=rel_docs, chunk_store=store)

    # Run with expansion
    cfg_expanded = ChunkSelectionConfig(top_k=20, max_chunks_per_document=2, passage_expansion=PassageExpansionConfig(enabled=True, max_tokens=768))
    sel_exp = ChunkSelector(cfg_expanded).select_chunks(query_id=1, relevant_docs=rel_docs, chunk_store=store)

    assert len(sel_unexp) == len(sel_exp)
    for u, e in zip(sel_unexp, sel_exp, strict=True):
        assert u["chunk_id"] == e["chunk_id"]
        assert u["doc_id"] == e["doc_id"]


# ---------------------------------------------------------------------------
# 12. Preserves Relevant Docs Ranking 100% Identically to Baseline
# ---------------------------------------------------------------------------


def test_preserves_relevant_docs_ranking_identically(tmp_path: Path):
    """Verify that relevant_docs is 100% identical between baseline and expanded submissions."""
    q_file = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pydict({"id": [1, 2], "query": ["huyết áp", "đái tháo đường"]}), q_file)

    doc_ids_q1 = list(range(1001, 1101))
    doc_ids_q2 = list(range(2001, 2101))
    fusion_file = tmp_path / "fusion.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "query_id": [1] * 100 + [2] * 100,
                "doc_id": doc_ids_q1 + doc_ids_q2,
                "final_rank": list(range(1, 101)) * 2,
                "fused_score": [1.0 / (60 + r) for r in range(1, 101)] * 2,
                "dense_best_chunk_id": [f"{d}_c0_a1" for d in (doc_ids_q1 + doc_ids_q2)],
            }
        ),
        fusion_file,
    )

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pq.write_table(
        pa.Table.from_pydict(
            {
                "chunk_id": [f"{d}_c0_a1" for d in (doc_ids_q1 + doc_ids_q2)],
                "doc_id": doc_ids_q1 + doc_ids_q2,
                "chunk_index": [0] * 200,
                "text": [f"Nội dung y khoa của tài liệu số {d}." for d in (doc_ids_q1 + doc_ids_q2)],
            }
        ),
        chunks_dir / "chunks_part_00000.parquet",
    )

    corpus_file = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "doc_id": doc_ids_q1 + doc_ids_q2,
                "text": [
                    f"Nội dung y khoa của tài liệu số {d}. Bệnh nhân cần tuân thủ phác đồ điều trị nghiêm ngặt."
                    for d in (doc_ids_q1 + doc_ids_q2)
                ],
            }
        ),
        corpus_file,
    )

    # 1. Baseline A1.2 export
    base_json = tmp_path / "a1_2" / "submission.json"
    exp_base = SubmissionExporter()
    exp_base.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        corpus_path=corpus_file,
        output_json_path=base_json,
        top_docs=100,
        top_chunks=100,
        max_chunks_per_doc=2,
        expand_passages=False,
    )

    # 2. A1.3 (768) export
    cand_json = tmp_path / "a1_3" / "submission.json"
    exp_cand = SubmissionExporter()
    exp_cand.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        corpus_path=corpus_file,
        output_json_path=cand_json,
        top_docs=100,
        top_chunks=100,
        max_chunks_per_doc=2,
        expand_passages=True,
        expansion_max_tokens=768,
    )

    with open(base_json, encoding="utf-8") as f:
        b_data = json.load(f)
    with open(cand_json, encoding="utf-8") as f:
        c_data = json.load(f)

    for b, c in zip(b_data, c_data, strict=True):
        assert b["id"] == c["id"]
        assert b["relevant_docs"] == c["relevant_docs"]  # 100% IDENTICAL ORDER AND CONTENT


# ---------------------------------------------------------------------------
# 13. Integer Document IDs Strictly Enforced
# ---------------------------------------------------------------------------


def test_integer_document_ids_strictly_enforced(tmp_path: Path):
    """Verify that submission schema contains strictly integer IDs for queries, docs, and chunks."""
    valid_sub = [
        {
            "id": 1,
            "relevant_docs": [101, 102],
            "relevant_chunks": [
                {"doc_id": 101, "chunk_text": "Đoạn trích văn bản mở rộng."},
            ],
        }
    ]
    sub_file = tmp_path / "sub_int.json"
    with open(sub_file, "w", encoding="utf-8") as f:
        json.dump(valid_sub, f)

    validator = SubmissionValidator(max_chunks_per_doc=2)
    res = validator.validate_file(sub_file, max_chunks_per_doc=2)
    assert res.is_valid is True
    assert "Strictly integer" in res.summary["doc_id_format"]


# ---------------------------------------------------------------------------
# 14. Submission ZIP and JSON Package Validity
# ---------------------------------------------------------------------------


def test_submission_zip_and_json_validity(tmp_path: Path):
    """Verify that export produces a compliant submission.zip containing strictly one JSON at root."""
    q_file = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pydict({"id": [10], "query": ["sốt rét"]}), q_file)
    fusion_file = tmp_path / "fusion.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "query_id": [10],
                "doc_id": [555],
                "final_rank": [1],
                "fused_score": [0.05],
                "dense_best_chunk_id": ["555_c0_a1"],
            }
        ),
        fusion_file,
    )
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pq.write_table(
        pa.Table.from_pydict(
            {
                "chunk_id": ["555_c0_a1"],
                "doc_id": [555],
                "chunk_index": [0],
                "text": ["Bệnh sốt rét do ký sinh trùng Plasmodium gây ra."],
            }
        ),
        chunks_dir / "chunks_part_00000.parquet",
    )

    corpus_file = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "doc_id": [555],
                "text": ["Bệnh sốt rét do ký sinh trùng Plasmodium gây ra. Các triệu chứng bao gồm sốt cao rét run."],
            }
        ),
        corpus_file,
    )

    out_json = tmp_path / "a1_3" / "submission.json"
    out_zip = tmp_path / "a1_3" / "submission.zip"

    exporter = SubmissionExporter()
    exporter.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        corpus_path=corpus_file,
        output_json_path=out_json,
        zip_output_path=out_zip,
        top_docs=10,
        top_chunks=10,
        max_chunks_per_doc=2,
        expand_passages=True,
        expansion_max_tokens=768,
    )

    validator = SubmissionValidator(max_chunks_per_doc=2)
    res = validator.validate_file(out_zip, max_chunks_per_doc=2)
    assert res.is_valid is True
    assert res.summary["zip_archive"] == "submission.zip"
    assert res.summary["inner_file"] == "submission.json"


# ---------------------------------------------------------------------------
# 15. Experiment Isolation: Does Not Overwrite Baseline Artifacts
# ---------------------------------------------------------------------------


def test_experiment_isolation_does_not_overwrite_baseline(tmp_path: Path):
    """Verify that A1.3 and A1.4 write to isolated directories and never mutate baseline."""
    base_dir = tmp_path / "a1_2_max2_chunks" / "submission"
    base_dir.mkdir(parents=True)
    sentinel_json = base_dir / "submission.json"
    sentinel_text = json.dumps([{"id": 99, "relevant_docs": [999], "relevant_chunks": []}])
    sentinel_json.write_text(sentinel_text, encoding="utf-8")

    q_file = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pydict({"id": [1], "query": ["ho khan"]}), q_file)
    fusion_file = tmp_path / "fusion.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "query_id": [1],
                "doc_id": [111],
                "final_rank": [1],
                "fused_score": [0.05],
                "dense_best_chunk_id": ["111_c0_a1"],
            }
        ),
        fusion_file,
    )
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pq.write_table(
        pa.Table.from_pydict(
            {
                "chunk_id": ["111_c0_a1"],
                "doc_id": [111],
                "chunk_index": [0],
                "text": ["Triệu chứng ho khan kéo dài."],
            }
        ),
        chunks_dir / "chunks_part_00000.parquet",
    )

    corpus_file = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "doc_id": [111],
                "text": ["Triệu chứng ho khan kéo dài. Cần chẩn đoán phân biệt với các bệnh lý đường hô hấp."],
            }
        ),
        corpus_file,
    )

    out_a1_3 = tmp_path / "a1_3_passage_768" / "submission" / "submission.json"
    exporter = SubmissionExporter()
    exporter.export(
        retrieval_file=fusion_file,
        query_path=q_file,
        chunks_dir=chunks_dir,
        corpus_path=corpus_file,
        output_json_path=out_a1_3,
        top_docs=10,
        top_chunks=10,
        max_chunks_per_doc=2,
        expand_passages=True,
        expansion_max_tokens=768,
    )

    # Baseline file must be completely untouched
    assert sentinel_json.exists()
    assert sentinel_json.read_text(encoding="utf-8") == sentinel_text
    # Candidate written to its own destination
    assert out_a1_3.exists()
    assert out_a1_3.read_text(encoding="utf-8") != sentinel_text
