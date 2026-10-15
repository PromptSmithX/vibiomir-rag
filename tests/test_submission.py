"""Unit tests for ViBioMIR Official Submission Exporter and Validator.

Covers:
- Valid submission schema with strictly integer document IDs
- Rejection of string document IDs (BTC official requirement)
- Detection of duplicate document IDs within a query
- Missing and duplicate queries against official dataset
- Invalid field types
- Empty result arrays for zero-hit queries
- Forbidden metadata fields (score, rank, etc.)
- Source text vs embedding text leakage
- ZIP structure and root-only enforcement
- Document ID verification against corpus
- End-to-end synthetic export and validation
"""

import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retrieval.dense.config import SubmissionConfig
from retrieval.submission.chunk_selector import ChunkSelectionConfig, ChunkSelector
from retrieval.submission.chunk_store import ChunkStore
from retrieval.submission.exporter import SubmissionExporter
from retrieval.submission.validator import SubmissionValidator, ValidationResult


# ---------------------------------------------------------------------------
# 1. Valid Submission with Integer Document IDs
# ---------------------------------------------------------------------------


def test_valid_submission_passes():
    """Verify that a compliant submission structure with integer doc_ids passes all checks."""
    valid_data = [
        {
            "id": 1,
            "relevant_docs": [1001, 1002],
            "relevant_chunks": [
                {
                    "doc_id": 1001,
                    "chunk_text": "Thuốc kháng sinh được chỉ định điều trị nhiễm khuẩn.",
                    "chunk_order": 0,
                },
                {
                    "doc_id": 1002,
                    "chunk_text": "Triệu chứng thường gặp bao gồm sốt cao và ho khan.",
                    "chunk_order": 1,
                },
            ],
        },
        {
            "id": 2,
            "relevant_docs": [2001],
            "relevant_chunks": [
                {
                    "doc_id": 2001,
                    "chunk_text": "Bệnh nhân cần uống nhiều nước và nghỉ ngơi hợp lý.",
                }
            ],
        },
    ]

    validator = SubmissionValidator()
    result = validator.validate_json(valid_data)

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.summary["total_queries"] == 2
    assert result.summary["total_relevant_docs"] == 3
    assert result.summary["total_relevant_chunks"] == 3
    assert "Strictly integer" in result.summary["doc_id_format"]


def test_string_doc_id_fails_validation():
    """Verify that string document IDs are strictly rejected as invalid (BTC requirement)."""
    validator = SubmissionValidator()

    # Case A: String in relevant_docs
    string_docs_data = [
        {
            "id": 1,
            "relevant_docs": ["1001", "1002"],  # INVALID: must be int
            "relevant_chunks": [],
        }
    ]
    res_a = validator.validate_json(string_docs_data)
    assert res_a.is_valid is False
    assert any("must be integer" in err for err in res_a.errors)

    # Case B: String doc_id in relevant_chunks
    string_chunk_data = [
        {
            "id": 1,
            "relevant_docs": [1001],
            "relevant_chunks": [
                {
                    "doc_id": "1001",  # INVALID: must be int
                    "chunk_text": "Nội dung hợp lệ.",
                }
            ],
        }
    ]
    res_b = validator.validate_json(string_chunk_data)
    assert res_b.is_valid is False
    assert any("'doc_id' must be integer" in err for err in res_b.errors)


def test_duplicate_doc_ids_in_query_fails():
    """Verify that duplicate doc_ids within relevant_docs trigger an error."""
    validator = SubmissionValidator()
    dup_doc_data = [
        {
            "id": 1,
            "relevant_docs": [1001, 1002, 1001],  # Duplicate 1001
            "relevant_chunks": [],
        }
    ]
    res = validator.validate_json(dup_doc_data)
    assert res.is_valid is False
    assert any("duplicate doc_id" in err.lower() for err in res.errors)


# ---------------------------------------------------------------------------
# 2. Missing & Duplicate Queries
# ---------------------------------------------------------------------------


def test_missing_query_fails(tmp_path: Path):
    """Verify that missing an official query ID fails validation."""
    # Create mock official query.parquet with queries 1, 2, 3
    q_file = tmp_path / "query.parquet"
    tbl = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "query": ["query 1", "query 2", "query 3"],
        }
    )
    pq.write_table(tbl, q_file)

    # Submission only contains queries 1 and 2 (missing 3)
    incomplete_data = [
        {"id": 1, "relevant_docs": [10], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [20], "relevant_chunks": []},
    ]

    validator = SubmissionValidator(query_path=q_file)
    result = validator.validate_json(incomplete_data)

    assert result.is_valid is False
    assert any("Missing 1 official query ID" in err for err in result.errors)


def test_duplicate_query_fails():
    """Verify that duplicate query IDs in submission trigger a validation error."""
    dup_data = [
        {"id": 10, "relevant_docs": [1], "relevant_chunks": []},
        {"id": 20, "relevant_docs": [2], "relevant_chunks": []},
        {"id": 10, "relevant_docs": [3], "relevant_chunks": []},  # Duplicate ID 10
    ]

    validator = SubmissionValidator()
    result = validator.validate_json(dup_data)

    assert result.is_valid is False
    assert any("duplicate query id" in err.lower() for err in result.errors)


# ---------------------------------------------------------------------------
# 3. Invalid Field Types
# ---------------------------------------------------------------------------


def test_invalid_field_types():
    """Verify that incorrect data types for id, relevant_docs, or relevant_chunks fail."""
    validator = SubmissionValidator()

    # Case A: id is string instead of int
    res_a = validator.validate_json(
        [{"id": "1", "relevant_docs": [10], "relevant_chunks": []}]
    )
    assert res_a.is_valid is False
    assert any("invalid 'id' type" in err for err in res_a.errors)

    # Case B: relevant_docs contains boolean
    res_b = validator.validate_json(
        [{"id": 1, "relevant_docs": [True], "relevant_chunks": []}]
    )
    assert res_b.is_valid is False
    assert any("must be integer" in err for err in res_b.errors)

    # Case C: chunk doc_id is boolean
    res_c = validator.validate_json(
        [
            {
                "id": 1,
                "relevant_docs": [10],
                "relevant_chunks": [{"doc_id": True, "chunk_text": "valid text"}],
            }
        ]
    )
    assert res_c.is_valid is False
    assert any("'doc_id' must be integer" in err for err in res_c.errors)

    # Case D: chunk_order is negative
    res_d = validator.validate_json(
        [
            {
                "id": 1,
                "relevant_docs": [10],
                "relevant_chunks": [
                    {"doc_id": 10, "chunk_text": "valid text", "chunk_order": -1}
                ],
            }
        ]
    )
    assert res_d.is_valid is False
    assert any("chunk_order must be non-negative int" in err for err in res_d.errors)


# ---------------------------------------------------------------------------
# 4. Empty Result Arrays Handling
# ---------------------------------------------------------------------------


def test_empty_result_arrays_valid():
    """Verify that a query with zero hits correctly formats as empty arrays."""
    data = [
        {"id": 101, "relevant_docs": [], "relevant_chunks": []},
        {
            "id": 102,
            "relevant_docs": [55],
            "relevant_chunks": [{"doc_id": 55, "chunk_text": "Sample content."}],
        },
    ]

    validator = SubmissionValidator()
    result = validator.validate_json(data)

    assert result.is_valid is True
    assert result.summary["empty_queries_count"] == 1


# ---------------------------------------------------------------------------
# 5. Forbidden Metadata Fields
# ---------------------------------------------------------------------------


def test_forbidden_metadata_fields():
    """Verify that scores, ranks, or experiment metadata are rejected from official JSON."""
    data_with_scores = [
        {
            "id": 1,
            "score": 0.985,  # Forbidden
            "rank": 1,  # Forbidden
            "relevant_docs": [100],
            "relevant_chunks": [
                {
                    "doc_id": 100,
                    "chunk_text": "Valid text.",
                    "score": 0.95,  # Forbidden
                }
            ],
        }
    ]

    validator = SubmissionValidator()
    result = validator.validate_json(data_with_scores)

    assert result.is_valid is False
    assert any("forbidden metadata fields" in err for err in result.errors)
    assert any("forbidden fields" in err for err in result.errors)


# ---------------------------------------------------------------------------
# 6. Source Text vs Embedding Text Prefix Leakage
# ---------------------------------------------------------------------------


def test_source_text_vs_embedding_text():
    """Verify that leaked embedding prefixes like [Tiêu đề: ...] or Instruct: fail."""
    validator = SubmissionValidator()

    # Leaked title prefix
    leaked_title_data = [
        {
            "id": 1,
            "relevant_docs": [10],
            "relevant_chunks": [
                {
                    "doc_id": 10,
                    "chunk_text": "[Tiêu đề: Huyết áp cao] Đây là nội dung bài viết.",
                }
            ],
        }
    ]
    res_title = validator.validate_json(leaked_title_data)
    assert res_title.is_valid is False
    assert any("leaked prefix '[Tiêu đề:'" in err for err in res_title.errors)

    # Leaked model instruction prefix
    leaked_prompt_data = [
        {
            "id": 2,
            "relevant_docs": [20],
            "relevant_chunks": [
                {
                    "doc_id": 20,
                    "chunk_text": "Instruct: Given a web search query\nQuery: Triệu chứng cảm cúm",
                }
            ],
        }
    ]
    res_prompt = validator.validate_json(leaked_prompt_data)
    assert res_prompt.is_valid is False
    assert any("leaked prefix 'Instruct:'" in err for err in res_prompt.errors)


# ---------------------------------------------------------------------------
# 7. ZIP Structure & Root-Only Enforcement
# ---------------------------------------------------------------------------


def test_zip_structure_valid(tmp_path: Path):
    """Verify that a ZIP with strictly one JSON at root passes validation."""
    valid_data = [{"id": 1, "relevant_docs": [10], "relevant_chunks": []}]
    json_path = tmp_path / "submission.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(valid_data, f)

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(json_path, arcname="submission.json")

    validator = SubmissionValidator()
    result = validator.validate_zip(zip_path)

    assert result.is_valid is True
    assert result.summary["zip_archive"] == "submission.zip"
    assert result.summary["inner_file"] == "submission.json"


def test_zip_structure_invalid_subdirectory(tmp_path: Path):
    """Verify that placing submission.json in a subfolder inside ZIP fails validation."""
    json_path = tmp_path / "submission.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"id": 1, "relevant_docs": [], "relevant_chunks": []}], f)

    zip_path = tmp_path / "bad_structure.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Invalid nested arcname
        zf.write(json_path, arcname="subfolder/submission.json")

    validator = SubmissionValidator()
    result = validator.validate_zip(zip_path)

    assert result.is_valid is False
    assert any("must be located at the root" in err for err in result.errors)


def test_zip_structure_multiple_files(tmp_path: Path):
    """Verify that multiple files inside ZIP fail validation."""
    json_path = tmp_path / "submission.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"id": 1, "relevant_docs": [], "relevant_chunks": []}], f)

    extra_file = tmp_path / "readme.txt"
    extra_file.write_text("extra content")

    zip_path = tmp_path / "multi_files.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(json_path, arcname="submission.json")
        zf.write(extra_file, arcname="readme.txt")

    validator = SubmissionValidator()
    result = validator.validate_zip(zip_path)

    assert result.is_valid is False
    assert any("must contain exactly ONE file" in err for err in result.errors)


# ---------------------------------------------------------------------------
# 8. Document ID Verification Against Corpus
# ---------------------------------------------------------------------------


def test_corpus_doc_id_warning(tmp_path: Path):
    """Verify warning when submission contains doc_ids not present in corpus."""
    corpus_file = tmp_path / "links_corpus.parquet"
    tbl = pa.Table.from_pydict({"doc_id": [100, 200, 300]})
    pq.write_table(tbl, corpus_file)

    data = [
        {
            "id": 1,
            "relevant_docs": [100, 9999],  # 9999 not in corpus
            "relevant_chunks": [],
        }
    ]

    validator = SubmissionValidator(corpus_path=corpus_file)
    result = validator.validate_json(data)

    assert any("9999" in warn for warn in result.warnings)


# ---------------------------------------------------------------------------
# 9. End-to-End Synthetic Export and Validation with Integer IDs
# ---------------------------------------------------------------------------


def test_exporter_end_to_end_synthetic(tmp_path: Path):
    """Test full exporter execution from synthetic query, chunk, and fusion artifacts."""
    # 1. Create mock query.parquet
    query_file = tmp_path / "query.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "id": [10, 20, 30],
                "query": ["huyết áp cao", "tiểu đường tuýp 2", "sốt xuất huyết"],
            }
        ),
        query_file,
    )

    # 2. Create mock chunks parquet shard
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pq.write_table(
        pa.Table.from_pydict(
            {
                "chunk_id": ["100_c0_a1", "200_c0_a1", "300_c0_a1"],
                "doc_id": [100, 200, 300],
                "chunk_index": [0, 0, 0],
                "text": [
                    "Nguyên nhân huyết áp cao do di truyền và lối sống.",
                    "Bệnh tiểu đường tuýp 2 cần kiểm soát lượng đường huyết.",
                    "Sốt xuất huyết lây truyền qua muỗi vằn Aedes aegypti.",
                ],
                "embed_text": [
                    "[Tiêu đề: Tim mạch] Nguyên nhân huyết áp cao do di truyền và lối sống.",
                    "[Tiêu đề: Nội tiết] Bệnh tiểu đường tuýp 2 cần kiểm soát lượng đường huyết.",
                    "[Tiêu đề: Truyền nhiễm] Sốt xuất huyết lây truyền qua muỗi vằn Aedes aegypti.",
                ],
            }
        ),
        chunks_dir / "chunks_part_00000.parquet",
    )

    # 3. Create mock rrf_fusion_top_1000.parquet
    fusion_file = tmp_path / "rrf_fusion_top_1000.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "query_id": [10, 10, 20],
                "doc_id": [100, 200, 200],
                "final_rank": [1, 2, 1],
                "fused_score": [0.032, 0.016, 0.032],
                "dense_best_chunk_id": ["100_c0_a1", "200_c0_a1", "200_c0_a1"],
            }
        ),
        fusion_file,
    )

    # 4. Run export
    out_json = tmp_path / "submission.json"
    out_zip = tmp_path / "submission.zip"

    cfg = SubmissionConfig(top_docs=5, top_chunks=5, include_chunk_order=True)
    exporter = SubmissionExporter(cfg)
    summary = exporter.export(
        retrieval_file=fusion_file,
        query_path=query_file,
        chunks_dir=chunks_dir,
        output_json_path=out_json,
        zip_output_path=out_zip,
    )

    # Verify summary
    assert summary.total_official_queries == 3
    assert summary.total_queries_with_results == 2
    assert summary.total_empty_queries == 1  # Query 30 had no hits
    assert summary.total_relevant_docs == 3
    assert summary.total_relevant_chunks == 3
    assert summary.missing_chunks_count == 0

    # 5. Read produced JSON
    with open(out_json, encoding="utf-8") as f:
        exported_data = json.load(f)

    # Check query 10: ensure STRICT INTEGER doc_ids
    q10 = next(item for item in exported_data if item["id"] == 10)
    assert q10["relevant_docs"] == [100, 200]
    assert all(isinstance(d, int) and not isinstance(d, bool) for d in q10["relevant_docs"])
    assert len(q10["relevant_chunks"]) == 2
    assert q10["relevant_chunks"][0]["doc_id"] == 100
    assert isinstance(q10["relevant_chunks"][0]["doc_id"], int)
    assert q10["relevant_chunks"][0]["chunk_text"] == "Nguyên nhân huyết áp cao do di truyền và lối sống."
    assert q10["relevant_chunks"][0]["chunk_order"] == 0

    # Check query 30 (empty result coverage)
    q30 = next(item for item in exported_data if item["id"] == 30)
    assert q30["relevant_docs"] == []
    assert q30["relevant_chunks"] == []

    # 6. Validate with SubmissionValidator
    validator = SubmissionValidator(query_path=query_file)
    json_result = validator.validate_file(out_json)
    assert json_result.is_valid is True
    assert len(json_result.errors) == 0

    zip_result = validator.validate_file(out_zip)
    assert zip_result.is_valid is True
    assert len(zip_result.errors) == 0
