"""Independent Submission Validator for ViBioMIR competition submissions.

Performs comprehensive pre-flight verification of submission.json and
submission.zip before uploading to the evaluation portal.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from retrieval.searcher import BM25Searcher

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Detailed results of submission validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> str:
        """Render a formatted human-readable validation report."""
        lines: list[str] = []
        lines.append("=" * 65)
        lines.append("=== VIBIOMIR SUBMISSION VALIDATION REPORT ===")
        lines.append("=" * 65)

        status_str = "PASSED [VALID SUBMISSION]" if self.is_valid else "FAILED [INVALID SUBMISSION]"
        lines.append(f"Overall Status : {status_str}")

        if self.summary:
            lines.append("\n[SUMMARY METRICS]")
            for k, v in self.summary.items():
                lines.append(f"  - {k:<28}: {v}")

        if self.errors:
            lines.append(f"\n[ERRORS - {len(self.errors)} CRITICAL ISSUE(S)]")
            for i, err in enumerate(self.errors[:30], 1):
                lines.append(f"  {i}. [ERROR] {err}")
            if len(self.errors) > 30:
                lines.append(f"  ... and {len(self.errors) - 30} more errors.")

        if self.warnings:
            lines.append(f"\n[WARNINGS - {len(self.warnings)} NOTICE(S)]")
            for i, warn in enumerate(self.warnings[:20], 1):
                lines.append(f"  {i}. [WARN]  {warn}")
            if len(self.warnings) > 20:
                lines.append(f"  ... and {len(self.warnings) - 20} more warnings.")

        lines.append("=" * 65)
        return "\n".join(lines)


class SubmissionValidator:
    """Independent validator checking ViBioMIR official submission JSON and ZIP files."""

    FORBIDDEN_RECORD_KEYS = {
        "score",
        "rank",
        "bm25_score",
        "bm25_rank",
        "dense_score",
        "dense_rank",
        "fused_score",
        "final_rank",
        "best_chunk_id",
        "query_text",
        "metadata",
    }

    FORBIDDEN_CHUNK_KEYS = {
        "score",
        "rank",
        "chunk_id",
        "title",
        "heading",
        "tokens",
        "embed_text",
        "similarity",
    }

    FORBIDDEN_TEXT_PREFIXES = (
        "Instruct:",
        "[Tiêu đề:",
        "[Mục:",
        "Query:",
    )

    def __init__(
        self,
        query_path: Path | str | None = None,
        corpus_path: Path | str | None = None,
    ) -> None:
        self.query_path = Path(query_path) if query_path else None
        self.corpus_path = Path(corpus_path) if corpus_path else None

    def validate_file(self, target_path: Path | str) -> ValidationResult:
        """Validate either a submission.json or a submission.zip file."""
        p = Path(target_path)
        if not p.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"Target submission file does not exist: {p.resolve()}"],
            )

        if p.suffix.lower() == ".zip":
            return self.validate_zip(p)
        elif p.suffix.lower() == ".json":
            return self.validate_json_file(p)
        else:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unsupported submission file format '{p.suffix}'. Expected .json or .zip"],
            )

    def validate_zip(self, zip_path: Path | str) -> ValidationResult:
        """Validate submission.zip structure and inner submission.json."""
        p = Path(zip_path)
        errors: list[str] = []
        warnings: list[str] = []

        if not zipfile.is_zipfile(p):
            return ValidationResult(
                is_valid=False,
                errors=[f"File '{p.name}' is not a valid ZIP archive."],
            )

        try:
            with zipfile.ZipFile(p, "r") as zf:
                infolist = zf.infolist()

                # Rule: exactly one file in ZIP
                if len(infolist) == 0:
                    return ValidationResult(
                        is_valid=False,
                        errors=["ZIP archive is empty."],
                    )
                if len(infolist) > 1:
                    errors.append(
                        f"ZIP archive must contain exactly ONE file at root, but found {len(infolist)} files: "
                        f"{[item.filename for item in infolist]}"
                    )

                target_info = infolist[0]
                target_filename = target_info.filename

                # Rule: file must be at root (no subdirectories)
                if "/" in target_filename or "\\" in target_filename:
                    errors.append(
                        f"ZIP member '{target_filename}' must be located at the root of the archive, "
                        f"not inside subdirectories."
                    )

                # Rule: file must be .json
                if not target_filename.lower().endswith(".json"):
                    errors.append(
                        f"ZIP member '{target_filename}' must be a JSON file ending with .json"
                    )

                # Extract and parse JSON
                try:
                    with zf.open(target_info) as jf:
                        raw_data = json.load(jf)
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        errors=errors + [f"Failed to parse inner JSON file '{target_filename}': {e}"],
                    )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to inspect ZIP archive: {e}"],
            )

        # Validate inner JSON content
        content_res = self.validate_json(raw_data)
        all_errors = errors + content_res.errors
        all_warnings = warnings + content_res.warnings

        summary = dict(content_res.summary)
        summary["zip_archive"] = p.name
        summary["inner_file"] = target_filename

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            summary=summary,
        )

    def validate_json_file(self, json_path: Path | str) -> ValidationResult:
        """Validate submission JSON directly from disk."""
        p = Path(json_path)
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to parse JSON file '{p.resolve()}': {e}"],
            )

        res = self.validate_json(data)
        res.summary["json_file"] = p.name
        return res

    def validate_json(self, data: Any) -> ValidationResult:
        """Validate parsed submission JSON data against official schema rules."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Root structure: must be list
        if not isinstance(data, list):
            return ValidationResult(
                is_valid=False,
                errors=[f"Root submission object must be a JSON array (list), got {type(data).__name__}"],
            )

        total_queries = len(data)
        if total_queries == 0:
            return ValidationResult(
                is_valid=False,
                errors=["Submission JSON array is empty."],
            )

        # 2. Query ID collection & uniqueness
        seen_query_ids: set[int] = set()
        duplicate_query_ids: set[int] = set()

        total_docs = 0
        total_chunks = 0
        empty_queries_count = 0
        docs_per_query_counts: list[int] = []
        chunks_per_query_counts: list[int] = []
        prefix_leak_count = 0

        # Check each record
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"Item at index {idx} is not a JSON object: {type(item).__name__}")
                continue

            # Check required keys
            if "id" not in item:
                errors.append(f"Item at index {idx} is missing required 'id' field.")
                continue

            q_id = item["id"]
            # Rule: id must be integer
            if not isinstance(q_id, int) or isinstance(q_id, bool):
                errors.append(f"Item at index {idx} has invalid 'id' type: expected int, got {type(q_id).__name__} ({q_id!r})")
                continue

            if q_id in seen_query_ids:
                duplicate_query_ids.add(q_id)
            seen_query_ids.add(q_id)

            # Check forbidden keys at item level
            extra_keys = set(item.keys()) - {"id", "relevant_docs", "relevant_chunks"}
            if extra_keys:
                forbidden = extra_keys & self.FORBIDDEN_RECORD_KEYS
                if forbidden:
                    errors.append(f"Query {q_id} contains forbidden metadata fields: {forbidden}")
                else:
                    warnings.append(f"Query {q_id} contains non-standard fields: {extra_keys}")

            # Check relevant_docs
            if "relevant_docs" not in item:
                errors.append(f"Query {q_id} is missing required field 'relevant_docs'.")
                rel_docs = []
            else:
                rel_docs = item["relevant_docs"]
                if not isinstance(rel_docs, list):
                    errors.append(f"Query {q_id}: 'relevant_docs' must be a list, got {type(rel_docs).__name__}")
                    rel_docs = []
                else:
                    for d_idx, doc in enumerate(rel_docs):
                        if not isinstance(doc, (int, str)) or isinstance(doc, bool):
                            errors.append(
                                f"Query {q_id}: relevant_docs[{d_idx}] must be int or string, got {type(doc).__name__} ({doc!r})"
                            )

            # Check relevant_chunks
            if "relevant_chunks" not in item:
                errors.append(f"Query {q_id} is missing required field 'relevant_chunks'.")
                rel_chunks = []
            else:
                rel_chunks = item["relevant_chunks"]
                if not isinstance(rel_chunks, list):
                    errors.append(f"Query {q_id}: 'relevant_chunks' must be a list, got {type(rel_chunks).__name__}")
                    rel_chunks = []
                else:
                    rel_doc_set = set(rel_docs) if isinstance(rel_docs, list) else set()

                    for c_idx, chunk in enumerate(rel_chunks):
                        if not isinstance(chunk, dict):
                            errors.append(
                                f"Query {q_id}: relevant_chunks[{c_idx}] must be a dict, got {type(chunk).__name__}"
                            )
                            continue

                        # Check forbidden keys in chunk
                        chunk_extra = set(chunk.keys()) - {"doc_id", "chunk_text", "chunk_order"}
                        if chunk_extra:
                            errors.append(f"Query {q_id}, chunk {c_idx} contains forbidden fields: {chunk_extra}")

                        # Check doc_id
                        if "doc_id" not in chunk:
                            errors.append(f"Query {q_id}, chunk {c_idx} missing 'doc_id'")
                        elif not isinstance(chunk["doc_id"], (int, str)) or isinstance(chunk["doc_id"], bool):
                            errors.append(
                                f"Query {q_id}, chunk {c_idx} 'doc_id' must be int or string, got {type(chunk['doc_id']).__name__}"
                            )
                        elif rel_doc_set:
                            cid = chunk["doc_id"]
                            str_rel_set = {str(x) for x in rel_doc_set}
                            if str(cid) not in str_rel_set:
                                warnings.append(
                                    f"Query {q_id}: chunk doc_id '{cid}' is not in relevant_docs"
                                )

                        # Check chunk_text
                        if "chunk_text" not in chunk:
                            errors.append(f"Query {q_id}, chunk {c_idx} missing 'chunk_text'")
                        elif not isinstance(chunk["chunk_text"], str):
                            errors.append(
                                f"Query {q_id}, chunk {c_idx} 'chunk_text' must be string, got {type(chunk['chunk_text']).__name__}"
                            )
                        else:
                            ctext = chunk["chunk_text"].strip()
                            if not ctext:
                                warnings.append(f"Query {q_id}, chunk {c_idx} has empty chunk_text")

                            # Check for leaked model instructions or [Tiêu đề: ...]
                            for pfx in self.FORBIDDEN_TEXT_PREFIXES:
                                if ctext.startswith(pfx):
                                    prefix_leak_count += 1
                                    errors.append(
                                        f"Query {q_id}, chunk {c_idx}: chunk_text starts with leaked prefix '{pfx}'"
                                    )
                                    break

                        # Check optional chunk_order
                        if "chunk_order" in chunk:
                            co = chunk["chunk_order"]
                            if not isinstance(co, int) or isinstance(co, bool) or co < 0:
                                errors.append(
                                    f"Query {q_id}, chunk {c_idx}: chunk_order must be non-negative int, got {co!r}"
                                )

            if len(rel_docs) == 0 and len(rel_chunks) == 0:
                empty_queries_count += 1

            total_docs += len(rel_docs)
            total_chunks += len(rel_chunks)
            docs_per_query_counts.append(len(rel_docs))
            chunks_per_query_counts.append(len(rel_chunks))

        # Check duplicate query IDs
        if duplicate_query_ids:
            errors.append(
                f"Found {len(duplicate_query_ids)} duplicate query ID(s) in submission: {sorted(list(duplicate_query_ids))[:10]}"
            )

        # 3. Check official query coverage against query.parquet if provided
        if self.query_path and self.query_path.exists():
            try:
                official_queries = BM25Searcher.load_queries_from_parquet(self.query_path)
                official_ids = {q.query_id for q in official_queries}

                missing_ids = official_ids - seen_query_ids
                if missing_ids:
                    errors.append(
                        f"Missing {len(missing_ids)} official query ID(s) from {self.query_path.name}: "
                        f"{sorted(list(missing_ids))[:10]}"
                    )

                extra_ids = seen_query_ids - official_ids
                if extra_ids:
                    errors.append(
                        f"Found {len(extra_ids)} extra query ID(s) not present in official query dataset: "
                        f"{sorted(list(extra_ids))[:10]}"
                    )
            except Exception as e:
                warnings.append(f"Could not load official queries from {self.query_path}: {e}")

        # 4. Check doc_ids against corpus if corpus_path provided
        if self.corpus_path and self.corpus_path.exists():
            try:
                tbl = pq.read_table(self.corpus_path, columns=["doc_id"])
                corpus_doc_ids = set(tbl.column("doc_id").to_pylist())

                all_submission_doc_ids: set[int] = set()
                for item in data:
                    for d in item.get("relevant_docs", []):
                        try:
                            all_submission_doc_ids.add(int(d))
                        except (ValueError, TypeError):
                            pass

                invalid_docs = all_submission_doc_ids - corpus_doc_ids
                if invalid_docs:
                    warnings.append(
                        f"Found {len(invalid_docs)} document ID(s) in submission that do not appear in "
                        f"corpus {self.corpus_path.name}: {sorted(list(invalid_docs))[:10]}"
                    )
            except Exception as e:
                warnings.append(f"Could not check doc_ids against corpus {self.corpus_path}: {e}")

        # Compute summary metrics
        avg_docs = round(total_docs / total_queries, 2) if total_queries > 0 else 0.0
        avg_chunks = round(total_chunks / total_queries, 2) if total_queries > 0 else 0.0
        max_docs = max(docs_per_query_counts) if docs_per_query_counts else 0
        min_docs = min(docs_per_query_counts) if docs_per_query_counts else 0

        summary_data: dict[str, Any] = {
            "total_queries": total_queries,
            "unique_query_ids": len(seen_query_ids),
            "duplicate_queries": len(duplicate_query_ids),
            "total_relevant_docs": total_docs,
            "avg_docs_per_query": avg_docs,
            "min_max_docs_range": f"[{min_docs}, {max_docs}]",
            "total_relevant_chunks": total_chunks,
            "avg_chunks_per_query": avg_chunks,
            "empty_queries_count": empty_queries_count,
            "prefix_leak_violations": prefix_leak_count,
            "schema_errors_count": len(errors),
            "warnings_count": len(warnings),
        }

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            summary=summary_data,
        )
