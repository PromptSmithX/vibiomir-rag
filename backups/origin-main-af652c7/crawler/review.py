from __future__ import annotations

import json
from pathlib import Path

from crawler.config import AppConfig
from crawler.manifest import Manifest
from crawler.utils.files import replace_with_retry
from crawler.utils.hashing import stable_int


def create_review_sample(
    config: AppConfig,
    manifest: Manifest,
    output: Path,
    success_size: int = 150,
    error_size: int = 50,
    seed: int = 20261004,
) -> dict[str, object]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required; run `uv sync`") from exc

    manifest.connection.create_function(
        "stable_score", 1, lambda value: stable_int(str(value), seed), deterministic=True
    )

    def select(where: str, limit: int) -> list[int]:
        return [
            int(row[0])
            for row in manifest.connection.execute(
                f"""
                WITH ranked AS (
                    SELECT doc_id, domain, content_type, stable_score(doc_id) AS score,
                           ROW_NUMBER() OVER (
                               PARTITION BY domain, COALESCE(content_type, '')
                               ORDER BY stable_score(doc_id), doc_id
                           ) AS group_rank
                    FROM crawl_tasks WHERE {where} AND output_shard IS NOT NULL
                )
                SELECT doc_id FROM ranked
                ORDER BY group_rank, score, doc_id LIMIT ?
                """,
                (limit,),
            )
        ]

    success_ids = select("status='SUCCESS'", success_size)
    error_ids = select("status!='SUCCESS'", error_size)
    selected = set(success_ids + error_ids)
    found: dict[int, dict[str, object]] = {}
    if selected:
        for shard in sorted(config.paths.corpus_dir.glob("part-*.parquet")):
            table = pq.read_table(shard)
            for row in table.to_pylist():
                doc_id = int(row["doc_id"])
                if doc_id in selected:
                    found[doc_id] = row

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for doc_id in success_ids + error_ids:
            row = found.get(doc_id)
            if row is None:
                continue
            row["review_title_correct"] = None
            row["review_body_correct"] = None
            row["review_structure_ok"] = None
            row["review_noise_ok"] = None
            row["review_medical_values_ok"] = None
            row["review_notes"] = ""
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    replace_with_retry(temporary, output)
    return {
        "output": str(output),
        "requested_success": success_size,
        "requested_errors": error_size,
        "written": len(found),
        "seed": seed,
    }
