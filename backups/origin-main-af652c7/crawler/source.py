from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from crawler.config import AppConfig
from crawler.utils.files import replace_with_retry
from crawler.utils.hashing import sha256_file, stable_int
from crawler.utils.urls import domain_from_url, is_pdf_like, normalize_fetch_url


def _pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError("pyarrow is required; run `uv sync`") from exc
    return pq


def sync_source(config: AppConfig, force: bool = False) -> dict[str, object]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required; run `uv sync`") from exc

    destination = config.paths.source_file
    provenance_path = destination.with_suffix(destination.suffix + ".source.json")
    if destination.exists() and provenance_path.exists() and not force:
        return json.loads(provenance_path.read_text(encoding="utf-8"))

    api = HfApi()
    info = api.dataset_info(config.source.repo_id, revision=config.source.revision)
    files = api.list_repo_files(
        config.source.repo_id, repo_type="dataset", revision=info.sha
    )
    matches = [name for name in files if Path(name).name == config.source.filename]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {config.source.filename!r} in the dataset; found {matches}"
        )

    cached = Path(
        hf_hub_download(
            repo_id=config.source.repo_id,
            repo_type="dataset",
            filename=matches[0],
            revision=info.sha,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with cached.open("rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    replace_with_retry(temporary, destination)

    provenance: dict[str, object] = {
        "repo_id": config.source.repo_id,
        "revision": info.sha,
        "remote_path": matches[0],
        "local_path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "downloaded_at": datetime.now(UTC).isoformat(),
    }
    temporary_provenance = provenance_path.with_suffix(provenance_path.suffix + ".part")
    temporary_provenance.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    replace_with_retry(temporary_provenance, provenance_path)
    return provenance


def validate_source_schema(path: Path) -> tuple[str, str]:
    pq = _pyarrow()
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    id_column = "id" if "id" in names else "doc_id" if "doc_id" in names else None
    if id_column is None or "url" not in names:
        raise ValueError(
            f"source must contain id (or doc_id) and url columns; found {sorted(names)}"
        )
    return id_column, "url"


def iter_source_rows(path: Path, batch_size: int = 65_536) -> Iterator[tuple[int, str]]:
    pq = _pyarrow()
    id_column, url_column = validate_source_schema(path)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[id_column, url_column]):
        ids = batch.column(0).to_pylist()
        urls = batch.column(1).to_pylist()
        for raw_id, raw_url in zip(ids, urls, strict=True):
            if raw_id is None:
                raise ValueError("source contains a null document ID")
            if isinstance(raw_id, bool):
                raise ValueError(f"invalid boolean document ID: {raw_id!r}")
            if isinstance(raw_id, float) and not raw_id.is_integer():
                raise ValueError(f"non-integral document ID: {raw_id!r}")
            try:
                doc_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid document ID: {raw_id!r}") from exc
            if not -(2**63) <= doc_id < 2**63:
                raise ValueError(f"document ID is outside int64 range: {doc_id}")
            if raw_url is None or not str(raw_url).strip():
                raise ValueError(f"document {doc_id} has an empty URL")
            yield doc_id, str(raw_url).strip()


def inspect_source(path: Path, output_path: Path) -> dict[str, object]:
    domain_counts: Counter[str] = Counter()
    scheme_counts: Counter[str] = Counter()
    query_params: Counter[str] = Counter()
    invalid_samples: list[dict[str, object]] = []
    invalid_count = 0
    pdf_like = 0
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".sqlite", delete=False, dir=output_path.parent
    ) as handle:
        temp_path = Path(handle.name)
    connection = sqlite3.connect(temp_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE ids (doc_id INTEGER PRIMARY KEY);
            CREATE TABLE urls (url TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 1);
            """
        )
        for doc_id, url in iter_source_rows(path):
            total += 1
            try:
                connection.execute("INSERT INTO ids(doc_id) VALUES (?)", (doc_id,))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate document ID: {doc_id}") from exc
            connection.execute(
                """
                INSERT INTO urls(url, count) VALUES (?, 1)
                ON CONFLICT(url) DO UPDATE SET count = count + 1
                """,
                (url,),
            )
            try:
                normalized = normalize_fetch_url(url)
                parts = urlsplit(normalized)
                domain_counts[domain_from_url(normalized)] += 1
                scheme_counts[parts.scheme] += 1
                query_params.update(
                    key for key, _ in parse_qsl(parts.query, keep_blank_values=True)
                )
                pdf_like += int(is_pdf_like(normalized))
            except ValueError as exc:
                invalid_count += 1
                if len(invalid_samples) < 100:
                    invalid_samples.append({"doc_id": doc_id, "url": url, "error": str(exc)})
            if total % 50_000 == 0:
                connection.commit()
        connection.commit()
        unique_urls = connection.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
        duplicate_rows = connection.execute(
            "SELECT COALESCE(SUM(count - 1), 0) FROM urls WHERE count > 1"
        ).fetchone()[0]
        duplicate_samples = [
            {"url": row[0], "count": row[1]}
            for row in connection.execute(
                "SELECT url, count FROM urls WHERE count > 1 ORDER BY count DESC, url LIMIT 100"
            )
        ]
    finally:
        connection.close()
        temp_path.unlink(missing_ok=True)

    report: dict[str, object] = {
        "source": str(path),
        "source_size_bytes": path.stat().st_size,
        "source_mtime_ns": path.stat().st_mtime_ns,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_rows": total,
        "unique_urls": unique_urls,
        "duplicate_url_rows": duplicate_rows,
        "unique_domains": len(domain_counts),
        "pdf_like_urls": pdf_like,
        "schemes": dict(scheme_counts),
        "top_domains": [
            {"domain": domain, "count": count}
            for domain, count in domain_counts.most_common(100)
        ],
        "top_query_parameters": [
            {"parameter": name, "count": count}
            for name, count in query_params.most_common(100)
        ],
        "duplicate_url_samples": duplicate_samples,
        "invalid_url_count": invalid_count,
        "invalid_url_samples": invalid_samples,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_with_retry(temporary, output_path)
    return report


def create_pilot(
    source_path: Path,
    output_path: Path,
    size: int = 10_000,
    seed: int = 20261004,
) -> dict[str, object]:
    """Create a deterministic domain/content-type stratified pilot on disk."""
    if size < 1:
        raise ValueError("pilot size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".sqlite", delete=False, dir=output_path.parent
    ) as handle:
        temp_path = Path(handle.name)
    db = sqlite3.connect(temp_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute(
        """
        CREATE TABLE candidates (
            doc_id INTEGER PRIMARY KEY,
            url TEXT NOT NULL,
            domain TEXT NOT NULL,
            is_pdf INTEGER NOT NULL,
            score INTEGER NOT NULL
        )
        """
    )
    valid = 0
    try:
        for doc_id, url in iter_source_rows(source_path):
            try:
                normalized = normalize_fetch_url(url)
            except ValueError:
                continue
            db.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                (
                    doc_id,
                    url,
                    domain_from_url(normalized),
                    int(is_pdf_like(normalized)),
                    stable_int(str(doc_id), seed),
                ),
            )
            valid += 1
            if valid % 50_000 == 0:
                db.commit()
        db.commit()
        if valid < size:
            raise ValueError(f"requested {size} rows but source only has {valid} valid URLs")

        pdf_count = db.execute(
            "SELECT COUNT(*) FROM candidates WHERE is_pdf=1"
        ).fetchone()[0]
        pdf_quota = min(pdf_count, round(size * 0.20))
        class_quotas = {1: pdf_quota, 0: size - pdf_quota}
        db.execute("CREATE TABLE selected (doc_id INTEGER PRIMARY KEY)")
        cap = max(1, round(size * 0.15))

        for is_pdf, quota in class_quotas.items():
            if quota <= 0:
                continue
            proportional = round(quota * 0.70)
            rows = db.execute(
                """
                WITH ranked AS (
                    SELECT doc_id, score,
                           ROW_NUMBER() OVER (PARTITION BY domain ORDER BY score, doc_id) AS rn
                    FROM candidates WHERE is_pdf=?
                )
                SELECT doc_id FROM ranked WHERE rn <= ? ORDER BY score, doc_id LIMIT ?
                """,
                (is_pdf, cap, proportional),
            ).fetchall()
            db.executemany("INSERT OR IGNORE INTO selected VALUES (?)", rows)
            remaining = quota - len(rows)
            if remaining > 0:
                rows = db.execute(
                    """
                    WITH ranked AS (
                        SELECT c.doc_id, c.domain, c.score,
                               ROW_NUMBER() OVER (
                                   PARTITION BY c.domain ORDER BY c.score, c.doc_id
                               ) AS rn
                        FROM candidates c
                        LEFT JOIN selected s ON s.doc_id=c.doc_id
                        WHERE c.is_pdf=? AND s.doc_id IS NULL
                    )
                    SELECT doc_id FROM ranked
                    ORDER BY rn, domain, score, doc_id LIMIT ?
                    """,
                    (is_pdf, remaining),
                ).fetchall()
                db.executemany("INSERT OR IGNORE INTO selected VALUES (?)", rows)
        db.commit()

        selected_rows = db.execute(
            """
            SELECT c.doc_id, c.url, c.domain, c.is_pdf
            FROM candidates c JOIN selected s USING(doc_id)
            ORDER BY c.doc_id
            """
        ).fetchall()
        if len(selected_rows) != size:
            raise RuntimeError(
                f"pilot selection produced {len(selected_rows)} rows, expected {size}"
            )
    finally:
        db.close()
        temp_path.unlink(missing_ok=True)

    temporary = output_path.with_suffix(output_path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for doc_id, url, domain, pdf_flag in selected_rows:
            handle.write(
                json.dumps(
                    {"doc_id": doc_id, "url": url, "domain": domain, "is_pdf": bool(pdf_flag)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    replace_with_retry(temporary, output_path)
    return {
        "output": str(output_path),
        "size": len(selected_rows),
        "seed": seed,
        "pdf_like": sum(row[3] for row in selected_rows),
    }


def pilot_ids(path: Path) -> set[int]:
    values: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                doc_id = int(json.loads(line)["doc_id"])
                if doc_id in values:
                    raise ValueError(f"duplicate doc_id in selection: {doc_id}")
                values.add(doc_id)
    return values
