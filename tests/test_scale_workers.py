import asyncio
import csv
import json
import sqlite3
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from build_domain_profile import build_domain_profile
from crawler.config import load_config
from crawler.kaggle_runner import (
    CheckpointWriter,
    _prepare_output,
    _run_with_deadline,
    _validate_resume,
    run_worker,
)
from crawler.manifest import Manifest
from crawler.metrics import Metrics
from crawler.writer import ParquetShardWriter, corpus_schema
from merge_crawl_workers import merge_workers
from prepare_crawl_partitions import prepare_partitions
from tests.test_manifest import records, seed_group


def _source(path: Path, rows: list[tuple[int, str]]) -> None:
    table = pa.table(
        {
            "id": pa.array([row[0] for row in rows], type=pa.int64()),
            "url": [row[1] for row in rows],
        }
    )
    pq.write_table(table, path)


@pytest.mark.parametrize("code_layout", ["extracted", "zip"])
def test_kaggle_notebook_discovers_dataset_mounts(tmp_path: Path, code_layout: str) -> None:
    notebook = json.loads(
        (Path(__file__).parents[1] / "kaggle_crawl_worker.ipynb").read_text(encoding="utf-8")
    )
    bootstrap = "".join(notebook["cells"][1]["source"])
    discovery = bootstrap.split("subprocess.run(", maxsplit=1)[0]

    input_root = tmp_path / "input" / "datasets" / "owner" / "vibiomir-crawl-input"
    input_root.mkdir(parents=True)
    (input_root / "worker_1_kaggle.parquet").touch()
    (input_root / "domain_assignment.parquet").touch()
    code_root = tmp_path / "input" / "datasets" / "owner" / "vibiomir-crawler-code"
    code_root.mkdir(parents=True)
    if code_layout == "extracted":
        (code_root / "crawler").mkdir()
        (code_root / "crawler" / "kaggle_runner.py").write_text(
            "# supports --target-limit\n", encoding="utf-8"
        )
    else:
        with zipfile.ZipFile(code_root / "crawler_bundle.zip", "w") as archive:
            archive.writestr("crawler/kaggle_runner.py", "# supports --target-limit\n")

    discovery = discovery.replace(
        "kaggle_input = Path('/kaggle/input')",
        f"kaggle_input = Path({str(tmp_path / 'input')!r})",
    )
    app_dir = tmp_path / f"app-{code_layout}"
    namespace = {
        "WORKER_ID": 1,
        "MAX_DOCS": 1_000,
        "INPUT_DATASET_SLUG": "vibiomir-crawl-input",
        "CODE_DATASET_SLUG": "vibiomir-crawler-code",
        "APP_DIR": str(app_dir),
    }
    exec(compile(discovery, "bootstrap-cell", "exec"), namespace)

    assert namespace["INPUT_FILE"] == str(input_root / "worker_1_kaggle.parquet")
    assert (app_dir / "crawler" / "kaggle_runner.py").is_file()


def test_partitions_preserve_ids_urls_and_domain_ownership(tmp_path: Path) -> None:
    source = tmp_path / "links.parquet"
    rows = [
        (index, f"https://{domain}.example.com/article/{index}?x=1#original")
        for index, domain in enumerate("aaaaabbbbcccdddeef", start=100)
    ]
    _source(source, rows)

    result = prepare_partitions(source, tmp_path / "partitions")

    assert result["input_rows"] == result["partition_rows"] == len(rows)
    assert result["unique_doc_ids"] == len(rows)
    observed = []
    owner = {}
    for worker_id, path in enumerate(result["workers"]):
        for row in pq.read_table(path).to_pylist():
            observed.append((row["doc_id"], row["url"]))
            assert owner.setdefault(row["domain"], worker_id) == worker_id
    assert sorted(observed) == sorted(rows)
    with (tmp_path / "partitions" / "partition_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 6
    assert sum(int(row["url_count"]) for row in summary) == len(rows)


def test_partitions_reject_duplicate_doc_ids(tmp_path: Path) -> None:
    source = tmp_path / "links.parquet"
    _source(source, [(7, "https://a.example.com/one"), (7, "https://b.example.com/two")])

    with pytest.raises(ValueError, match="duplicate doc_id"):
        prepare_partitions(source, tmp_path / "partitions")
    assert not (tmp_path / "partitions").exists()


def test_fetch_timing_log_records_attempt_metadata(tmp_path: Path) -> None:
    timing_path = tmp_path / "fetch_timings.jsonl"
    metrics = Metrics(tmp_path, 20, tmp_path, timing_path)

    metrics.record_fetch(
        "example.com",
        1.25,
        fetch_key="key",
        attempt=2,
        status="FAILED",
        retryable=True,
    )

    row = json.loads(timing_path.read_text(encoding="utf-8"))
    assert row["fetch_key"] == "key"
    assert row["attempt"] == 2
    assert row["elapsed_seconds"] == 1.25


def test_domain_profile_and_workload_split(tmp_path: Path) -> None:
    source = tmp_path / "links.parquet"
    rows: list[tuple[int, str]] = []
    doc_id = 1
    domain_counts = [("big", 100)] + [(f"small{i}", 10) for i in range(5)]
    for domain, count in domain_counts:
        for index in range(count):
            rows.append((doc_id, f"https://{domain}.example/{index}"))
            doc_id += 1
    _source(source, rows)

    status_manifest = tmp_path / "status.sqlite"
    connection = sqlite3.connect(status_manifest)
    connection.execute("CREATE TABLE crawl_tasks(domain TEXT, status TEXT)")
    connection.executemany(
        "INSERT INTO crawl_tasks VALUES (?, 'SUCCESS')",
        [(url.split("/")[2],) for _doc_id, url in rows],
    )
    connection.commit()
    connection.close()

    timing_manifest = tmp_path / "timing.sqlite"
    connection = sqlite3.connect(timing_manifest)
    connection.execute("CREATE TABLE fetch_targets(fetch_key TEXT, domain TEXT, status TEXT)")
    timing_rows = []
    timing_lines = []
    for index, domain in enumerate(sorted({url.split("/")[2] for _id, url in rows})):
        key = f"key-{index}"
        timing_rows.append((key, domain, "SUCCESS"))
        timing_lines.append(
            json.dumps(
                {
                    "fetch_key": key,
                    "domain": domain,
                    "attempt": 1,
                    "status": "SUCCESS",
                    "retryable": False,
                    "elapsed_seconds": 10.0 if domain == "big.example" else 1.0,
                }
            )
        )
    connection.executemany("INSERT INTO fetch_targets VALUES (?, ?, ?)", timing_rows)
    connection.commit()
    connection.close()
    timing_log = tmp_path / "fetch_timings.jsonl"
    timing_log.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    profile_result = build_domain_profile(
        source,
        status_manifest,
        timing_manifest,
        timing_log,
        tmp_path / "profile",
    )
    with (tmp_path / "profile" / "domain_status_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        status_rows = list(csv.DictReader(handle))
    assert {"empty_content", "needs_js", "sample_count"}.issubset(status_rows[0])

    partition_result = prepare_partitions(
        source,
        tmp_path / "partitions-v2",
        Path(profile_result["domain_profile"]),
    )
    assignments = pq.read_table(
        tmp_path / "partitions-v2" / "domain_assignment.parquet"
    ).to_pylist()
    big = [row for row in assignments if row["domain"] == "big.example"]
    assert len(big) == 2
    assert len({row["worker_id"] for row in big}) == 2
    assert sum(row["url_count"] for row in big) == 100
    assert all(row["per_worker_domain_concurrency"] == 1 for row in big)
    assert partition_result["split_domains"] == 1


def _record(doc_id: int, url: str, status: str, text: str | None) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "url": url,
        "final_url": url + "?final=1",
        "domain": "example.com",
        "content_type": "text/html",
        "title": "Title" if text else None,
        "text": text,
        "language": "en" if text else None,
        "content_hash": "hash" if text else None,
        "http_status": 200 if text else 500,
        "crawl_status": status,
        "text_chars": len(text) if text else 0,
        "bytes_downloaded": 100,
    }


def _worker_output(root: Path, worker_id: int, rows: list[dict[str, object]]) -> None:
    root.mkdir()
    corpus = root / "corpus"
    corpus.mkdir()
    manifest_name = "crawl_manifest.sqlite" if worker_id == 0 else "manifest.sqlite"
    connection = sqlite3.connect(root / manifest_name)
    connection.execute(
        "CREATE TABLE crawl_tasks(doc_id INTEGER PRIMARY KEY,url TEXT,output_shard TEXT)"
    )
    connection.execute(
        "CREATE TABLE shards(shard_name TEXT PRIMARY KEY,status TEXT,row_count INTEGER)"
    )
    if rows:
        name = "part-000000.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=corpus_schema()), corpus / name)
        connection.execute("INSERT INTO shards VALUES (?,?,?)", (name, "COMMITTED", len(rows)))
        connection.executemany(
            "INSERT INTO crawl_tasks VALUES (?,?,?)",
            [(int(row["doc_id"]), str(row["url"]), name) for row in rows],
        )
    connection.commit()
    connection.close()


def test_merge_prefers_success_and_preserves_winning_record(tmp_path: Path) -> None:
    url = "https://example.com/article"
    inputs = {worker_id: tmp_path / f"worker-{worker_id}" for worker_id in range(6)}
    _worker_output(inputs[0], 0, [_record(1, "https://example.com/one", "SUCCESS", "one")])
    _worker_output(inputs[1], 1, [_record(10, url, "FAILED", None)])
    winning = _record(10, url, "SUCCESS", "useful medical text")
    _worker_output(inputs[2], 2, [winning])
    for worker_id in (3, 4, 5):
        _worker_output(inputs[worker_id], worker_id, [])

    summary = merge_workers(inputs, tmp_path / "merged")

    assert summary["source_rows"] == 3
    assert summary["unique_doc_ids"] == summary["output_rows"] == 2
    assert summary["conflicts"] == 1
    output = pq.read_table(tmp_path / "merged" / "corpus" / "part-000000.parquet")
    rows = {row["doc_id"]: row for row in output.to_pylist()}
    assert rows[10] == winning
    conflict = json.loads(
        (tmp_path / "merged" / "merge_conflicts.jsonl").read_text(encoding="utf-8")
    )
    assert conflict["selected"] == "candidate"


def test_merge_rejects_conflicting_original_url(tmp_path: Path) -> None:
    inputs = {worker_id: tmp_path / f"worker-{worker_id}" for worker_id in range(6)}
    _worker_output(inputs[0], 0, [_record(10, "https://example.com/first", "FAILED", None)])
    _worker_output(inputs[1], 1, [_record(10, "https://example.com/second", "SUCCESS", "text")])
    for worker_id in (2, 3, 4, 5):
        _worker_output(inputs[worker_id], worker_id, [])

    with pytest.raises(ValueError, match="conflicting original URLs"):
        merge_workers(inputs, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_checkpoint_is_written_after_writer_commit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIBIOMIR_ROOT", str(tmp_path))
    config = load_config(Path(__file__).parents[1] / "config" / "crawler.yaml")
    config.ensure_directories()
    with Manifest(config.paths.manifest) as manifest:
        manifest.create_schema()
        seed_group(manifest)
        manifest.claim_targets(1)
        checkpoint = CheckpointWriter(
            tmp_path / "checkpoint.json",
            manifest,
            worker_id=1,
            partition_sha256="partition-hash",
            every_docs=2,
            started_monotonic=time.monotonic(),
        )
        writer = ParquetShardWriter(config, manifest, on_commit=checkpoint.on_shard_commit)
        writer.records = records()
        writer.flush()
        data = json.loads(checkpoint.path.read_text(encoding="utf-8"))
        assert data["committed_documents"] == 2
        assert data["committed_shards"] == 1
        assert data["status_counts"]["SUCCESS"] == 2


def test_resume_copies_full_output_and_checks_partition(tmp_path: Path) -> None:
    previous = tmp_path / "input" / "old-run" / "crawl_worker_1"
    previous.mkdir(parents=True)
    (previous / "corpus").mkdir()
    (previous / "corpus" / "part-000000.parquet").write_bytes(b"stored shard")
    with Manifest(previous / "manifest.sqlite") as manifest:
        manifest.create_schema()
        manifest.connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            [("worker_id", "1"), ("partition_sha256", "partition-hash")],
        )
    (previous / "checkpoint.json").write_text(
        json.dumps({"worker_id": 1, "partition_sha256": "partition-hash"}),
        encoding="utf-8",
    )
    destination = tmp_path / "working" / "crawl_worker_1"

    assert _prepare_output(destination, tmp_path / "input", 1) == str(previous.resolve())
    assert (destination / "corpus" / "part-000000.parquet").read_bytes() == b"stored shard"
    with Manifest(destination / "manifest.sqlite") as manifest:
        _validate_resume(
            manifest,
            destination / "checkpoint.json",
            worker_id=1,
            partition_sha256="partition-hash",
        )
        with pytest.raises(RuntimeError, match="does not match"):
            _validate_resume(
                manifest,
                destination / "checkpoint.json",
                worker_id=1,
                partition_sha256="different-hash",
            )


def test_runtime_guard_requests_graceful_stop() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.stopped = asyncio.Event()

        def request_stop(self) -> None:
            self.stopped.set()

        async def run(self, target_limit: int | None = None) -> dict[str, int]:
            assert target_limit is None
            await self.stopped.wait()
            return {"completed": 1}

    async def scenario() -> tuple[dict[str, int], bool]:
        pipeline = FakePipeline()
        deadline = asyncio.Event()
        result = await _run_with_deadline(pipeline, 0.01, deadline)  # type: ignore[arg-type]
        return result, deadline.is_set()

    result, deadline_reached = asyncio.run(scenario())
    assert result == {"completed": 1}
    assert deadline_reached is True


def test_kaggle_runner_target_limit_then_resume_keeps_success(tmp_path: Path) -> None:
    requests = {"/article-1": 0, "/article-2": 0, "/article-3": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/robots.txt":
                body = b"User-agent: *\nAllow: /\n"
                content_type = "text/plain"
            else:
                requests[self.path] += 1
                body = (
                    b"<html><main><h1>Medical article</h1><p>"
                    + b"Medical guidance with useful details and measurements. " * 5
                    + b"</p></main></html>"
                )
                content_type = "text/html"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        partition_dir = tmp_path / "input"
        partition_dir.mkdir()
        input_file = partition_dir / "worker_1_kaggle.parquet"
        urls = [f"http://127.0.0.1:{server.server_port}/article-{index}" for index in range(1, 4)]
        pq.write_table(
            pa.table(
                {
                    "doc_id": [42, 43, 44],
                    "url": urls,
                    "domain": ["127.0.0.1"] * 3,
                }
            ),
            input_file,
        )
        pq.write_table(
            pa.table({"domain": ["127.0.0.1"], "worker_id": [1], "url_count": [3]}),
            partition_dir / "domain_assignment.parquet",
        )
        config_path = Path(__file__).parents[1] / "config" / "crawler.yaml"
        first_output = tmp_path / "first" / "crawl_worker_1"
        first = run_worker(
            worker_id=1,
            input_file=input_file,
            output_dir=first_output,
            config_path=config_path,
            resume_root=None,
            max_runtime_hours=0.01,
            checkpoint_every_docs=1,
            global_concurrency=2,
            per_domain_concurrency=1,
            target_limit=1,
        )
        assert first["exit_reason"] == "target_limit"
        assert first["config"]["target_limit"] == 1
        assert first["status_counts"]["SUCCESS"] == 1
        assert first["verification"]["ok"] is True
        assert first["verification"]["rows"] == 1
        assert sum(requests.values()) == 1

        second_output = tmp_path / "second" / "crawl_worker_1"
        second = run_worker(
            worker_id=1,
            input_file=input_file,
            output_dir=second_output,
            config_path=config_path,
            resume_root=tmp_path / "first",
            max_runtime_hours=0.01,
            checkpoint_every_docs=1,
            global_concurrency=2,
            per_domain_concurrency=1,
        )
        assert second["exit_reason"] == "completed"
        assert second["config"]["target_limit"] is None
        assert second["status_counts"]["SUCCESS"] == 3
        assert second["verification"]["rows"] == 3
        assert sum(requests.values()) == 3
        assert sorted(requests.values()) == [1, 1, 1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
