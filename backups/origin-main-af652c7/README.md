# ViBioMIR crawler

Resumable HTML/PDF crawler for the URL corpus in
[`AIGuruTinix/ViBioMIR`](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR).
The design follows `ViBioMIR_Crawl_Pipeline_Spec.md`: source IDs are preserved,
clean documents are stored in ZSTD-compressed Parquet shards, and SQLite is the
source of truth for crawl state.

## Setup

The supported production target is Windows x86-64 with Python 3.13. Paths,
process locking, graceful Ctrl+C handling, process workers, SQLite recovery,
and atomic Parquet writes all have Windows-specific handling.

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --extra dev
uv lock
uv run pytest
uv run python -m crawler --help
```

The same setup is available as `powershell -ExecutionPolicy Bypass -File
scripts/setup_windows.ps1`.

All runtime values live in `config/crawler.yaml`. Paths are resolved from the
current directory, or from `VIBIOMIR_ROOT` when that environment variable is
set. Raw responses are discarded. For bounded extraction debugging, set
`crawler.debug_raw_limit` to a value from 1 through 100.

## Prepare a 10k pilot

```powershell
uv run python -m crawler source sync
uv run python -m crawler source inspect
uv run python -m crawler pilot create --size 10000 --seed 20261004
uv run python -m crawler manifest init --selection data/crawl/pilots/pilot-10000-seed-20261004.jsonl
```

The source sync pins the resolved Hugging Face commit and records its SHA-256
in `data/source/links_corpus.parquet.source.json`. If the source file is copied
to the machine separately, omit `source sync` and pass `--source-path` to the
remaining preparation commands.

## Run and resume

```powershell
uv run python -m crawler crawl run --target-limit 100
uv run python -m crawler crawl status
uv run python -m crawler corpus verify
uv run python -m crawler crawl run
uv run python -m crawler corpus sample --output logs/pilot-review.jsonl
uv run python -m crawler report build --output logs/pilot-report.json
```

Interactive crawl runs show a colored progress bar with throughput, ETA, active
requests, and one completion line per URL. Use `--progress bar` for only the
live bar or `--progress off` for machine-oriented runs. Progress is written to
stderr; the final JSON summary remains on stdout. After all URLs finish, the
bar shows `flushing` until the final partial Parquet shard is durably committed.

Fill the five `review_*` fields in the 200-row review JSONL. The pilot is ready
for expansion when corpus verification passes, at least 90% of reviewed
`SUCCESS` rows have correct title/body/structure, and no unexplained systemic
failure affects more than 5% of the pilot.

Use JSON booleans (`true`/`false`), not strings, for every review field. Mark
`review_noise_ok=false` when menus, related articles, sharing widgets, comments,
or other page chrome leak into the extracted body, and describe the pattern in
`review_notes`. The generated report includes failure counts per criterion and
groups unacceptable/noisy successful documents by domain so repeated templates
can be fixed before scaling the crawl.

Run the same `crawl run` command after a crash or restart. Startup reconciles
Parquet shards with staged SQLite metadata before scheduling new requests.
Only one process may use a manifest at a time.

To rerun the same selection without overwriting an earlier baseline, put the
global `--run-name` option before the command. Source data and pilot selections
remain shared, while the manifest, corpus, temporary files, metrics, and default
review/report outputs are isolated under the run name:

```powershell
uv run python -m crawler --run-name pilot-100-v2 manifest init `
  --selection data/crawl/pilots/pilot-100-seed-20261004.jsonl
uv run python -m crawler --run-name pilot-100-v2 crawl run --target-limit 100
uv run python -m crawler --run-name pilot-100-v2 corpus verify
uv run python -m crawler --run-name pilot-100-v2 corpus sample
```

The new review file is written to
`logs/runs/pilot-100-v2/pilot-review.jsonl`; the unnamed baseline is unchanged.

## Read-only crawl audit

Create deterministic QA samples and a domain-level risk summary without
modifying the corpus or manifest:

```powershell
uv run python audit_crawl.py --run-name pilot-10000-v1
```

This writes `qa_samples.jsonl`, a self-contained `qa_review.html`, and
Excel-friendly `domain_summary.csv` under the selected run's log directory.

To stop cleanly on Windows, press Ctrl+C once. The scheduler stops claiming new
work, finishes the bounded in-flight queues, commits the final shard, and exits.

## Move from pilot to the full manifest

After the pilot has been inspected, add the remaining IDs without replacing
successful work:

```powershell
uv run python -m crawler manifest init
```

The initializer is idempotent. Existing `doc_id` rows retain their state. A
source row that reuses an existing `doc_id` with a different URL is rejected.

## Tests

```powershell
uv run pytest
uv run ruff check .
```

Network tests use a local HTTP server. They do not contact external websites.
