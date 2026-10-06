from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(StrictModel):
    repo_id: str = "AIGuruTinix/ViBioMIR"
    filename: str = "links_corpus.parquet"
    revision: str = "main"


class PathsConfig(StrictModel):
    source_dir: Path = Path("data/source")
    source_file: Path = Path("data/source/links_corpus.parquet")
    manifest: Path = Path("data/crawl/crawl_manifest.sqlite")
    corpus_dir: Path = Path("data/crawl/corpus")
    temp_dir: Path = Path("data/crawl/tmp")
    pilots_dir: Path = Path("data/crawl/pilots")
    debug_raw_dir: Path = Path("data/crawl/debug_raw")
    logs_dir: Path = Path("logs")


class CrawlerRuntimeConfig(StrictModel):
    global_concurrency: int = Field(default=64, ge=1)
    per_domain_concurrency: int = Field(default=3, ge=1)
    extraction_workers: int = Field(default=0, ge=0)
    connect_timeout_seconds: float = Field(default=7, gt=0)
    total_timeout_seconds: float = Field(default=30, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    max_redirects: int = Field(default=10, ge=0)
    max_html_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    max_pdf_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    user_agent: str
    retry_base_seconds: float = Field(default=1.0, gt=0)
    robots_cache_seconds: int = Field(default=86400, ge=60)
    robots_timeout_seconds: float = Field(default=10, gt=0)
    debug_raw_limit: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_concurrency(self) -> CrawlerRuntimeConfig:
        if self.per_domain_concurrency > self.global_concurrency:
            raise ValueError("per_domain_concurrency cannot exceed global_concurrency")
        return self


class ExtractionConfig(StrictModel):
    min_text_chars: int = Field(default=100, ge=1)
    pdf_min_chars_per_page: int = Field(default=50, ge=0)
    language_confidence: float = Field(default=0.15, ge=0, le=1)


class WriterConfig(StrictModel):
    rows_per_shard: int = Field(default=10_000, ge=1)
    max_buffer_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    compression: str = "zstd"
    row_group_size: int = Field(default=2048, ge=1)


class QueuesConfig(StrictModel):
    fetch: int = Field(default=256, ge=1)
    extract: int = Field(default=128, ge=1)
    write: int = Field(default=2048, ge=1)
    response_byte_budget: int = Field(default=512 * 1024 * 1024, gt=0)


class MetricsConfig(StrictModel):
    interval_seconds: float = Field(default=20, gt=0)


class AppConfig(StrictModel):
    source: SourceConfig
    paths: PathsConfig
    crawler: CrawlerRuntimeConfig
    extraction: ExtractionConfig
    writer: WriterConfig
    queues: QueuesConfig
    metrics: MetricsConfig

    def ensure_directories(self) -> None:
        for path in (
            self.paths.source_dir,
            self.paths.manifest.parent,
            self.paths.corpus_dir,
            self.paths.temp_dir,
            self.paths.pilots_dir,
            self.paths.debug_raw_dir,
            self.paths.logs_dir,
            self.paths.logs_dir / "metrics",
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path = "config/crawler.yaml") -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = AppConfig.model_validate(raw)
    root = Path(os.environ.get("VIBIOMIR_ROOT", Path.cwd())).resolve()
    resolved = config.model_copy(deep=True)
    for name in PathsConfig.model_fields:
        value = getattr(resolved.paths, name)
        if not value.is_absolute():
            setattr(resolved.paths, name, root / value)
    return resolved


_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def apply_run_name(config: AppConfig, run_name: str | None) -> AppConfig:
    """Isolate mutable crawl artifacts while keeping source and pilots shared."""
    if run_name is None:
        return config
    if not _RUN_NAME.fullmatch(run_name) or run_name in {".", ".."}:
        raise ValueError(
            "run name must start with a letter or number and contain only "
            "letters, numbers, '.', '_' or '-'"
        )

    resolved = config.model_copy(deep=True)
    run_root = config.paths.manifest.parent / "runs" / run_name
    resolved.paths.manifest = run_root / "crawl_manifest.sqlite"
    resolved.paths.corpus_dir = run_root / "corpus"
    resolved.paths.temp_dir = run_root / "tmp"
    resolved.paths.debug_raw_dir = run_root / "debug_raw"
    resolved.paths.logs_dir = config.paths.logs_dir / "runs" / run_name
    return resolved
