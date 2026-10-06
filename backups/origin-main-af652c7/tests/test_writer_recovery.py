from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from crawler.config import load_config  # noqa: E402
from crawler.manifest import Manifest  # noqa: E402
from crawler.models import CrawlStatus  # noqa: E402
from crawler.writer import ParquetShardWriter, corpus_schema, verify_corpus  # noqa: E402
from tests.test_manifest import records, seed_group  # noqa: E402


def test_writer_commits_and_verifier_matches_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIBIOMIR_ROOT", str(tmp_path))
    config_path = Path(__file__).parents[1] / "config" / "crawler.yaml"
    config = load_config(config_path)
    config.ensure_directories()
    with Manifest(config.paths.manifest) as manifest:
        manifest.create_schema()
        seed_group(manifest)
        manifest.claim_targets(1)
        writer = ParquetShardWriter(config, manifest)
        writer.records = records()
        writer.buffer_bytes = 8
        writer.flush()
        result = verify_corpus(config, manifest)
        assert result["ok"] is True
        assert result["rows"] == 2


def test_recovery_commits_final_file_created_before_sqlite_commit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBIOMIR_ROOT", str(tmp_path))
    config_path = Path(__file__).parents[1] / "config" / "crawler.yaml"
    config = load_config(config_path)
    config.ensure_directories()
    with Manifest(config.paths.manifest) as manifest:
        manifest.create_schema()
        seed_group(manifest)
        manifest.claim_targets(1)
        writer = ParquetShardWriter(config, manifest)
        values = records()
        shard_name = "part-000000.parquet"
        temporary = config.paths.temp_dir / f"{shard_name}.tmp"
        final = config.paths.corpus_dir / shard_name
        manifest.stage_shard(shard_name, temporary, final, values)
        table = pa.Table.from_pylist(
            [record.parquet_dict() for record in values], schema=corpus_schema()
        )
        pq.write_table(table, final, compression="zstd")

        result = writer.recover()
        assert result["committed_shards"] == 1
        assert manifest.counts()[CrawlStatus.SUCCESS.value] == 2


def test_recovery_discards_temp_file_and_resets_tasks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIBIOMIR_ROOT", str(tmp_path))
    config_path = Path(__file__).parents[1] / "config" / "crawler.yaml"
    config = load_config(config_path)
    config.ensure_directories()
    with Manifest(config.paths.manifest) as manifest:
        manifest.create_schema()
        seed_group(manifest)
        manifest.claim_targets(1)
        writer = ParquetShardWriter(config, manifest)
        shard_name = "part-000000.parquet"
        temporary = config.paths.temp_dir / f"{shard_name}.tmp"
        final = config.paths.corpus_dir / shard_name
        manifest.stage_shard(shard_name, temporary, final, records())
        temporary.write_bytes(b"partial parquet")

        result = writer.recover()
        assert result["reset_targets"] == 1
        assert not temporary.exists()
        assert manifest.counts()[CrawlStatus.PENDING.value] == 2
