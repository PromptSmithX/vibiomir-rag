from pathlib import Path

import pytest

from crawler.cli import build_parser
from crawler.config import apply_run_name, load_config


def test_named_run_isolates_mutable_paths_and_shares_source_and_pilots() -> None:
    base = load_config("config/crawler.yaml")
    named = apply_run_name(base, "pilot-100-v2")
    run_root = base.paths.manifest.parent / "runs" / "pilot-100-v2"

    assert named.paths.manifest == run_root / "crawl_manifest.sqlite"
    assert named.paths.corpus_dir == run_root / "corpus"
    assert named.paths.temp_dir == run_root / "tmp"
    assert named.paths.debug_raw_dir == run_root / "debug_raw"
    assert named.paths.logs_dir == base.paths.logs_dir / "runs" / "pilot-100-v2"
    assert named.paths.source_file == base.paths.source_file
    assert named.paths.pilots_dir == base.paths.pilots_dir
    assert base.paths.manifest == Path("data/crawl/crawl_manifest.sqlite").resolve()


@pytest.mark.parametrize("name", ["../escape", "a/b", r"a\b", ".", "run name"])
def test_named_run_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="run name"):
        apply_run_name(load_config("config/crawler.yaml"), name)


def test_cli_accepts_global_run_name_before_command() -> None:
    args = build_parser().parse_args(
        ["--run-name", "pilot-100-v2", "crawl", "run", "--progress", "full"]
    )
    assert args.run_name == "pilot-100-v2"
    assert args.progress == "full"
