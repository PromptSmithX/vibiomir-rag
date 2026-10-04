import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from crawler.source import create_pilot, inspect_source  # noqa: E402


def make_source(path: Path, rows: int = 100) -> None:
    table = pa.table(
        {
            "id": list(range(1, rows + 1)),
            "url": [
                f"https://domain-{index % 10}.example/article/{index}"
                + (".pdf" if index % 10 == 0 else "")
                for index in range(1, rows + 1)
            ],
        }
    )
    pq.write_table(table, path)


def test_inspect_and_pilot_are_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "links.parquet"
    make_source(source)
    report = inspect_source(source, tmp_path / "analysis.json")
    assert report["total_rows"] == 100
    assert report["unique_domains"] == 10

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    create_pilot(source, first, size=30, seed=42)
    create_pilot(source, second, size=30, seed=42)
    assert first.read_bytes() == second.read_bytes()
    values = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(values) == 30
    assert len({row["doc_id"] for row in values}) == 30

