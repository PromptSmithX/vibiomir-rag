import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from curation.audit.lookup import ParquetBinaryLookup
from curation.audit.presenter import AuditHtmlPresenter
from curation.audit.sampler import AuditSamplePair, StratifiedSampler


def test_audit_sample_pair():
    pair = AuditSamplePair(
        doc_id=1,
        domain="vietnamnet.vn",
        url="https://vietnamnet.vn/test",
        title="Tiêu đề",
        raw_text="Nội dung thô có rác ## Tin cùng chuyên mục",
        clean_text="Nội dung thô có rác",
        raw_chars=40,
        clean_chars=19,
        removed_chars=21,
        removed_pct=52.5,
    )
    d = pair.to_dict()
    assert d["doc_id"] == 1
    assert d["removed_chars"] == 21
    assert d["removed_pct"] == 52.5


def test_stratified_sampler_and_lookup():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        clean_dir = tmp_path / "clean"
        raw_dir.mkdir()
        clean_dir.mkdir()

        # Create dummy raw shard
        pq.write_table(
            pa.table({
                "doc_id": [1, 2, 3],
                "text": ["Raw text 1 with junk", "Raw text 2 with footer", "Raw text 3"],
            }),
            raw_dir / "part-000000.parquet",
        )

        # Create dummy clean shard
        pq.write_table(
            pa.table({
                "doc_id": [1, 2, 3],
                "domain": ["vietnamnet.vn", "vietnamnet.vn", "vov2.vov.vn"],
                "url": ["https://vietnamnet.vn/1", "https://vietnamnet.vn/2", "https://vov2.vov.vn/3"],
                "title": ["Bài 1", "Bài 2", "Bài 3"],
                "text": ["Raw text 1", "Raw text 2", "Raw text 3"],
                "text_chars": [10, 10, 10],
                "raw_text_chars": [20, 22, 10],
                "removed_chars": [10, 12, 0],
                "curation_status": ["CLEAN", "CLEAN", "CLEAN"],
            }),
            clean_dir / "part-000000.parquet",
        )

        sampler = StratifiedSampler(raw_dir, clean_dir, target_total_samples=10)
        samples = sampler.sample(max_per_domain=2)
        assert len(samples) == 3
        domains = {s.domain for s in samples}
        assert "vietnamnet.vn" in domains
        assert "vov2.vov.vn" in domains

        lookup = ParquetBinaryLookup(raw_dir, clean_dir)
        found = lookup.find_by_url("https://vietnamnet.vn/1")
        assert found is not None
        assert found.doc_id == 1
        assert found.clean_text == "Raw text 1"
        assert found.raw_text == "Raw text 1 with junk"
        assert found.removed_chars == 10

        by_id = lookup.find_by_doc_id(3)
        assert by_id is not None
        assert by_id.domain == "vov2.vov.vn"

        not_found = lookup.find_by_url("https://non-existent.com")
        assert not_found is None


def test_audit_html_presenter():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_file = tmp_path / "report.html"

        samples = [
            AuditSamplePair(
                doc_id=10,
                domain="longchau.com.vn",
                url="https://longchau.com.vn/test",
                title="Vaccine",
                raw_text="Raw content with carousel",
                clean_text="Raw content",
                raw_chars=25,
                clean_chars=11,
                removed_chars=14,
                removed_pct=56.0,
            )
        ]

        presenter = AuditHtmlPresenter(samples)
        presenter.render_to_file(out_file)

        assert out_file.exists()
        html = out_file.read_text(encoding="utf-8")
        assert "ViBioMIR Curation Audit Dashboard" in html
        assert "search-input" in html
        assert "doc-list" in html
        assert "longchau.com.vn" in html
        assert "Vaccine" in html
