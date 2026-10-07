from pathlib import Path

from crawler.metrics import Metrics


def test_metrics_report_success_throughput_and_claim_distribution(
    tmp_path: Path,
) -> None:
    metrics = Metrics(tmp_path / "logs", interval_seconds=20, corpus_dir=tmp_path)
    metrics.record_claim("b.example")
    metrics.record_claim("a.example")
    metrics.record_claim("a.example")
    metrics.record(
        status="SUCCESS",
        domain="a.example",
        http_status=200,
        size=100,
        chars=500,
    )
    metrics.record(
        status="FAILED",
        domain="b.example",
        http_status=500,
        size=0,
        chars=0,
    )

    snapshot = metrics.snapshot({"SUCCESS": 1, "FAILED": 1})

    assert snapshot["rolling_success_documents_per_second"] > 0
    assert snapshot["rolling_success_rate_pct"] == 50.0
    assert snapshot["claimed_by_domain"] == {"a.example": 2, "b.example": 1}
