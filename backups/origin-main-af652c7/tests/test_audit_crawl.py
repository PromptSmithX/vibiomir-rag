import json

import pyarrow as pa
import pyarrow.parquet as pq

from audit_crawl import allocate_sqrt_quotas, audit_corpus, score_document


def _row(doc_id: int, domain: str, text: str, *, status: str = "SUCCESS") -> dict:
    return {
        "doc_id": doc_id,
        "url": f"https://{domain}/{doc_id}",
        "final_url": f"https://{domain}/{doc_id}",
        "domain": domain,
        "content_type": "text/html",
        "title": f"Medical title {doc_id}",
        "text": text,
        "language": "ENGLISH",
        "content_hash": f"hash-{doc_id}",
        "http_status": 200,
        "crawl_status": status,
        "text_chars": len(text),
        "bytes_downloaded": max(10_000, len(text) * 2),
    }


def test_risk_score_flags_short_low_ratio_duplicate_and_boilerplate() -> None:
    row = _row(1, "example.com", "Related articles\nShare this article")
    row["bytes_downloaded"] = 100_000
    score, reasons, flags = score_document(row, duplicate_count=3)
    codes = {reason["code"] for reason in reasons}
    assert score >= 70
    assert {"very_short_text", "very_low_text_ratio", "duplicate_content"} <= codes
    assert flags["boilerplate"] == 1


def test_sqrt_allocation_respects_capacity_and_total() -> None:
    quotas = allocate_sqrt_quotas(
        {"large": 100, "medium": 25, "small": 1},
        {"large": 100, "medium": 25, "small": 1},
        20,
    )
    assert sum(quotas.values()) == 20
    assert quotas["large"] > quotas["medium"] > quotas["small"]
    assert quotas["small"] == 1


def test_audit_outputs_three_disjoint_deterministic_samples(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = []
    for doc_id in range(360):
        domain = f"domain-{doc_id % 12}.example"
        text = f"Medical title {doc_id}\n\n" + ("Useful medical paragraph. " * (8 + doc_id % 5))
        rows.append(_row(doc_id, domain, text))
    rows.append(_row(9999, "ignored.example", "ignored", status="FAILED"))
    pq.write_table(pa.Table.from_pylist(rows), corpus / "part-000000.parquet")

    output = tmp_path / "qa"
    first = audit_corpus(corpus, output, seed=42)
    first_jsonl = (output / "qa_samples.jsonl").read_bytes()
    second = audit_corpus(corpus, output, seed=42)

    samples = [json.loads(line) for line in first_jsonl.decode("utf-8").splitlines()]
    groups = {
        group: [row for row in samples if row["sample_group"] == group]
        for group in ("high_risk", "random", "stratified")
    }
    assert first == second
    assert len(samples) == len({row["doc_id"] for row in samples}) == 300
    assert all(len(group) == 100 for group in groups.values())
    assert max(
        sum(row["domain"] == domain for row in groups["high_risk"])
        for domain in {row["domain"] for row in groups["high_risk"]}
    ) <= 10
    assert first_jsonl == (output / "qa_samples.jsonl").read_bytes()
    review_html = (output / "qa_review.html").read_text(encoding="utf-8")
    assert "exportReviewButton.onclick" in review_html
    assert "export.onclick" not in review_html
    assert ".join('\\n')+'\\n'" in review_html
    assert (output / "domain_summary.csv").read_text(encoding="utf-8-sig").startswith("domain,")
