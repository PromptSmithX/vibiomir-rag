import json

from crawler.report import summarize_review


def test_review_summary_groups_noise_failures_by_domain(tmp_path) -> None:
    review = tmp_path / "review.jsonl"
    rows = [
        {
            "domain": "clean.example",
            "crawl_status": "SUCCESS",
            "review_title_correct": True,
            "review_body_correct": True,
            "review_structure_ok": True,
            "review_noise_ok": True,
            "review_medical_values_ok": True,
        },
        {
            "domain": "noisy.example",
            "crawl_status": "SUCCESS",
            "review_title_correct": True,
            "review_body_correct": True,
            "review_structure_ok": True,
            "review_noise_ok": False,
            "review_medical_values_ok": True,
        },
    ]
    review.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = summarize_review(review)

    assert summary["reviewed_success"] == 2
    assert summary["acceptable_success_rate"] == 0.5
    assert summary["criterion_failures"]["review_noise_ok"] == 1
    assert summary["by_domain"][0] == {
        "domain": "noisy.example",
        "reviewed_success": 1,
        "unacceptable_success": 1,
        "noise_failures": 1,
    }
