from crawler.cleaner import clean_text, normalize_for_hash
from crawler.utils.urls import domain_from_url, fetch_key, normalize_fetch_url


def test_normalize_url_is_conservative() -> None:
    value = normalize_fetch_url("HTTPS://Example.COM:443/article?id=123&utm_source=x#section")
    assert value == "https://example.com/article?id=123&utm_source=x"
    assert domain_from_url(value) == "example.com"


def test_fetch_key_ignores_fragment_but_preserves_query() -> None:
    assert fetch_key("https://example.com/a?id=1#x") == fetch_key(
        "https://EXAMPLE.com/a?id=1#y"
    )
    assert fetch_key("https://example.com/a?id=1") != fetch_key(
        "https://example.com/a?id=2"
    )


def test_cleaner_preserves_medical_tokens_and_structure() -> None:
    raw = "# Dose\r\n\r\n  0.5 mg  \nmg/kg\n1:1000\nSpO2 95%\nNa+\nIV / IM\nACE inhibitor"
    cleaned = clean_text(raw)
    for token in ("0.5 mg", "mg/kg", "1:1000", "SpO2 95%", "Na+", "IV / IM"):
        assert token in cleaned
    assert cleaned.startswith("# Dose\n\n")
    assert normalize_for_hash(cleaned) == cleaned

