from __future__ import annotations

from crawler.adapters import DomainRequestAdapter, get_adapter, register_adapter
from crawler.adapters.ask_39 import Ask39RequestAdapter


def test_ask_39_adapter_implements_protocol() -> None:
    adapter = get_adapter("ask.39.net")
    assert adapter is not None
    assert isinstance(adapter, DomainRequestAdapter)
    assert isinstance(adapter, Ask39RequestAdapter)


def test_ask_39_adapter_rewrites_question_url() -> None:
    adapter = get_adapter("ask.39.net")
    assert adapter is not None

    url, headers = adapter.adapt_request("https://ask.39.net/question/100000370.html")
    assert url == "https://wapask.39.net/question/100000370.html"
    assert headers is not None
    assert "iPhone" in headers["User-Agent"]
    assert "zh-CN" in headers["Accept-Language"]


def test_ask_39_adapter_ignores_non_question_url() -> None:
    adapter = get_adapter("ask.39.net")
    assert adapter is not None

    url, headers = adapter.adapt_request("https://ask.39.net/about.html")
    assert url == "https://ask.39.net/about.html"
    assert headers is None


def test_ask_39_adapter_detects_bot_challenge() -> None:
    adapter = get_adapter("ask.39.net")
    assert adapter is not None

    assert adapter.is_bot_challenge("https://image.39.net/verify.html?referer=foo") is True
    assert adapter.is_bot_challenge("https://wapask.39.net/question/1.html") is False


def test_generic_domains_return_none() -> None:
    # Workers 1-5 domains have no custom adapter -> pure default generic flow
    assert get_adapter("www.cnkang.com") is None
    assert get_adapter("www.120ask.com") is None
    assert get_adapter("www.familydoctor.com.cn") is None
    assert get_adapter("example.com") is None


def test_custom_adapter_registration() -> None:
    class DummyAdapter:
        def adapt_request(self, url: str) -> tuple[str, dict[str, str] | None]:
            return url + "?custom=1", {"X-Custom": "test"}

        def is_bot_challenge(self, final_url: str | None, status: int | None = None) -> bool:
            return False

    register_adapter("custom-dummy.org", DummyAdapter())
    adapter = get_adapter("custom-dummy.org")
    assert adapter is not None
    url, headers = adapter.adapt_request("https://custom-dummy.org/api")
    assert url == "https://custom-dummy.org/api?custom=1"
    assert headers == {"X-Custom": "test"}
