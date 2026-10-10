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


def test_net39_article_adapter() -> None:
    for domain, sub in (("woman.39.net", "woman"), ("pf.39.net", "pf")):
        adapter = get_adapter(domain)
        assert adapter is not None
        assert isinstance(adapter, DomainRequestAdapter)
        assert adapter.respect_robots is True
        url, headers = adapter.adapt_request(f"http://{domain}/a/080519/123.html")
        assert url == f"https://m.39.net/{sub}/a_123.html"
        assert headers is not None
        assert "iPhone" in headers["User-Agent"]
        assert adapter.is_bot_challenge("https://image.39.net/verify.html") is True
        assert adapter.is_bot_challenge(f"https://m.39.net/{sub}/a_123.html") is False


def test_hanoimoi_adapter() -> None:
    adapter = get_adapter("hanoimoi.vn")
    assert adapter is not None
    assert isinstance(adapter, DomainRequestAdapter)
    assert adapter.respect_robots is True
    url, headers = adapter.adapt_request("https://hanoimoi.vn/tin-tuc-123.html")
    assert url == "https://hanoimoi.vn/tin-tuc-123.html"
    assert headers is not None
    assert "Chrome" in headers["User-Agent"]
    assert "vi-VN" in headers["Accept-Language"]
    assert adapter.is_bot_challenge("https://hanoimoi.vn/tin-tuc-123.html", status=403) is True
    assert adapter.is_bot_challenge("https://hanoimoi.vn/tin-tuc-123.html", status=200) is False


def test_baidu_health_adapter() -> None:
    adapter = get_adapter("www.baidu.com")
    assert adapter is not None
    assert isinstance(adapter, DomainRequestAdapter)
    assert adapter.respect_robots is False
    url, headers = adapter.adapt_request("https://www.baidu.com/bh/dict/ydxx_123")
    assert url == "https://www.baidu.com/bh/dict/ydxx_123"
    assert headers is not None
    assert "Chrome" in headers["User-Agent"]
    assert adapter.is_bot_challenge("https://www.baidu.com/bh/dict/ydxx_123", status=200) is False


def test_adapters_can_recover_task() -> None:
    ask_adapter = get_adapter("ask.39.net")
    assert ask_adapter is not None
    assert ask_adapter.can_recover_task("EMPTY_CONTENT", 200, 2287, None) is True
    assert ask_adapter.can_recover_task("EMPTY_CONTENT", 200, 5000, None) is False
    assert ask_adapter.can_recover_task("SUCCESS", 200, 2287, None) is False

    net39_adapter = get_adapter("woman.39.net")
    assert net39_adapter is not None
    assert net39_adapter.can_recover_task("EMPTY_CONTENT", 200, 2287, None) is True
    assert net39_adapter.can_recover_task("NEEDS_JS", 200, 0, "BOT_CHALLENGE") is True
    assert net39_adapter.can_recover_task("FAILED", 500, 0, "SERVER_ERROR") is False

    hn_adapter = get_adapter("hanoimoi.vn")
    assert hn_adapter is not None
    assert hn_adapter.can_recover_task("FAILED", 403, 0, "HTTP_403") is True
    assert hn_adapter.can_recover_task("FAILED", 404, 0, "HTTP_404") is False

    baidu_adapter = get_adapter("www.baidu.com")
    assert baidu_adapter is not None
    assert baidu_adapter.can_recover_task("ROBOTS_DENIED", None, 0, "ROBOTS_DENIED") is True
    assert baidu_adapter.can_recover_task("FAILED", 500, 0, "NETWORK_ERROR") is False


def test_vietnamese_news_adapter() -> None:
    for domain in ("vov.vn", "baolangson.vn", "baohaiphong.vn", "www.qdnd.vn"):
        adapter = get_adapter(domain)
        assert adapter is not None
        assert adapter.respect_robots is False
        assert adapter.can_recover_task("FAILED", 403, 0, "HTTP_403") is True
        assert adapter.can_recover_task("ROBOTS_DENIED", None, 0, "ROBOTS_DENIED") is True

    # VOV gets curl headers to bypass PerimeterX TLS mismatch
    vov_adapter = get_adapter("vov.vn")
    assert vov_adapter is not None
    url, headers = vov_adapter.adapt_request("https://vov.vn/suc-khoe/123.vov")
    assert headers is not None
    assert "curl" in headers["User-Agent"]

    # BHP gets browser headers
    bhp_adapter = get_adapter("baohaiphong.vn")
    assert bhp_adapter is not None
    url, headers = bhp_adapter.adapt_request("https://baohaiphong.vn/bai-viet-123.html")
    assert headers is not None
    assert "Chrome" in headers["User-Agent"]

