from __future__ import annotations

from crawler.adapters.ask_39 import Ask39RequestAdapter
from crawler.adapters.baidu_health import BaiduHealthRequestAdapter
from crawler.adapters.base import DomainRequestAdapter
from crawler.adapters.hanoimoi import HanoimoiRequestAdapter
from crawler.adapters.net39_article import Net39ArticleAdapter

from crawler.adapters.vietnamese_news import VietnameseNewsAdapter

_ADAPTERS: dict[str, DomainRequestAdapter] = {}


def register_adapter(domain: str, adapter: DomainRequestAdapter) -> None:
    _ADAPTERS[domain.lower()] = adapter


def get_adapter(domain: str) -> DomainRequestAdapter | None:
    return _ADAPTERS.get(domain.lower())


def get_all_adapters() -> dict[str, DomainRequestAdapter]:
    return dict(_ADAPTERS)


# Register default strategy adapters
_ask39 = Ask39RequestAdapter()
register_adapter("ask.39.net", _ask39)
register_adapter("wapask.39.net", _ask39)

_net39 = Net39ArticleAdapter()
for _sub in (
    "woman.39.net",
    "pf.39.net",
    "fk.39.net",
    "cancer.39.net",
    "shen.39.net",
    "gc.39.net",
    "wei.39.net",
    "gk.39.net",
):
    register_adapter(_sub, _net39)

_hanoimoi = HanoimoiRequestAdapter()
register_adapter("hanoimoi.vn", _hanoimoi)

_vietnews = VietnameseNewsAdapter()
register_adapter("vov.vn", _vietnews)
register_adapter("baolangson.vn", _vietnews)

_baidu = BaiduHealthRequestAdapter()
register_adapter("www.baidu.com", _baidu)

__all__ = ["DomainRequestAdapter", "get_adapter", "get_all_adapters", "register_adapter"]
