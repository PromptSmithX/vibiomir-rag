from __future__ import annotations

from crawler.adapters.ask_39 import Ask39RequestAdapter
from crawler.adapters.base import DomainRequestAdapter

_ADAPTERS: dict[str, DomainRequestAdapter] = {}


def register_adapter(domain: str, adapter: DomainRequestAdapter) -> None:
    _ADAPTERS[domain.lower()] = adapter


def get_adapter(domain: str) -> DomainRequestAdapter | None:
    return _ADAPTERS.get(domain.lower())


# Register default strategy adapters
_ask39 = Ask39RequestAdapter()
register_adapter("ask.39.net", _ask39)
register_adapter("wapask.39.net", _ask39)

__all__ = ["DomainRequestAdapter", "get_adapter", "register_adapter"]
