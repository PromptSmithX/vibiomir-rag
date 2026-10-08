"""Registry for high-volume domain-specific extractors added after the pilot."""

from __future__ import annotations

from collections.abc import Callable

from crawler.extractors.domains.ask_39_net import extract_ask_39
from crawler.extractors.domains.baothanhhoa_vn import extract_baothanhhoa
from crawler.extractors.domains.baoquangtri_vn import extract_baoquangtri
from crawler.extractors.domains.thanhnien_vn import extract_thanhnien

DomainExtractor = Callable[[bytes, str], tuple[str | None, str]]
EXTRACTORS: dict[str, DomainExtractor] = {}


def register(domain: str, extractor: DomainExtractor) -> None:
    EXTRACTORS[domain.lower()] = extractor


def get(domain: str) -> DomainExtractor | None:
    return EXTRACTORS.get(domain.lower())


register("ask.39.net", extract_ask_39)
register("wapask.39.net", extract_ask_39)
register("thanhnien.vn", extract_thanhnien)
register("baothanhhoa.vn", extract_baothanhhoa)
register("baoquangtri.vn", extract_baoquangtri)
