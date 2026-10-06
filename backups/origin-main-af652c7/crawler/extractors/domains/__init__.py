"""Registry for high-volume domain-specific extractors added after the pilot."""

from __future__ import annotations

from collections.abc import Callable

DomainExtractor = Callable[[bytes, str], tuple[str | None, str]]
EXTRACTORS: dict[str, DomainExtractor] = {}


def register(domain: str, extractor: DomainExtractor) -> None:
    EXTRACTORS[domain.lower()] = extractor


def get(domain: str) -> DomainExtractor | None:
    return EXTRACTORS.get(domain.lower())
