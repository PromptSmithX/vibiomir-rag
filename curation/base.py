from abc import ABC, abstractmethod

from curation.models import RawDoc


class BaseCleaningStage(ABC):
    """Abstract interface for a stage in the curation pipeline."""

    @abstractmethod
    def process(self, doc: RawDoc) -> RawDoc:
        """Process the document and return the modified doc."""
        pass


class BaseDomainRule(ABC):
    """Abstract Strategy interface for domain-specific text cleaning."""

    @abstractmethod
    def clean(self, text: str, title: str | None = None) -> str:
        """Clean domain-specific boilerplate, headers, and footers from article text."""
        pass
