
from curation.base import BaseCleaningStage
from curation.models import RawDoc
from curation.rules.registry import RuleRegistry, default_registry


class DomainCleanStage(BaseCleaningStage):
    """Stage 2: Apply domain-specific cleaning rules via Strategy Pattern."""

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        self.registry = registry or default_registry

    def process(self, doc: RawDoc) -> RawDoc:
        if not doc.text:
            return doc

        rule = self.registry.get(doc.domain)
        cleaned_text = rule.clean(doc.text, doc.title)
        doc.text = cleaned_text
        return doc
