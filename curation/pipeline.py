
from curation.models import CuratedDoc, RawDoc
from curation.rules.registry import RuleRegistry
from curation.stages.domain_clean import DomainCleanStage
from curation.stages.markdown_fmt import MarkdownFormatStage
from curation.stages.quality_gate import QualityGateStage
from curation.stages.unicode_norm import UnicodeNormalizeStage


class CurationPipeline:
    """Pipeline Pattern: Coordinates sequential cleaning and validation stages."""

    def __init__(
        self,
        registry: RuleRegistry | None = None,
        min_chars: int = 50,
    ) -> None:
        self.stage_unicode = UnicodeNormalizeStage()
        self.stage_domain = DomainCleanStage(registry)
        self.stage_markdown = MarkdownFormatStage()
        self.stage_gate = QualityGateStage(min_chars=min_chars)

    def curate_doc(self, raw_doc: RawDoc) -> CuratedDoc:
        """Run a single document through all curation stages."""
        initial_chars = len(raw_doc.text or "")

        # Shallow copy doc or work on it
        doc = RawDoc(
            doc_id=raw_doc.doc_id,
            url=raw_doc.url,
            final_url=raw_doc.final_url,
            domain=raw_doc.domain,
            content_type=raw_doc.content_type,
            title=raw_doc.title,
            text=raw_doc.text,
            language=raw_doc.language,
            content_hash=raw_doc.content_hash,
            http_status=raw_doc.http_status,
            crawl_status=raw_doc.crawl_status,
            text_chars=raw_doc.text_chars,
            bytes_downloaded=raw_doc.bytes_downloaded,
        )

        # Stage 1: Unicode & Invisible Char Normalization
        doc = self.stage_unicode.process(doc)

        # Stage 2: Domain-Specific Boilerplate Stripping (Strategy Pattern)
        doc = self.stage_domain.process(doc)

        # Stage 3: Markdown Formatting & Paragraph Normalization
        doc = self.stage_markdown.process(doc)

        # Stage 4: Quality Gate & Metrics Recalculation
        return self.stage_gate.process(doc, raw_chars=initial_chars)

    def curate_batch(self, docs: list[RawDoc]) -> list[CuratedDoc]:
        """Run a batch of documents through the pipeline."""
        return [self.curate_doc(doc) for doc in docs]
