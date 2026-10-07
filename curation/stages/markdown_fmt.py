import re

from curation.base import BaseCleaningStage
from curation.models import RawDoc


class MarkdownFormatStage(BaseCleaningStage):
    """Stage 3: Format and normalize markdown spacing and paragraph structures."""

    # Regex to collapse 3 or more consecutive newlines into 2 (one blank line)
    EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")

    def process(self, doc: RawDoc) -> RawDoc:
        if not doc.text:
            return doc

        text = doc.text.strip()

        # Collapse excessive blank lines
        text = self.EXCESSIVE_NEWLINES.sub("\n\n", text)

        doc.text = text
        return doc
