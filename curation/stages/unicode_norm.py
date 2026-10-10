import unicodedata

from curation.base import BaseCleaningStage
from curation.models import RawDoc


class UnicodeNormalizeStage(BaseCleaningStage):
    """Stage 1: Normalize unicode to NFC, strip zero-width chars, standardize whitespace."""

    # Set of invisible/zero-width characters to strip
    INVISIBLE_CHARS = {
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\ufeff",  # Zero-width no-break space (BOM)
        "\u2060",  # Word joiner
        "\u00ad",  # Soft hyphen
    }

    def process(self, doc: RawDoc) -> RawDoc:
        if not doc.text:
            return doc

        text = doc.text

        # 1. NFC normalization (crucial for Vietnamese composed vs decomposed accents)
        text = unicodedata.normalize("NFC", text)

        # 2. Strip invisible characters
        for char in self.INVISIBLE_CHARS:
            if char in text:
                text = text.replace(char, "")

        # 3. Standardize non-breaking space
        text = text.replace("\u00a0", " ")

        # 4. Standardize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 5. Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        doc.text = text
        if doc.title:
            title = unicodedata.normalize("NFC", doc.title)
            for char in self.INVISIBLE_CHARS:
                if char in title:
                    title = title.replace(char, "")
            doc.title = title.replace("\u00a0", " ").strip()

        return doc
