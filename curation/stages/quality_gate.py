import hashlib

from curation.models import CuratedDoc, RawDoc


class QualityGateStage:
    """Stage 4: Recalculate metrics, hash, and apply quality validation gates."""

    def __init__(self, min_chars: int = 50) -> None:
        self.min_chars = min_chars

    def process(self, doc: RawDoc, raw_chars: int | None = None) -> CuratedDoc:
        clean_text = (doc.text or "").strip()
        clean_chars = len(clean_text)
        original_chars = raw_chars if raw_chars is not None else (doc.text_chars or clean_chars)
        removed_chars = max(0, original_chars - clean_chars)

        # Hash of cleaned text
        content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

        if clean_chars < self.min_chars or not clean_text:
            curation_status = "LOW_QUALITY"
        else:
            curation_status = "CLEAN"

        return CuratedDoc(
            doc_id=doc.doc_id,
            url=doc.url,
            final_url=doc.final_url,
            domain=doc.domain or "",
            content_type=doc.content_type or "text/html",
            title=doc.title or "",
            text=clean_text,
            language=doc.language or "",
            content_hash=content_hash,
            http_status=doc.http_status or 200,
            crawl_status=doc.crawl_status or "SUCCESS",
            text_chars=clean_chars,
            raw_text_chars=original_chars,
            removed_chars=removed_chars,
            curation_status=curation_status,
        )
