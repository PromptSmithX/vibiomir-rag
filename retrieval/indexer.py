"""BM25 Indexer using Tantivy engine with multilingual medical tokenization."""

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
import tantivy

from retrieval.models import IndexStats
from retrieval.reader import CorpusReader
from retrieval.tokenizer import MultilingualMedicalTokenizer

logger = logging.getLogger(__name__)


class BM25Indexer:
    """Manages creation, population, and maintenance of the BM25 disk index."""

    TOKENIZER_NAME = "multilingual"

    def __init__(
        self,
        index_dir: Path | str,
        tokenizer: MultilingualMedicalTokenizer | None = None,
        heap_size_mb: int = 512,
        num_threads: int = 0,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.tokenizer = tokenizer or MultilingualMedicalTokenizer()
        self.heap_size_bytes = heap_size_mb * 1024 * 1024
        self.num_threads = num_threads

    @staticmethod
    def create_schema() -> tantivy.Schema:
        """Define Tantivy schema for medical document retrieval."""
        builder = tantivy.SchemaBuilder()
        builder.add_integer_field("doc_id", stored=True, indexed=True, fast=True)
        builder.add_text_field("title", stored=False, tokenizer_name=BM25Indexer.TOKENIZER_NAME)
        builder.add_text_field("body", stored=False, tokenizer_name=BM25Indexer.TOKENIZER_NAME)
        return builder.build()

    def _setup_analyzer(self, index: tantivy.Index) -> None:
        """Register whitespace analyzer for pre-tokenized medical text."""
        analyzer = (
            tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.whitespace())
            .filter(tantivy.Filter.lowercase())
            .build()
        )
        index.register_tokenizer(self.TOKENIZER_NAME, analyzer)

    def open_or_create_index(self, recreate: bool = False) -> tantivy.Index:
        """Open existing index or create a fresh one."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        schema = self.create_schema()

        if recreate or not (self.index_dir / "meta.json").exists():
            if recreate and self.index_dir.exists():
                for item in self.index_dir.iterdir():
                    if item.is_file():
                        item.unlink()
            index = tantivy.Index(schema, path=str(self.index_dir))
        else:
            index = tantivy.Index.open(str(self.index_dir))

        self._setup_analyzer(index)
        return index

    def build_index(
        self,
        corpus_dirs: list[Path | str],
        recreate: bool = True,
        commit_every: int = 100_000,
        only_success: bool = True,
        max_docs: int | None = None,
    ) -> IndexStats:
        """Stream documents from corpus and index them into Tantivy."""
        logger.info("Initializing BM25 indexing in: %s", self.index_dir)
        index = self.open_or_create_index(recreate=recreate)
        writer = index.writer(heap_size=self.heap_size_bytes, num_threads=self.num_threads)

        reader = CorpusReader(corpus_dirs, only_success=only_success)
        start_time = time.time()
        last_log_time = start_time

        indexed_count = 0
        duplicate_count = 0
        seen_doc_ids: set[int] = set()

        stats = IndexStats(
            corpus_sources=[str(p) for p in corpus_dirs],
            created_at=datetime.now(UTC).isoformat(),
        )

        try:
            for doc_id, raw_title, raw_text in reader.iter_docs():
                if doc_id in seen_doc_ids:
                    duplicate_count += 1
                    continue
                seen_doc_ids.add(doc_id)

                # Tokenize title and text
                title_toks = self.tokenizer.tokenize_to_str(raw_title)
                body_toks = self.tokenizer.tokenize_to_str(raw_text)

                doc = tantivy.Document(
                    doc_id=doc_id,
                    title=title_toks,
                    body=body_toks,
                )
                writer.add_document(doc)
                indexed_count += 1

                if max_docs and indexed_count >= max_docs:
                    logger.info("Reached max_docs limit: %d", max_docs)
                    break

                if indexed_count % 50_000 == 0:
                    now = time.time()
                    elapsed = now - start_time
                    step_speed = 50_000 / (now - last_log_time)
                    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    logger.info(
                        "Indexed %d docs (%.0f docs/s) | RAM: %.1f MB | Elapsed: %.1fs",
                        indexed_count,
                        step_speed,
                        rss_mb,
                        elapsed,
                    )
                    last_log_time = now

                if commit_every and indexed_count % commit_every == 0:
                    logger.info("Committing segment to disk at %d docs...", indexed_count)
                    writer.commit()

            # Final commit
            logger.info("Performing final commit for %d total documents...", indexed_count)
            writer.commit()

        finally:
            del writer

        total_time = time.time() - start_time
        index_size = sum(f.stat().st_size for f in self.index_dir.rglob("*") if f.is_file())

        stats.total_docs = indexed_count
        stats.total_shards = len(reader.get_shard_paths())
        stats.index_size_bytes = index_size
        stats.build_time_seconds = total_time

        # Save metadata JSON
        meta_file = self.index_dir / "retrieval_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "total_docs": stats.total_docs,
                    "total_shards": stats.total_shards,
                    "duplicates_skipped": duplicate_count,
                    "index_size_mb": round(index_size / 1024 / 1024, 2),
                    "build_time_seconds": round(total_time, 2),
                    "docs_per_second": round(indexed_count / total_time, 1)
                    if total_time > 0
                    else 0,
                    "corpus_sources": stats.corpus_sources,
                    "created_at": stats.created_at,
                },
                fp,
                indent=2,
            )

        logger.info(
            "Index complete: %d docs in %.2fs (%.0f docs/s). Size: %.1f MB. Duplicates: %d",
            indexed_count,
            total_time,
            indexed_count / total_time if total_time > 0 else 0,
            index_size / 1024 / 1024,
            duplicate_count,
        )

        return stats
