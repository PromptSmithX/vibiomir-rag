"""Chunk store for retrieving, indexing, and verifying clean source chunk text."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ChunkStore:
    """Manages fast lookup of original, un-prefixed source chunk texts
    from candidate chunk parquet shards and optional fallback corpus.
    """

    def __init__(
        self,
        chunks_dir: Path | str | None = None,
        corpus_path: Path | str | None = None,
    ) -> None:
        self.chunks_dir = Path(chunks_dir) if chunks_dir else None
        self.corpus_path = Path(corpus_path) if corpus_path else None
        self._chunk_text_cache: dict[str, str] = {}
        self._chunk_meta_cache: dict[str, dict[str, Any]] = {}
        self._doc_chunks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._doc_to_first_chunk: dict[int, tuple[str, str]] = {}
        self._corpus_text_cache: dict[int, str] = {}
        self._missing_corpus_dids: set[int] = set()
        self._loaded = False

    def preload_for_retrieval(
        self,
        needed_chunk_ids: set[str],
        needed_doc_ids: set[int],
    ) -> int:
        """Stream through chunk parquet shards to load source chunk texts and metadata
        for specified chunk_ids and doc_ids.

        Reads strictly the clean 'text' column, preserving source integrity
        and avoiding any 'embed_text' title/heading prefixes.

        Returns:
            int: Number of unique chunks loaded into memory.
        """
        if not self.chunks_dir or not self.chunks_dir.exists():
            logger.warning("Chunks directory does not exist: %s", self.chunks_dir)
            return 0

        shard_files = sorted(self.chunks_dir.glob("*.parquet"))
        if not shard_files:
            logger.warning("No parquet chunk shards found in %s", self.chunks_dir)
            return 0

        clean_needed_cids = {c for c in needed_chunk_ids if c}
        clean_needed_dids = {int(d) for d in needed_doc_ids if d is not None}

        # Keep track of best (lowest) chunk index per doc for fallback
        doc_lowest_index: dict[int, int] = {}
        loaded_count = 0

        for shard in shard_files:
            try:
                # Read only required columns to minimize RAM and I/O
                available_cols = set(pq.read_schema(shard).names)
                read_cols = ["chunk_id", "doc_id", "chunk_index", "text"]
                if "title" in available_cols:
                    read_cols.append("title")
                if "heading" in available_cols:
                    read_cols.append("heading")

                tbl = pq.read_table(shard, columns=read_cols)
                c_ids = tbl.column("chunk_id").to_pylist()
                d_ids = tbl.column("doc_id").to_pylist()
                c_idxs = tbl.column("chunk_index").to_pylist()
                texts = tbl.column("text").to_pylist()
                titles = tbl.column("title").to_pylist() if "title" in available_cols else [None] * len(c_ids)
                headings = tbl.column("heading").to_pylist() if "heading" in available_cols else [None] * len(c_ids)

                for i in range(len(c_ids)):
                    cid = str(c_ids[i])
                    did = int(d_ids[i])
                    cidx = int(c_idxs[i])
                    raw_text = str(texts[i] or "")

                    match_cid = cid in clean_needed_cids
                    match_did = did in clean_needed_dids

                    if match_cid or match_did:
                        chunk_record = {
                            "chunk_id": cid,
                            "doc_id": did,
                            "chunk_index": cidx,
                            "text": raw_text,
                            "title": titles[i],
                            "heading": headings[i],
                        }
                        self._chunk_text_cache[cid] = raw_text
                        self._chunk_meta_cache[cid] = chunk_record
                        self._doc_chunks[did].append(chunk_record)
                        loaded_count += 1

                        prev_best_idx = doc_lowest_index.get(did, 999999)
                        if cidx < prev_best_idx:
                            doc_lowest_index[did] = cidx
                            self._doc_to_first_chunk[did] = (cid, raw_text)

            except Exception as e:
                logger.error("Failed to read chunk shard %s: %s", shard, e)

        # Sort chunks within each doc by chunk_index
        for did in self._doc_chunks:
            self._doc_chunks[did].sort(key=lambda x: x["chunk_index"])

        self._loaded = True
        logger.info(
            "ChunkStore loaded %d chunks across %d unique documents from %d shards",
            len(self._chunk_text_cache),
            len(self._doc_chunks),
            len(shard_files),
        )
        return len(self._chunk_text_cache)

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """Get source text for a specific chunk ID."""
        if not chunk_id:
            return None
        return self._chunk_text_cache.get(chunk_id)

    def get_chunk_meta(self, chunk_id: str) -> dict[str, Any] | None:
        """Get full metadata dict for a specific chunk ID."""
        if not chunk_id:
            return None
        return self._chunk_meta_cache.get(chunk_id)

    def get_chunks_for_doc(self, doc_id: int) -> list[dict[str, Any]]:
        """Get all loaded chunks for a document, ordered by chunk_index."""
        return self._doc_chunks.get(int(doc_id), [])

    def get_fallback_chunk_for_doc(self, doc_id: int) -> tuple[str, str] | None:
        """Get (chunk_id, source_text) of the first available chunk for a doc_id."""
        return self._doc_to_first_chunk.get(int(doc_id))

    def set_corpus_text(self, doc_id: int, text: str) -> None:
        """Manually inject or cache full document text for a doc_id."""
        self._corpus_text_cache[int(doc_id)] = text.strip()

    def discover_corpus_files(self) -> list[Path]:
        """Find all parquet files containing source document text."""
        files: list[Path] = []
        if self.corpus_path and self.corpus_path.exists():
            if self.corpus_path.is_file():
                files.append(self.corpus_path)
            elif self.corpus_path.is_dir():
                files.extend(sorted(self.corpus_path.glob("*.parquet")))

        if not files:
            # Check default repository locations
            clean_runs = Path("data/clean/runs")
            if clean_runs.exists():
                for d in sorted(clean_runs.iterdir()):
                    c_dir = d / "corpus"
                    if c_dir.exists():
                        files.extend(sorted(c_dir.glob("*.parquet")))

            if not files:
                for candidate_p in [
                    Path("data/crawl/runs/worker-0-local/corpus"),
                    Path("data/source/corpus.parquet"),
                    Path("data/corpus"),
                ]:
                    if candidate_p.is_file():
                        files.append(candidate_p)
                    elif candidate_p.is_dir():
                        files.extend(sorted(candidate_p.glob("*.parquet")))

        return files

    def preload_corpus(self, needed_doc_ids: set[int]) -> int:
        """Preload full source text from corpus parquet shards for needed documents.

        Optimizes retrieval-time passage expansion by batching I/O and caching in memory.
        """
        remaining_dids = {
            int(d)
            for d in needed_doc_ids
            if int(d) not in self._corpus_text_cache and int(d) not in self._missing_corpus_dids
        }
        if not remaining_dids:
            return len(self._corpus_text_cache)

        corpus_files = self.discover_corpus_files()
        if not corpus_files:
            self._missing_corpus_dids.update(remaining_dids)
            return len(self._corpus_text_cache)

        for shard in corpus_files:
            if not remaining_dids:
                break
            try:
                schema_cols = set(pq.read_schema(shard).names)
                if "doc_id" not in schema_cols or "text" not in schema_cols:
                    continue

                tbl = pq.read_table(shard, columns=["doc_id", "text"])
                dids = tbl.column("doc_id").to_pylist()
                texts = tbl.column("text").to_pylist()

                for did, t in zip(dids, texts, strict=False):
                    int_did = int(did)
                    if int_did in remaining_dids:
                        self._corpus_text_cache[int_did] = str(t or "").strip()
                        remaining_dids.remove(int_did)
            except Exception as e:
                logger.warning("Error reading corpus shard %s: %s", shard, e)

        # Mark any remaining unfound document IDs as missing so they are never scanned again
        self._missing_corpus_dids.update(remaining_dids)
        return len(self._corpus_text_cache)

    def get_text_from_corpus(self, doc_id: int) -> str | None:
        """Retrieve raw text directly from corpus parquet if document
        is not already in cache.
        """
        int_did = int(doc_id)
        if int_did in self._corpus_text_cache:
            return self._corpus_text_cache[int_did]
        if int_did in self._missing_corpus_dids:
            return None

        corpus_files = self.discover_corpus_files()
        if not corpus_files:
            self._missing_corpus_dids.add(int_did)
            return None

        for shard in corpus_files:
            try:
                schema_cols = set(pq.read_schema(shard).names)
                if "doc_id" not in schema_cols or "text" not in schema_cols:
                    continue

                tbl = pq.read_table(shard, columns=["doc_id", "text"])
                dids = tbl.column("doc_id").to_pylist()
                texts = tbl.column("text").to_pylist()
                for did, t in zip(dids, texts, strict=False):
                    d_int = int(did)
                    if d_int == int_did:
                        res = str(t or "").strip()
                        self._corpus_text_cache[int_did] = res
                        return res
            except Exception as e:
                logger.warning("Could not read doc %d from corpus %s: %s", int_did, shard, e)

        self._missing_corpus_dids.add(int_did)
        return None
