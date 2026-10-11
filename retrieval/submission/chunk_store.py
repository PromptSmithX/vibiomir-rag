"""Chunk store for retrieving and verifying clean source chunk text."""

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
        self._doc_to_first_chunk: dict[int, tuple[str, str]] = {}
        self._loaded = False

    def preload_for_retrieval(
        self,
        needed_chunk_ids: set[str],
        needed_doc_ids: set[int],
    ) -> int:
        """Stream through chunk parquet shards to load source chunk texts
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
        clean_needed_dids = {d for d in needed_doc_ids if d is not None}

        # Keep track of best (lowest) chunk index per doc for fallback
        doc_lowest_index: dict[int, int] = {}

        loaded_count = 0
        for shard in shard_files:
            try:
                # Read only required columns to minimize RAM and I/O
                tbl = pq.read_table(
                    shard,
                    columns=["chunk_id", "doc_id", "chunk_index", "text"],
                )
                c_ids = tbl.column("chunk_id").to_pylist()
                d_ids = tbl.column("doc_id").to_pylist()
                c_idxs = tbl.column("chunk_index").to_pylist()
                texts = tbl.column("text").to_pylist()

                for i in range(len(c_ids)):
                    cid = c_ids[i]
                    did = d_ids[i]
                    cidx = c_idxs[i]
                    raw_text = texts[i] or ""

                    # Match specific chunk ID
                    if cid in clean_needed_cids:
                        self._chunk_text_cache[cid] = raw_text
                        loaded_count += 1

                    # Match doc ID for first-chunk fallback
                    if did in clean_needed_dids:
                        prev_best_idx = doc_lowest_index.get(did, 999999)
                        if cidx < prev_best_idx:
                            doc_lowest_index[did] = cidx
                            self._doc_to_first_chunk[did] = (cid, raw_text)

            except Exception as e:
                logger.error("Failed to read chunk shard %s: %s", shard, e)

        self._loaded = True
        logger.info(
            "ChunkStore loaded %d specific chunks and %d document fallback chunks from %d shards",
            len(self._chunk_text_cache),
            len(self._doc_to_first_chunk),
            len(shard_files),
        )
        return len(self._chunk_text_cache)

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """Get source text for a specific chunk ID."""
        if not chunk_id:
            return None
        return self._chunk_text_cache.get(chunk_id)

    def get_fallback_chunk_for_doc(self, doc_id: int) -> tuple[str, str] | None:
        """Get (chunk_id, source_text) of the first available chunk for a doc_id."""
        return self._doc_to_first_chunk.get(doc_id)

    def get_text_from_corpus(self, doc_id: int) -> str | None:
        """Fallback to retrieve raw text directly from corpus parquet if document
        was not in candidate chunks.
        """
        if not self.corpus_path or not self.corpus_path.exists():
            return None

        try:
            tbl = pq.read_table(self.corpus_path, columns=["doc_id", "text"])
            dids = tbl.column("doc_id").to_pylist()
            texts = tbl.column("text").to_pylist()
            for did, t in zip(dids, texts, strict=False):
                if did == doc_id:
                    return str(t or "").strip()
        except Exception as e:
            logger.warning("Could not read doc %d from corpus %s: %s", doc_id, self.corpus_path, e)
        return None
