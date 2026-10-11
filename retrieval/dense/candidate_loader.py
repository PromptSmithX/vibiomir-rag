"""Candidate document extraction, filtering, and streaming chunker."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.dense.chunker import BaseChunker
from retrieval.dense.config import ChunkingConfig
from retrieval.models import ChunkStats, DocumentChunk
from retrieval.reader import CorpusReader

logger = logging.getLogger(__name__)

# Parquet schema for serialized chunks
SCHEMA_CHUNKS = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.int64()),
        ("chunk_index", pa.int32()),
        ("text", pa.string()),
        ("embed_text", pa.string()),
        ("title", pa.string()),
        ("heading", pa.string()),
        ("language", pa.string()),
        ("token_count", pa.int32()),
        ("chunking_strategy", pa.string()),
        ("chunking_version", pa.string()),
    ]
)


def load_candidate_doc_ids(candidate_path: Path | str) -> set[int]:
    """Load candidate doc_ids from Parquet, JSON, or text file."""
    p = Path(candidate_path)
    if not p.exists():
        raise FileNotFoundError(f"Candidate file does not exist: {p}")

    if p.suffix == ".parquet":
        tbl = pq.read_table(p, columns=["doc_id"])
        ids = set(tbl.column("doc_id").to_pylist())
    elif p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                ids = {int(x) for x in data}
            elif isinstance(data, dict) and "doc_ids" in data:
                ids = {int(x) for x in data["doc_ids"]}
            else:
                ids = {int(x) for x in data}
    else:
        # Plain text with 1 ID per line
        with open(p, encoding="utf-8") as f:
            ids = {int(line.strip()) for line in f if line.strip().isdigit()}

    logger.info("Loaded %d candidate doc_ids from %s", len(ids), p)
    return ids


class CandidateChunkingPipeline:
    """Streams cleaned corpus shards, filters to candidate documents,
    chunks them using configured chunker, and writes chunk Parquet tables."""

    def __init__(
        self,
        chunker: BaseChunker,
        output_dir: Path | str,
        rows_per_shard: int = 50_000,
    ) -> None:
        self.chunker = chunker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard

    def run(
        self,
        candidate_ids: set[int],
        corpus_dirs: list[Path | str],
        max_docs: int | None = None,
        only_success: bool = True,
    ) -> tuple[Path, ChunkStats]:
        """Stream corpus shards, chunk matching candidates, and persist chunks to Parquet.

        Returns:
            (Path, ChunkStats): Directory containing chunk shards and execution stats.
        """
        reader = CorpusReader(corpus_dirs, only_success=only_success)
        shard_paths = reader.get_shard_paths()
        logger.info(
            "Filtering %d candidate documents across %d corpus shards in %d directories...",
            len(candidate_ids),
            len(shard_paths),
            len(corpus_dirs),
        )

        start_time = time.time()
        matched_doc_ids: set[int] = set()
        all_chunks: list[DocumentChunk] = []
        shard_counter = 0

        chunks_dir = self.output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        total_chunks = 0
        total_tokens = 0

        def write_chunk_batch(chunks_to_write: list[DocumentChunk], shard_num: int) -> Path:
            if not chunks_to_write:
                return chunks_dir
            out_file = chunks_dir / f"chunks_part_{shard_num:05d}.parquet"
            data: dict[str, list[Any]] = {
                "chunk_id": [c.chunk_id for c in chunks_to_write],
                "doc_id": [c.doc_id for c in chunks_to_write],
                "chunk_index": [c.chunk_index for c in chunks_to_write],
                "text": [c.text for c in chunks_to_write],
                "embed_text": [c.embed_text for c in chunks_to_write],
                "title": [c.title or "" for c in chunks_to_write],
                "heading": [c.heading or "" for c in chunks_to_write],
                "language": [c.language or "" for c in chunks_to_write],
                "token_count": [c.token_count for c in chunks_to_write],
                "chunking_strategy": [c.chunking_strategy for c in chunks_to_write],
                "chunking_version": [c.chunking_version for c in chunks_to_write],
            }
            tbl = pa.Table.from_pydict(data, schema=SCHEMA_CHUNKS)
            pq.write_table(tbl, out_file, compression="zstd")
            logger.info("Wrote %d chunks to %s", len(chunks_to_write), out_file.name)
            return out_file

        for doc_id, title, text in reader.iter_docs():
            if doc_id not in candidate_ids:
                continue

            if doc_id in matched_doc_ids:
                # Duplicate doc_id across crawl shards, keep first
                continue

            matched_doc_ids.add(doc_id)
            doc_chunks = self.chunker.chunk_document(
                doc_id=doc_id,
                title=title,
                text=text,
            )

            all_chunks.extend(doc_chunks)
            total_chunks += len(doc_chunks)
            total_tokens += sum(c.token_count for c in doc_chunks)

            if len(all_chunks) >= self.rows_per_shard:
                write_chunk_batch(all_chunks, shard_counter)
                shard_counter += 1
                all_chunks = []

            if max_docs and len(matched_doc_ids) >= max_docs:
                logger.info("Reached max_docs limit: %d", max_docs)
                break

            if len(matched_doc_ids) % 10_000 == 0:
                logger.info(
                    "Matched %d / %d candidates | Chunks created: %d | Elapsed: %.1fs",
                    len(matched_doc_ids),
                    len(candidate_ids),
                    total_chunks,
                    time.time() - start_time,
                )

        # Flush remaining chunks
        if all_chunks:
            write_chunk_batch(all_chunks, shard_counter)
            shard_counter += 1

        elapsed = time.time() - start_time
        missing_count = len(candidate_ids) - len(matched_doc_ids)

        if missing_count > 0:
            logger.warning(
                "Missing candidates: %d / %d candidate IDs were not found in the provided corpus directories.",
                missing_count,
                len(candidate_ids),
            )
            # Save missing IDs list for inspection
            missing_ids = candidate_ids - matched_doc_ids
            missing_file = self.output_dir / "missing_candidate_ids.json"
            with open(missing_file, "w", encoding="utf-8") as f:
                json.dump(sorted(list(missing_ids)), f)
            logger.info("Saved %d missing candidate IDs to %s", len(missing_ids), missing_file)

        stats = ChunkStats(
            total_docs_processed=len(matched_doc_ids),
            total_chunks_created=total_chunks,
            total_tokens=total_tokens,
            avg_tokens_per_chunk=total_tokens / total_chunks if total_chunks > 0 else 0.0,
            elapsed_seconds=elapsed,
        )

        # Save chunk manifest metadata
        manifest = {
            "chunking_strategy": self.chunker.config.strategy,
            "chunking_version": self.chunker.config.chunking_version,
            "target_tokens": self.chunker.config.target_tokens,
            "max_total_tokens": self.chunker.config.max_total_tokens,
            "overlap_tokens": self.chunker.config.overlap_tokens,
            "total_candidates": len(candidate_ids),
            "matched_candidates": len(matched_doc_ids),
            "missing_candidates": missing_count,
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": round(stats.avg_tokens_per_chunk, 2),
            "elapsed_seconds": round(elapsed, 2),
            "shards_count": shard_counter,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        manifest_file = chunks_dir / "chunk_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Chunking completed: %d documents -> %d chunks (%.1f avg tokens/chunk) in %.2fs",
            len(matched_doc_ids),
            total_chunks,
            stats.avg_tokens_per_chunk,
            elapsed,
        )

        return chunks_dir, stats
