"""Dense embedding generator with checkpointing, streaming batches, and GPU safety."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.dense.config import ModelConfig
from retrieval.dense.model_loader import DenseModelLoader, ModelMetadata
from retrieval.models import EmbeddingStats

logger = logging.getLogger(__name__)

SCHEMA_EMBEDDING_META = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.int64()),
        ("chunk_index", pa.int32()),
        ("row_idx", pa.int64()),
    ]
)


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Extract embeddings from the last non-padded token according to Qwen embedding architecture."""
    import torch  # type: ignore

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def mean_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Mean pooling across non-padded tokens."""
    import torch  # type: ignore

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class DenseEmbedder:
    """Handles batched encoding for queries and chunks with CUDA memory safety and resuming."""

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: Any = None,
        model: Any = None,
        metadata: ModelMetadata | None = None,
    ) -> None:
        self.config = config
        self._tokenizer = tokenizer
        self._model = model
        self.metadata = metadata

    def ensure_model_loaded(self) -> None:
        """Lazy load model if not yet instantiated."""
        if self._model is None or self._tokenizer is None:
            self._tokenizer, self._model, self.metadata = DenseModelLoader.load_embedding_model(
                self.config
            )

    def encode_texts(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode a list of text strings into normalized numpy vectors."""
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        self.ensure_model_loaded()
        bs = batch_size or self.config.batch_size
        device = next(self._model.parameters()).device

        # Prepend query instruction for queries
        if is_query and self.config.query_instruction:
            formatted_texts = [f"{self.config.query_instruction}{t}" for t in texts]
        else:
            formatted_texts = texts

        all_embeddings: list[np.ndarray] = []
        total_texts = len(formatted_texts)
        start_t = time.time()
        last_log_t = start_t

        with torch.no_grad():
            for i in range(0, total_texts, bs):
                batch_slice = formatted_texts[i : i + bs]
                encoded = self._tokenizer(
                    batch_slice,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                try:
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden = outputs.last_hidden_state

                    if self.config.pooling_method == "mean":
                        pooled = mean_pool(last_hidden, attention_mask)
                    else:
                        pooled = last_token_pool(last_hidden, attention_mask)

                    if self.config.normalize:
                        pooled = F.normalize(pooled, p=2, dim=1)

                    all_embeddings.append(pooled.detach().cpu().to(torch.float32).numpy())

                except torch.cuda.OutOfMemoryError as oom:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA Out of Memory during embedding. Batch size was {bs}. "
                        f"Try reducing batch size (e.g. --batch-size 8 or 4): {oom}"
                    ) from oom

                done_count = min(i + len(batch_slice), total_texts)
                now = time.time()
                if (
                    now - last_log_t >= 10.0
                    or done_count % 1000 == 0
                    or done_count == total_texts
                ) and total_texts > 100:
                    speed = done_count / (now - start_t) if (now - start_t) > 0 else 0
                    pct = done_count / total_texts * 100
                    rem_secs = (total_texts - done_count) / speed if speed > 0 else 0
                    rem_mins = rem_secs / 60
                    logger.info(
                        "    Progress: [%d / %d] (%.1f%%) | Speed: %.1f chunks/s | ETA: %.1fm",
                        done_count,
                        total_texts,
                        pct,
                        speed,
                        rem_mins,
                    )
                    last_log_t = now

        if not all_embeddings:
            dim = self.metadata.embedding_dimension if self.metadata else 1024
            return np.empty((0, dim), dtype=np.float32)

        return np.vstack(all_embeddings)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode queries using query instructions and normalization."""
        return self.encode_texts(queries, is_query=True)

    def embed_chunks_sharded(
        self,
        chunks_dir: Path | str,
        output_dir: Path | str,
        resume: bool = True,
    ) -> tuple[Path, EmbeddingStats]:
        """Embed all chunk parquet files in chunks_dir, with checkpointing and resumability.

        Embeddings are saved to:
            <output_dir>/embeddings.npy
            <output_dir>/embeddings_meta.parquet
            <output_dir>/embedding_manifest.json
        """
        self.ensure_model_loaded()
        assert self.metadata is not None

        c_dir = Path(chunks_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        npy_path = out_dir / "embeddings.npy"
        meta_path = out_dir / "embeddings_meta.parquet"
        checkpoint_path = out_dir / "embedding_checkpoint.json"

        # Check existing checkpoint
        processed_chunk_ids: set[str] = set()
        if resume and checkpoint_path.exists() and npy_path.exists() and meta_path.exists():
            try:
                with open(checkpoint_path, encoding="utf-8") as f:
                    ckpt = json.load(f)
                    processed_chunk_ids = set(ckpt.get("processed_chunk_ids", []))
                logger.info("Resuming embedding pipeline: %d chunks already completed.", len(processed_chunk_ids))
            except Exception as e:
                logger.warning("Could not load checkpoint: %s. Re-running.", e)
                processed_chunk_ids = set()

        chunk_files = sorted(c_dir.glob("*.parquet"))
        if not chunk_files:
            raise FileNotFoundError(f"No parquet chunk files found in {c_dir}")

        start_time = time.time()
        existing_vectors: list[np.ndarray] = []
        meta_chunk_ids: list[str] = []
        meta_doc_ids: list[int] = []
        meta_indices: list[int] = []

        if processed_chunk_ids and npy_path.exists():
            old_vectors = np.load(npy_path)
            existing_vectors.append(old_vectors)
            tbl_old = pq.read_table(meta_path)
            meta_chunk_ids = tbl_old.column("chunk_id").to_pylist()
            meta_doc_ids = tbl_old.column("doc_id").to_pylist()
            meta_indices = tbl_old.column("chunk_index").to_pylist()

        new_vectors_list: list[np.ndarray] = []
        total_embedded_this_run = 0

        for cf in chunk_files:
            tbl = pq.read_table(cf)
            c_ids = tbl.column("chunk_id").to_pylist()
            d_ids = tbl.column("doc_id").to_pylist()
            c_idxs = tbl.column("chunk_index").to_pylist()
            embed_texts = tbl.column("embed_text").to_pylist()

            batch_texts: list[str] = []
            batch_meta: list[tuple[str, int, int]] = []

            for i in range(len(c_ids)):
                cid = c_ids[i]
                if cid in processed_chunk_ids:
                    continue
                batch_texts.append(embed_texts[i])
                batch_meta.append((cid, d_ids[i], c_idxs[i]))

            if not batch_texts:
                continue

            # Compute embeddings for this shard
            shard_vecs = self.encode_texts(batch_texts, is_query=False)
            new_vectors_list.append(shard_vecs)

            for cid, did, cidx in batch_meta:
                meta_chunk_ids.append(cid)
                meta_doc_ids.append(did)
                meta_indices.append(cidx)
                processed_chunk_ids.add(cid)

            total_embedded_this_run += len(batch_texts)
            elapsed_now = time.time() - start_time
            rate = total_embedded_this_run / elapsed_now if elapsed_now > 0 else 0
            logger.info(
                "Embedded shard %s (+%d chunks, total %d | %.1f chunks/s)",
                cf.name,
                len(batch_texts),
                len(processed_chunk_ids),
                rate,
            )

        # Combine all vectors and save atomically
        if new_vectors_list:
            all_vecs = (
                np.vstack(existing_vectors + new_vectors_list)
                if existing_vectors
                else np.vstack(new_vectors_list)
            )
        elif existing_vectors:
            all_vecs = np.vstack(existing_vectors)
        else:
            dim = self.metadata.embedding_dimension
            all_vecs = np.empty((0, dim), dtype=np.float32)

        # Write vectors
        tmp_npy = npy_path.with_suffix(".tmp.npy")
        np.save(tmp_npy, all_vecs)
        if npy_path.exists():
            npy_path.unlink()
        tmp_npy.rename(npy_path)

        # Write metadata table
        tbl_meta = pa.Table.from_pydict(
            {
                "chunk_id": meta_chunk_ids,
                "doc_id": meta_doc_ids,
                "chunk_index": meta_indices,
                "row_idx": list(range(len(meta_chunk_ids))),
            },
            schema=SCHEMA_EMBEDDING_META,
        )
        tmp_meta = meta_path.with_suffix(".tmp.parquet")
        pq.write_table(tbl_meta, tmp_meta, compression="zstd")
        if meta_path.exists():
            meta_path.unlink()
        tmp_meta.rename(meta_path)

        # Write checkpoint
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"processed_chunk_ids": meta_chunk_ids}, f)

        elapsed_total = time.time() - start_time
        stats = EmbeddingStats(
            total_chunks_embedded=len(all_vecs),
            embedding_dimension=int(all_vecs.shape[1]) if len(all_vecs) > 0 else 0,
            elapsed_seconds=elapsed_total,
            throughput_chunks_per_sec=(
                len(all_vecs) / elapsed_total if elapsed_total > 0 else 0.0
            ),
        )

        # Write manifest
        manifest_data = {
            "model_metadata": self.metadata.to_dict(),
            "total_vectors": len(all_vecs),
            "vector_dimension": stats.embedding_dimension,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(elapsed_total, 2),
            "throughput_chunks_per_sec": round(stats.throughput_chunks_per_sec, 1),
        }
        with open(out_dir / "embedding_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(
            "Embedding complete: %d chunks embedded (%d dim) in %.2fs (%.1f chunks/s)",
            len(all_vecs),
            stats.embedding_dimension,
            elapsed_total,
            stats.throughput_chunks_per_sec,
        )
        return out_dir, stats
