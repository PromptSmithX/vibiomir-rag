"""Model loader and metadata manager for Qwen3-Embedding and extensible dense models."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from retrieval.dense.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Resolved model metadata recorded into artifacts to prevent cache incompatibility."""

    model_id: str
    local_model_path: str | None
    model_revision: str
    embedding_dimension: int
    tokenizer_identity: str
    pooling_method: str
    normalization: bool
    precision: str
    query_document_formatting: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DenseModelLoader:
    """Loads embedding models and tokenizers with strict offline validation,
    GPU/CPU fallback, precision control, and metadata capture."""

    @staticmethod
    def resolve_model_path(config: ModelConfig) -> str:
        """Resolve model path giving priority to existing local directory."""
        # 1. Explicit local path
        if config.local_model_path:
            local_p = Path(config.local_model_path)
            if local_p.exists() and local_p.is_dir():
                logger.info("Using configured local model path: %s", local_p.resolve())
                return str(local_p.resolve())
            elif config.offline_mode:
                raise FileNotFoundError(
                    f"Offline mode enabled, but local model path does not exist: {local_p}"
                )

        # 2. Check standard local models folder: models/<basename>
        model_name = config.model_id.split("/")[-1]
        common_local = Path("models") / model_name
        if common_local.exists() and common_local.is_dir():
            logger.info("Discovered local model directory: %s", common_local.resolve())
            return str(common_local.resolve())

        # 3. If offline mode is enabled and no local folder found, fail fast
        if config.offline_mode:
            raise FileNotFoundError(
                f"Strict offline mode is enabled, but no local directory was found for "
                f"model '{config.model_id}'. Expected at '{common_local}' or "
                f"'{config.local_model_path}'."
            )

        # 4. Fallback to HF hub identifier
        return config.model_id

    @classmethod
    def load_tokenizer_only(cls, config: ModelConfig) -> Any:
        """Load only tokenizer for chunking phase without loading weights or touching GPU."""
        from transformers import AutoTokenizer  # type: ignore

        resolved_path = cls.resolve_model_path(config)
        logger.info(
            "Loading tokenizer from: %s (offline_mode=%s)",
            resolved_path,
            config.offline_mode,
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                local_files_only=config.offline_mode,
                trust_remote_code=True,
                padding_side="left",
            )
            return tokenizer
        except Exception as e:
            if config.offline_mode:
                raise FileNotFoundError(
                    f"Failed to load tokenizer in offline mode from '{resolved_path}'. "
                    f"Ensure model weights and tokenizer files are downloaded: {e}"
                ) from e
            raise

    @classmethod
    def load_embedding_model(cls, config: ModelConfig) -> tuple[Any, Any, ModelMetadata]:
        """Load tokenizer and model weights into device with specified precision.

        Returns:
            (tokenizer, model, ModelMetadata)
        """
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        resolved_path = cls.resolve_model_path(config)
        logger.info(
            "Loading embedding model from: %s (device=%s, precision=%s, offline=%s)",
            resolved_path,
            config.device,
            config.precision,
            config.offline_mode,
        )

        # 1. Resolve Device
        target_device = config.device.lower()
        if target_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            target_device = "cpu"
        elif target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. Resolve Precision
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(config.precision.lower(), torch.float16)
        if target_device == "cpu" and dtype == torch.float16:
            # CPU generally performs better / avoids unsupported float16 ops with float32
            logger.info("Using float32 on CPU for numerical stability.")
            dtype = torch.float32

        # 3. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path,
            local_files_only=config.offline_mode,
            trust_remote_code=True,
            padding_side="left",
        )

        # 4. Load Model
        try:
            try:
                model = AutoModel.from_pretrained(
                    resolved_path,
                    dtype=dtype,
                    local_files_only=config.offline_mode,
                    trust_remote_code=True,
                )
            except TypeError:
                model = AutoModel.from_pretrained(
                    resolved_path,
                    torch_dtype=dtype,
                    local_files_only=config.offline_mode,
                    trust_remote_code=True,
                )
            model.to(target_device)
            model.eval()
        except torch.cuda.OutOfMemoryError as oom_err:
            raise RuntimeError(
                f"CUDA Out of Memory while loading embedding model. RTX 3050 has 4 GB VRAM. "
                f"Try reducing batch size or using precision='float16' / device='cpu': {oom_err}"
            ) from oom_err
        except Exception as e:
            if config.offline_mode:
                raise FileNotFoundError(
                    f"Failed to load model weights in offline mode from '{resolved_path}': {e}"
                ) from e
            raise

        # 5. Extract metadata
        hidden_dim = getattr(
            model.config,
            "hidden_size",
            getattr(model.config, "embedding_size", 1024),
        )

        metadata = ModelMetadata(
            model_id=config.model_id,
            local_model_path=resolved_path if Path(resolved_path).exists() else None,
            model_revision=getattr(model.config, "_commit_hash", "resolved"),
            embedding_dimension=int(hidden_dim),
            tokenizer_identity=tokenizer.__class__.__name__,
            pooling_method=config.pooling_method,
            normalization=config.normalize,
            precision=config.precision,
            query_document_formatting=(
                f"query_instruction='{config.query_instruction}' | doc=plain_text"
            ),
        )

        logger.info(
            "Model successfully loaded on %s | Hidden dimension: %d | Precision: %s",
            target_device,
            hidden_dim,
            config.precision,
        )
        return tokenizer, model, metadata

    @staticmethod
    def verify_compatibility(cached_meta: dict[str, Any], current_config: ModelConfig) -> None:
        """Verify that cached embeddings or index are compatible with current config."""
        cached_model = cached_meta.get("model_id")
        if cached_model and cached_model != current_config.model_id:
            raise ValueError(
                f"Model incompatibility detected! Cached artifacts were built with '{cached_model}', "
                f"but current configuration specifies '{current_config.model_id}'. "
                f"Do not mix vectors across different models (e.g. 0.6B and 4B) in the same index."
            )

        cached_norm = cached_meta.get("normalization")
        if cached_norm is not None and cached_norm != current_config.normalize:
            raise ValueError(
                f"Normalization setting mismatch! Cached={cached_norm}, Current={current_config.normalize}"
            )
