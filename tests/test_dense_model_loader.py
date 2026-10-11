"""Unit tests for DenseModelLoader and model configuration management.

Tests offline validation, local path discovery, metadata capture, and compatibility checks.
"""

from pathlib import Path
import pytest

from retrieval.dense.config import ModelConfig
from retrieval.dense.model_loader import DenseModelLoader


def test_resolve_model_path_local_priority(tmp_path: Path):
    """When a local model directory exists, resolve_model_path must prioritize it."""
    local_dir = tmp_path / "my_local_qwen"
    local_dir.mkdir()

    cfg = ModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        local_model_path=str(local_dir),
        offline_mode=False,
    )
    resolved = DenseModelLoader.resolve_model_path(cfg)
    assert resolved == str(local_dir.resolve())


def test_strict_offline_mode_missing_model_raises_error(tmp_path: Path):
    """Strict offline mode must raise FileNotFoundError if the local model folder is missing."""
    non_existent = tmp_path / "does_not_exist_model"

    cfg = ModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        local_model_path=str(non_existent),
        offline_mode=True,
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        DenseModelLoader.resolve_model_path(cfg)

    assert "Strict offline mode" in str(exc_info.value) or "does not exist" in str(exc_info.value)


def test_offline_mode_without_local_path_raises_error():
    """Offline mode with no local directory discovered must raise FileNotFoundError."""
    cfg = ModelConfig(
        model_id="SomeOrg/NonExistentModel",
        local_model_path=None,
        offline_mode=True,
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        DenseModelLoader.resolve_model_path(cfg)

    assert "Strict offline mode is enabled" in str(exc_info.value)


def test_verify_compatibility_mismatched_model():
    """Embedding cache from a different model must trigger a compatibility error."""
    cached_meta = {
        "model_id": "Qwen/Qwen3-Embedding-4B",
        "normalization": True,
    }
    current_cfg = ModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        normalize=True,
    )

    with pytest.raises(ValueError) as exc_info:
        DenseModelLoader.verify_compatibility(cached_meta, current_cfg)

    assert "Model incompatibility detected" in str(exc_info.value)


def test_verify_compatibility_mismatched_normalization():
    """Normalization flag mismatch must trigger an error."""
    cached_meta = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "normalization": False,
    }
    current_cfg = ModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        normalize=True,
    )

    with pytest.raises(ValueError) as exc_info:
        DenseModelLoader.verify_compatibility(cached_meta, current_cfg)

    assert "Normalization setting mismatch" in str(exc_info.value)


def test_verify_compatibility_matching_configs():
    """Matching configurations must pass verification silently."""
    cached_meta = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "normalization": True,
    }
    current_cfg = ModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        normalize=True,
    )
    # Should not raise any error
    DenseModelLoader.verify_compatibility(cached_meta, current_cfg)
