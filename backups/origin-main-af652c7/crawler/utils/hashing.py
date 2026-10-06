from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(value: str, seed: int = 0) -> int:
    raw = hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") & ((1 << 63) - 1)

