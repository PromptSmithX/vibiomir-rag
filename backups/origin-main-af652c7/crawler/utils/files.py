from __future__ import annotations

import os
import time
from pathlib import Path

WINDOWS_TRANSIENT_FILE_ERRORS = {5, 32}


def replace_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 5,
    initial_delay_seconds: float = 0.05,
) -> None:
    """Atomically replace a file, tolerating short Windows scanner locks."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            retryable = os.name == "nt" and winerror in WINDOWS_TRANSIENT_FILE_ERRORS
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(initial_delay_seconds * (2**attempt))

