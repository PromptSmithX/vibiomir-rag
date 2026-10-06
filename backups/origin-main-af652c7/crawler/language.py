from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lingua-language-detector is required; run `uv sync`") from exc
    return LanguageDetectorBuilder.from_languages(
        Language.VIETNAMESE, Language.ENGLISH, Language.CHINESE
    ).build()


def detect_language(text: str, confidence_threshold: float = 0.15) -> str:
    sample = text.strip()
    if len(sample) < 20:
        return "unknown"
    detector = _detector()
    values = detector.compute_language_confidence_values(sample)
    if not values:
        return "unknown"
    top = values[0]
    runner_up = values[1].value if len(values) > 1 else 0.0
    if top.value - runner_up < confidence_threshold:
        return "unknown"
    mapping = {"VIETNAMESE": "vi", "ENGLISH": "en", "CHINESE": "zh"}
    return mapping.get(top.language.name, "unknown")

