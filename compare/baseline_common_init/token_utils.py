from __future__ import annotations

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - optional dependency/fallback
    _ENCODER = None


def count_tokens(text: str) -> int:
    """Return a stable token-count estimate used by both baselines."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 3)


def length_bucket(input_tokens: int, output_tokens: int) -> str:
    """TokenScale's 3x3 short/medium/long request bucket."""
    def label(value: int, short_max: int, medium_max: int) -> str:
        if value <= short_max:
            return "S"
        if value <= medium_max:
            return "M"
        return "L"

    return f"{label(input_tokens, 256, 1024)}-{label(output_tokens, 100, 350)}"

