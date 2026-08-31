from __future__ import annotations

import re

_CURLY_APOSTROPHE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u2032": "'",
    }
)


def normalize_for_tts(text: str) -> str:
    """Collapse whitespace and keep contractions as written."""
    cleaned = re.sub(r"\s+", " ", text.strip()).translate(_CURLY_APOSTROPHE)
    if not cleaned:
        return cleaned

    if not cleaned.endswith((".", "!", "?", "…", ",", ";", ":", "—")) and not cleaned.endswith(
        "..."
    ):
        cleaned = f"{cleaned}."

    return cleaned


def theatrical_tts_text(text: str) -> str:
    """Keep the spoken words. On a monologue, line-break after each sentence so the model stops."""
    cleaned = normalize_for_tts(text)
    if not cleaned:
        return cleaned
    stops = cleaned.count(".") + cleaned.count("!") + cleaned.count("?")
    if stops < 4:
        return cleaned
    return re.sub(r"(?<!\.)([.!?]) +", r"\1\n", cleaned)
