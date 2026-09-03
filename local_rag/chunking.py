from __future__ import annotations

import re


def chunk_text(text: str, *, max_words: int = 384, overlap_words: int = 64) -> list[str]:
    """Split text into bounded overlapping chunks without dropping the tail."""
    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Expected max_words > overlap_words >= 0")
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []
    words = normalized.split()
    if len(words) <= max_words:
        return [normalized]
    chunks: list[str] = []
    step = max_words - overlap_words
    for start in range(0, len(words), step):
        chunk = words[start : start + max_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + max_words >= len(words):
            break
    return chunks
