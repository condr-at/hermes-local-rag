from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    text: str
    kind: str
    confidence: float


_PATTERNS = (
    ("preference", 0.95, re.compile(r"(?i)\b(?:i prefer|i like|my preference is|я предпочитаю|мне нравится|мне нравятся)\b[^.!?\n]{3,240}")),
    ("environment", 0.92, re.compile(r"(?i)\b(?:the project uses|we use|my setup uses|проект использует|мы используем|у меня установлен[аоы]?)\b[^.!?\n]{3,240}")),
    ("decision", 0.98, re.compile(r"(?i)\b(?:remember that|please remember|запомни|сохрани в память)\b[^.!?\n]{3,240}")),
)


def extract_candidates(text: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for kind, confidence, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip(" ,;:-")
            if value:
                candidates.append(Candidate(value, kind, confidence))
    return candidates[:5]


def summarize_session(messages: list[dict[str, Any]], *, max_chars: int = 2400) -> str:
    """Create a deterministic, provenance-safe extractive summary."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        clean = re.sub(r"\s+", " ", content).strip()
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {clean[:700]}")
        if sum(len(line) for line in lines) >= max_chars:
            break
    return "\n".join(lines)[:max_chars]
