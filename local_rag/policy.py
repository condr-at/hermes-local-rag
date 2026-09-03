from __future__ import annotations

import re
from enum import Enum


class IngestDecision(Enum):
    BLOCK = "block"
    SKIP = "skip"
    INDEX = "index"


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|github_pat|xox[baprs]|AKIA)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b"),
)

_INJECTION_PATTERNS = (
    re.compile(r"(?i)\bignore (?:all |any )?(?:previous|prior|system) instructions\b"),
    re.compile(r"(?i)\b(?:reveal|print|exfiltrate|show) (?:the )?(?:system prompt|hidden instructions|credentials|secrets)\b"),
    re.compile(r"(?i)\byou are now (?:in|a|an)\b"),
)


def classify_text(text: str) -> IngestDecision:
    cleaned = " ".join(text.split())
    if any(pattern.search(cleaned) for pattern in (*_SECRET_PATTERNS, *_INJECTION_PATTERNS)):
        return IngestDecision.BLOCK
    if len(cleaned) < 12 or len(cleaned.split()) < 3:
        return IngestDecision.SKIP
    return IngestDecision.INDEX
