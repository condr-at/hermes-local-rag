from __future__ import annotations

from pathlib import Path

from .service import LocalRagService


def import_session_jsonl(path: str | Path, service: LocalRagService, *, ttl_seconds: float | None = None) -> int:
    """Reject raw session exports; historical import requires semantic extraction."""
    raise ValueError(
        "Selective extraction is required; raw Hermes session exports are never indexed"
    )
