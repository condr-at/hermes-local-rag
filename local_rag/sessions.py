from __future__ import annotations

import json
from pathlib import Path

from .policy import IngestDecision, classify_text
from .service import LocalRagService


def import_session_jsonl(path: str | Path, service: LocalRagService, *, ttl_seconds: float | None = None) -> int:
    """Import a redacted `hermes sessions export --only user-prompts` JSONL file."""
    resolved = Path(path).expanduser().resolve()
    if not service.allowed_roots or not any(resolved.is_relative_to(root) for root in service.allowed_roots):
        raise ValueError("Session export is outside the allowed project root")
    if not resolved.is_file() or resolved.stat().st_size > 100_000_000:
        raise ValueError("Session export is missing or too large")
    imported = 0
    with resolved.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
            if record.get("role") != "user" or not isinstance(record.get("text"), str):
                continue
            text = record["text"].strip()
            if classify_text(text) is not IngestDecision.INDEX:
                continue
            session_id = str(record.get("session_id") or "unknown")
            message_id = str(record.get("message_id") or record.get("index") or line_number)
            source = f"session:{session_id}:message:{message_id}"
            imported += service.index_text(
                text,
                source=source,
                kind="episodic",
                ttl_seconds=ttl_seconds,
                metadata={
                    "session_id": session_id,
                    "message_id": message_id,
                    "created_at": record.get("created_at"),
                    "backfilled": True,
                },
            )
    return imported
