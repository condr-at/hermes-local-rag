from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalRagConfig:
    embedding_dimensions: int = 512
    episodic_ttl_days: float | None = None
    summary_ttl_days: float | None = None

    @classmethod
    def load(cls, hermes_home: str | Path) -> "LocalRagConfig":
        path = Path(hermes_home).expanduser() / "local-rag" / "config.json"
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            embedding_dimensions=_dimensions(raw.get("embedding_dimensions", 512)),
            episodic_ttl_days=_ttl_value(raw.get("episodic_ttl_days"), "episodic_ttl_days"),
            summary_ttl_days=_ttl_value(raw.get("summary_ttl_days"), "summary_ttl_days"),
        )

    @property
    def episodic_ttl_seconds(self) -> float | None:
        return _days_to_seconds(self.episodic_ttl_days)

    @property
    def summary_ttl_seconds(self) -> float | None:
        return _days_to_seconds(self.summary_ttl_days)


def _ttl_value(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be null or a positive number of days")
    return float(value)


def _dimensions(value: Any) -> int:
    if isinstance(value, bool) or value not in {128, 256, 512, 768}:
        raise ValueError("embedding_dimensions must be one of 128, 256, 512, 768")
    return int(value)


def _days_to_seconds(value: float | None) -> float | None:
    return None if value is None else value * 86400
