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
    visual_enabled: bool = False

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
            visual_enabled=_boolean(raw.get("visual_enabled", False), "visual_enabled"),
        )

    @classmethod
    def from_values(cls, values: dict[str, Any]) -> "LocalRagConfig":
        return cls(
            embedding_dimensions=_dimensions(values.get("embedding_dimensions", 512)),
            episodic_ttl_days=_ttl_value(values.get("episodic_ttl_days"), "episodic_ttl_days"),
            summary_ttl_days=_ttl_value(values.get("summary_ttl_days"), "summary_ttl_days"),
            visual_enabled=_boolean(values.get("visual_enabled", False), "visual_enabled"),
        )

    def save(self, hermes_home: str | Path) -> Path:
        directory = Path(hermes_home).expanduser() / "local-rag"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "embedding_dimensions": self.embedding_dimensions,
                    "episodic_ttl_days": self.episodic_ttl_days,
                    "summary_ttl_days": self.summary_ttl_days,
                    "visual_enabled": self.visual_enabled,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(path)
        return path

    @property
    def episodic_ttl_seconds(self) -> float | None:
        return _days_to_seconds(self.episodic_ttl_days)

    @property
    def summary_ttl_seconds(self) -> float | None:
        return _days_to_seconds(self.summary_ttl_days)


def _ttl_value(value: Any, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be blank or a positive number of days") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be null or a positive number of days")
    return float(value)


def _dimensions(value: Any) -> int:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or value not in {128, 256, 512, 768}:
        raise ValueError("embedding_dimensions must be one of 128, 256, 512, 768")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must be a boolean")


def _days_to_seconds(value: float | None) -> float | None:
    return None if value is None else value * 86400
