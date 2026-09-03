from __future__ import annotations

import sys
from pathlib import Path


def namespace_for(session: dict) -> str:
    profile = str(session.get("profile_name") or "default")
    source = str(session.get("source") or "desktop")
    user_id = session.get("user_id")
    principal = str(user_id) if source not in {"desktop", "cli"} and user_id else "local"
    return f"{profile}:{principal}"


def import_full_export(path: Path, *, hermes_home: Path) -> dict[str, int]:
    raise RuntimeError(
        "Selective historical extraction is not implemented; raw session exports are never indexed"
    )


def main() -> int:
    print(
        "Selective historical extraction is not implemented; raw session exports are never indexed.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
