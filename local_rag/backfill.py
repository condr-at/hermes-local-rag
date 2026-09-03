from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from .config import LocalRagConfig
from .embedder import get_shared_embedder
from .extraction import extract_candidates, summarize_session
from .policy import IngestDecision, classify_text
from .service import LocalRagService
from .store import MemoryStore


def namespace_for(session: dict) -> str:
    profile = str(session.get("profile_name") or "default")
    source = str(session.get("source") or "desktop")
    user_id = session.get("user_id")
    principal = str(user_id) if source not in {"desktop", "cli"} and user_id else "local"
    return f"{profile}:{principal}"


def import_full_export(path: Path, *, hermes_home: Path) -> dict[str, int]:
    config = LocalRagConfig.load(hermes_home)
    embedder = get_shared_embedder(config.embedding_dimensions)
    store = MemoryStore(hermes_home / "local-rag" / "memory.sqlite", dimensions=embedder.dimensions)
    counts: dict[str, int] = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                session = json.loads(line)
                namespace = namespace_for(session)
                session_id = str(session.get("id") or session.get("session_id") or "unknown")
                service = LocalRagService(store=store, embedder=embedder, namespace=namespace)
                messages = session.get("messages") or []
                for message in messages:
                    if message.get("role") != "user" or not isinstance(message.get("content"), str):
                        continue
                    text = message["content"].strip()
                    if classify_text(text) is not IngestDecision.INDEX:
                        continue
                    message_id = str(message.get("id") or "unknown")
                    counts[namespace] = counts.get(namespace, 0) + service.index_text(
                        text,
                        source=f"session:{session_id}:message:{message_id}",
                        kind="episodic",
                        ttl_seconds=config.episodic_ttl_seconds,
                        metadata={"session_id": session_id, "message_id": message_id, "backfilled": True},
                    )
                    for candidate in extract_candidates(text):
                        store.propose(namespace, candidate.text, kind=candidate.kind, confidence=candidate.confidence, source=f"session:{session_id}:message:{message_id}")
                summary = summarize_session(messages)
                if classify_text(summary) is IngestDecision.INDEX:
                    counts[namespace] = counts.get(namespace, 0) + service.index_text(
                        summary,
                        source=f"summary:{session_id}:backfill",
                        kind="summary",
                        ttl_seconds=config.summary_ttl_seconds,
                        metadata={"session_id": session_id, "backfilled": True},
                    )
        return counts
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill local RAG from an official redacted Hermes session export")
    parser.add_argument("--home", default=str(Path.home() / ".hermes"))
    parser.add_argument("--export", help="Use an existing redacted full JSONL export instead of creating one")
    args = parser.parse_args()
    home = Path(args.home).expanduser().resolve()
    if args.export:
        counts = import_full_export(Path(args.export).expanduser().resolve(), hermes_home=home)
    else:
        with tempfile.TemporaryDirectory(prefix="hermes-rag-") as temporary:
            export = Path(temporary) / "sessions.jsonl"
            subprocess.run(
                ["hermes", "sessions", "export", str(export), "--format", "jsonl", "--redact", "--yes"],
                check=True,
            )
            counts = import_full_export(export, hermes_home=home)
    print(json.dumps({"imported": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
