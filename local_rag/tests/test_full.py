from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR.parent))

from local_rag.chunking import chunk_text
from local_rag.config import LocalRagConfig
from local_rag.extraction import extract_candidates, summarize_session
from local_rag import LocalRagProvider
from local_rag.service import LocalRagService
from local_rag.sessions import import_session_jsonl
from local_rag.store import MemoryStore


class FakeEmbedder:
    dimensions = 3

    def embed_query(self, text: str) -> list[float]:
        return self.embed_document(text)

    def embed_document(self, text: str, *, title: str = "none") -> list[float]:
        value = text.lower()
        return [float("guitar" in value), float("python" in value), float("coffee" in value)]


def test_ttl_is_infinite_by_default_and_can_be_configured(tmp_path: Path) -> None:
    default = LocalRagConfig.load(tmp_path)
    assert default.embedding_dimensions == 512
    assert default.episodic_ttl_days is None
    assert default.summary_ttl_days is None

    (tmp_path / "local-rag").mkdir()
    (tmp_path / "local-rag" / "config.json").write_text(
        json.dumps({"episodic_ttl_days": 90, "summary_ttl_days": 730})
    )
    configured = LocalRagConfig.load(tmp_path)
    assert configured.episodic_ttl_seconds == 90 * 86400
    assert configured.summary_ttl_seconds == 730 * 86400


def test_chunker_keeps_overlap_and_never_exceeds_budget() -> None:
    text = " ".join(f"word{i}" for i in range(30))
    chunks = chunk_text(text, max_words=10, overlap_words=2)

    assert all(len(chunk.split()) <= 10 for chunk in chunks)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    assert "word29" in chunks[-1]


def test_explicit_preferences_become_review_candidates() -> None:
    candidates = extract_candidates("I prefer maple-neck guitars and concise replies.")

    assert candidates
    assert candidates[0].kind == "preference"
    assert candidates[0].confidence >= 0.9


def test_session_summary_contains_user_context_but_not_tool_noise() -> None:
    messages = [
        {"role": "user", "content": "Help me choose a guitar."},
        {"role": "tool", "content": "several kilobytes of irrelevant output"},
        {"role": "assistant", "content": "A Telecaster fits those requirements."},
    ]

    summary = summarize_session(messages)

    assert "choose a guitar" in summary
    assert "irrelevant output" not in summary


def test_expired_entries_are_not_retrieved(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite", dimensions=3)
    store.add("owner", "guitar note", [1, 0, 0], source="session:a", ttl_seconds=-1)

    assert store.search("owner", "guitar", [1, 0, 0]) == []


def test_review_candidate_can_be_promoted_to_durable_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite", dimensions=3)
    candidate_id = store.propose(
        "owner",
        "User prefers maple-neck guitars.",
        kind="preference",
        confidence=0.95,
        source="session:a",
    )

    assert store.promote("owner", candidate_id, [1, 0, 0]) is True
    result = store.search("owner", "guitar", [1, 0, 0])[0]
    assert result.kind == "durable"


def test_file_index_replaces_stale_chunks_and_blocks_env(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    note = root / "notes.md"
    note.write_text("Python project architecture " * 20)
    secret = root / ".env"
    secret.write_text("TOKEN=secret")
    service = LocalRagService(
        store=MemoryStore(tmp_path / "memory.sqlite", dimensions=3),
        embedder=FakeEmbedder(),
        namespace="owner",
        allowed_roots=[root],
    )

    first = service.index_path(note)
    note.write_text("Guitar maintenance guide " * 20)
    second = service.index_path(note)

    assert first > 0 and second > 0
    assert service.search("Python project") == []
    assert service.search("guitar")
    try:
        service.index_path(secret)
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError(".env must not be indexed")


def test_provider_lifecycle_creates_summary_and_review_candidate(tmp_path: Path) -> None:
    provider = LocalRagProvider(embedder=FakeEmbedder())
    provider.initialize(
        "session-a",
        hermes_home=str(tmp_path),
        agent_identity="default",
        user_id="owner",
        agent_context="primary",
        cwd=str(tmp_path),
    )
    provider.sync_turn("I prefer guitar notes in concise replies.", "Understood.", session_id="session-a")
    pending = json.loads(provider.handle_tool_call("local_rag_review", {}))["candidates"]
    assert pending
    assert json.loads(provider.handle_tool_call("local_rag_approve", {"id": pending[0]["id"]})) == {"promoted": True}

    provider.on_session_end([
        {"role": "user", "content": "Help with my guitar."},
        {"role": "assistant", "content": "Check the neck relief."},
    ], session_id="session-a")

    results = json.loads(provider.handle_tool_call("local_rag_search", {"query": "guitar"}))["results"]
    assert any(item["kind"] == "durable" for item in results)
    assert any(item["kind"] == "summary" for item in results)


def test_provider_file_index_tool_is_confined_to_cwd(tmp_path: Path) -> None:
    allowed = tmp_path / "project"
    allowed.mkdir()
    document = allowed / "README.md"
    document.write_text("Python guitar catalogue " * 30)
    outside = tmp_path / "outside.md"
    outside.write_text("must remain private")
    provider = LocalRagProvider(embedder=FakeEmbedder())
    provider.initialize(
        "session-a", hermes_home=str(tmp_path), agent_identity="default",
        user_id="owner", agent_context="primary", cwd=str(allowed),
    )

    indexed = json.loads(provider.handle_tool_call("local_rag_index_file", {"path": str(document)}))
    blocked = json.loads(provider.handle_tool_call("local_rag_index_file", {"path": str(outside)}))

    assert indexed["chunks"] > 0
    assert "error" in blocked


def test_redacted_session_export_can_be_backfilled(tmp_path: Path) -> None:
    export = tmp_path / "sessions.jsonl"
    export.write_text(
        json.dumps({"session_id": "abc", "message_id": 7, "role": "user", "text": "My guitar uses a maple neck."}) + "\n"
    )
    store = MemoryStore(tmp_path / "memory.sqlite", dimensions=3)
    service = LocalRagService(store=store, embedder=FakeEmbedder(), namespace="owner", allowed_roots=[tmp_path])

    imported = import_session_jsonl(export, service)

    assert imported == 1
    result = service.search("guitar")[0]
    assert result.source == "session:abc:message:7"


def test_legacy_database_is_backed_up_before_migration(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, namespace TEXT, content_hash TEXT, text TEXT, source TEXT, embedding BLOB, created_at REAL)")
    connection.commit()
    connection.close()

    MemoryStore(path, dimensions=3).close()

    assert (tmp_path / "memory.pre-v2.sqlite").exists()


def test_embedding_dimension_mismatch_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    MemoryStore(path, dimensions=3).close()

    try:
        MemoryStore(path, dimensions=4)
    except RuntimeError as exc:
        assert "dimension" in str(exc).lower()
    else:
        raise AssertionError("dimension mismatch must stop startup")


def test_provider_config_schema_and_save(tmp_path: Path) -> None:
    provider = LocalRagProvider(embedder=FakeEmbedder())
    schema = {field["key"]: field for field in provider.get_config_schema()}
    assert schema["embedding_dimensions"]["default"] == "512"
    assert schema["episodic_ttl_days"]["default"] == ""
    assert schema["summary_ttl_days"]["default"] == ""
    assert schema["visual_enabled"]["default"] is False

    provider.save_config(
        {
            "embedding_dimensions": "768",
            "episodic_ttl_days": "30",
            "summary_ttl_days": "",
            "visual_enabled": False,
        },
        str(tmp_path),
    )
    config = LocalRagConfig.load(tmp_path)
    assert config.embedding_dimensions == 768
    assert config.episodic_ttl_days == 30
    assert config.summary_ttl_days is None
    assert config.visual_enabled is False
