from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SearchResult:
    id: int
    text: str
    source: str
    score: float
    kind: str = "episodic"
    confidence: float = 1.0


class MemoryStore:
    def __init__(self, path: Path, *, dimensions: int) -> None:
        self.path = Path(path)
        self.dimensions = dimensions
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_before_migration()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _backup_before_migration(self) -> None:
        if not self.path.exists() or not self.path.stat().st_size:
            return
        source = sqlite3.connect(self.path)
        try:
            table = source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
            columns = {row[1] for row in source.execute("PRAGMA table_info(memories)")} if table else set()
            backup_path = self.path.with_name("memory.pre-v2.db")
            if table and "kind" not in columns and not backup_path.exists():
                target = sqlite3.connect(backup_path)
                try:
                    source.backup(target)
                finally:
                    target.close()
        finally:
            source.close()

    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(namespace, content_hash)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    text, content='memories', content_rowid='id', tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
                END;
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    UNIQUE(namespace, text, status)
                );
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)")}
            additions = {
                "kind": "TEXT NOT NULL DEFAULT 'episodic'",
                "confidence": "REAL NOT NULL DEFAULT 1.0",
                "importance": "REAL NOT NULL DEFAULT 0.5",
                "expires_at": "REAL",
                "project": "TEXT NOT NULL DEFAULT ''",
                "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
                "accessed_at": "REAL",
                "access_count": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in additions.items():
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {definition}")
            dimension_row = self._conn.execute("SELECT value FROM meta WHERE key='embedding_dimensions'").fetchone()
            if dimension_row and int(dimension_row[0]) != self.dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch: database={dimension_row[0]}, configured={self.dimensions}. Reindex explicitly."
                )
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('embedding_dimensions', ?)", (str(self.dimensions),))
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '2')")

    def add(
        self,
        namespace: str,
        text: str,
        embedding: Iterable[float],
        *,
        source: str,
        kind: str = "episodic",
        confidence: float = 1.0,
        importance: float = 0.5,
        ttl_seconds: float | None = None,
        project: str = "",
        metadata: dict | None = None,
        dedupe_key: str | None = None,
    ) -> bool:
        vector = list(embedding)
        if len(vector) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(vector)}")
        cleaned = text.strip()
        digest_input = dedupe_key if dedupe_key is not None else f"{source}\0{cleaned}"
        digest = hashlib.sha256(digest_input.encode()).hexdigest()
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO memories(namespace, content_hash, text, source, embedding, created_at, "
                    "kind, confidence, importance, expires_at, project, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (namespace, digest, cleaned, source, array("f", vector).tobytes(), time.time(),
                     kind, confidence, importance, expires_at, project, json.dumps(metadata or {})),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def replace_source(self, namespace: str, source: str, records: list[dict]) -> int:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM memories WHERE namespace=? AND source=?", (namespace, source))
        inserted = 0
        for record in records:
            inserted += int(self.add(namespace, source=source, **record))
        return inserted

    def search(
        self,
        namespace: str,
        query: str,
        query_embedding: Iterable[float],
        *,
        limit: int = 5,
        project: str | None = None,
        session_id: str | None = None,
    ) -> list[SearchResult]:
        qvec = list(query_embedding)
        if len(qvec) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(qvec)}")
        now = time.time()
        with self._lock:
            visibility = ""
            lexical_visibility = ""
            visibility_args: tuple[str, ...] = ()
            if project is not None and session_id is not None:
                def visibility_clause(prefix: str) -> str:
                    metadata = f"{prefix}metadata_json"
                    project_column = f"{prefix}project"
                    return (
                    " AND ("
                    f"(json_extract({metadata},'$.scope') IS NULL AND ({project_column}='' OR {project_column}=?)) OR "
                    f"json_extract({metadata},'$.scope')='global' OR "
                    f"(json_extract({metadata},'$.scope')='project' AND {project_column}=?) OR "
                    f"(json_extract({metadata},'$.scope')='session' AND json_extract({metadata},'$.session_id')=?)"
                    ")"
                    )
                visibility = visibility_clause("")
                lexical_visibility = visibility_clause("m.")
                visibility_args = (project, project, session_id)
            rows = self._conn.execute(
                "SELECT id,text,source,embedding,kind,confidence,importance,created_at "
                "FROM memories WHERE namespace=? AND (expires_at IS NULL OR expires_at>?)" + visibility,
                (namespace, now, *visibility_args),
            ).fetchall()
            lexical: dict[int, float] = {}
            terms = re.findall(r"[\w-]{2,}", query, flags=re.UNICODE)
            if terms:
                expression = " OR ".join(f'"{term}"' for term in terms[:16])
                for row in self._conn.execute(
                    "SELECT m.id,bm25(memories_fts) rank FROM memories_fts "
                    "JOIN memories m ON m.id=memories_fts.rowid "
                    "WHERE memories_fts MATCH ? AND m.namespace=? "
                    "AND (m.expires_at IS NULL OR m.expires_at>?)" + lexical_visibility + " LIMIT 100",
                    (expression, namespace, now, *visibility_args),
                ):
                    lexical[int(row["id"])] = 1.0
        ranked: list[SearchResult] = []
        for row in rows:
            memory_id = int(row["id"])
            semantic = self._cosine(qvec, array("f", row["embedding"]))
            age_days = max(0.0, (now - float(row["created_at"])) / 86400)
            freshness = 1.0 / (1.0 + age_days / 90.0)
            durable_bonus = 0.08 if row["kind"] == "durable" else 0.0
            score = 0.68 * semantic + 0.20 * lexical.get(memory_id, 0.0) + 0.07 * float(row["importance"]) + 0.05 * freshness + durable_bonus
            if score >= 0.34:
                ranked.append(SearchResult(memory_id, row["text"], row["source"], score, row["kind"], float(row["confidence"])))
        results = sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]
        if results:
            with self._lock, self._conn:
                self._conn.executemany(
                    "UPDATE memories SET accessed_at=?, access_count=access_count+1 WHERE id=?",
                    [(now, item.id) for item in results],
                )
        return results

    def propose(self, namespace: str, text: str, *, kind: str, confidence: float, source: str) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO candidates(namespace,text,kind,confidence,source,created_at) VALUES (?,?,?,?,?,?)",
                (namespace, text.strip(), kind, confidence, source, time.time()),
            )
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = self._conn.execute(
                "SELECT id FROM candidates WHERE namespace=? AND text=? AND status='pending'",
                (namespace, text.strip()),
            ).fetchone()
            return int(row[0])

    def list_candidates(self, namespace: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,text,kind,confidence,source,created_at FROM candidates WHERE namespace=? AND status='pending' ORDER BY id",
                (namespace,),
            ).fetchall()
        return [dict(row) for row in rows]

    def promote(self, namespace: str, candidate_id: int, embedding: Iterable[float]) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM candidates WHERE id=? AND namespace=? AND status='pending'",
                (candidate_id, namespace),
            ).fetchone()
        if not row:
            return False
        added = self.add(namespace, row["text"], embedding, source=row["source"], kind="durable", confidence=float(row["confidence"]), importance=0.9)
        with self._lock, self._conn:
            self._conn.execute("UPDATE candidates SET status='promoted' WHERE id=? AND namespace=?", (candidate_id, namespace))
        return added

    def reject(self, namespace: str, candidate_id: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute("UPDATE candidates SET status='rejected' WHERE id=? AND namespace=? AND status='pending'", (candidate_id, namespace))
        return cursor.rowcount == 1

    def delete(self, namespace: str, memory_id: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM memories WHERE namespace=? AND id=?", (namespace, memory_id))
        return cursor.rowcount == 1

    def delete_source_containing(self, namespace: str, source: str, needle: str) -> int:
        """Delete mirrored rows whose text contains Hermes' resolved old_text selector."""
        value = needle.strip()
        if not value:
            return 0
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id,text FROM memories WHERE namespace=? AND source=?",
                (namespace, source),
            ).fetchall()
            folded = value.casefold()
            ids = [int(row["id"]) for row in rows if folded in str(row["text"]).casefold()]
            if ids:
                self._conn.executemany("DELETE FROM memories WHERE id=?", [(memory_id,) for memory_id in ids])
            return len(ids)

    def count(self, namespace: str) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM memories WHERE namespace=?", (namespace,)).fetchone()[0])

    def namespaces(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT namespace,COUNT(*) entries,SUM(kind='durable') durable,SUM(kind='summary') summaries "
                "FROM memories GROUP BY namespace ORDER BY namespace"
            ).fetchall()
        return [dict(row) for row in rows]

    def prune(self, namespace: str) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM memories WHERE namespace=? AND expires_at IS NOT NULL AND expires_at<=?", (namespace, time.time()))
        return cursor.rowcount

    def clear_expirations(self, namespace: str) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE memories SET expires_at=NULL WHERE namespace=? AND expires_at IS NOT NULL",
                (namespace,),
            )
        return cursor.rowcount

    @staticmethod
    def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
        a, b = list(left), list(right)
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        return dot / norm if norm else 0.0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
