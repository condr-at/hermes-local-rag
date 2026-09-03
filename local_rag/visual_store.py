from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VisualResult:
    id: int
    path: str
    score: float
    metadata: dict[str, Any]


class VisualStore:
    def __init__(self, path: Path, *, dimensions: int = 512) -> None:
        self.path = Path(path)
        self.dimensions = dimensions
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, path TEXT NOT NULL, "
                "content_hash TEXT NOT NULL, embedding BLOB NOT NULL, created_at REAL NOT NULL, UNIQUE(namespace,path))"
            )

    def upsert(self, namespace: str, path: str, content_hash: str, embedding: Iterable[float]) -> int:
        vector = list(embedding)
        if len(vector) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} visual dimensions, got {len(vector)}")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO images(namespace,path,content_hash,embedding,created_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(namespace,path) DO UPDATE SET content_hash=excluded.content_hash,embedding=excluded.embedding,created_at=excluded.created_at",
                (namespace, path, content_hash, array("f", vector).tobytes(), time.time()),
            )
            row = self._conn.execute("SELECT id FROM images WHERE namespace=? AND path=?", (namespace, path)).fetchone()
        return int(row[0])

    def search(self, namespace: str, embedding: Iterable[float], *, limit: int = 5, minimum: float = 0.18) -> list[VisualResult]:
        query = list(embedding)
        with self._lock:
            rows = self._conn.execute("SELECT id,path,content_hash,embedding FROM images WHERE namespace=?", (namespace,)).fetchall()
        results = []
        for row in rows:
            score = self._cosine(query, array("f", row["embedding"]))
            if score >= minimum:
                results.append(VisualResult(int(row["id"]), row["path"], score, {"sha256": row["content_hash"]}))
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def count(self, namespace: str) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM images WHERE namespace=?", (namespace,)).fetchone()[0])

    def delete(self, namespace: str, image_id: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM images WHERE namespace=? AND id=?", (namespace, image_id))
        return cursor.rowcount == 1

    @staticmethod
    def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
        a, b = list(left), list(right)
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        return dot / norm if norm else 0.0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
