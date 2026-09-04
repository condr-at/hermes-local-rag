from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _require_integrity(connection: sqlite3.Connection, path: Path) -> None:
    result = connection.execute("PRAGMA integrity_check").fetchone()
    if not result or result[0] != "ok":
        raise RuntimeError(f"SQLite integrity check failed for {path}")


def _sidecars(path: Path) -> tuple[Path, ...]:
    return (Path(f"{path}-wal"), Path(f"{path}-shm"), Path(f"{path}-journal"))


def _unlink_database(path: Path) -> None:
    path.unlink(missing_ok=True)
    for sidecar in _sidecars(path):
        sidecar.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _migration_lock(directory: Path, name: str) -> Iterator[None]:
    path = directory / f".{name}.migration.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired = True
        yield
    finally:
        if acquired and os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        elif acquired:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def canonical_database_path(directory: Path, name: str) -> Path:
    """Return a backup-safe .db path, migrating a quiescent legacy .sqlite DB.

    Migration must run only after every process using the old plugin has stopped.
    The inter-process lock serializes new-version migrators but cannot coordinate
    with old versions that do not participate in this protocol.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.db"
    legacy = directory / f"{name}.sqlite"

    with _migration_lock(directory, name):
        if not legacy.exists():
            old_schema_backup = directory / "memory.pre-v2.sqlite"
            if name == "memory" and old_schema_backup.exists() and target.exists():
                with sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True) as current:
                    _require_integrity(current, target)
                _unlink_database(old_schema_backup)
                _fsync_directory(directory)
            return target

        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".db", dir=directory)
        os.close(descriptor)
        temporary = Path(temporary_name)
        source = destination = None
        try:
            source = sqlite3.connect(f"{legacy.resolve().as_uri()}?mode=ro", uri=True)
            destination = sqlite3.connect(temporary)
            source.backup(destination)
            _require_integrity(destination, temporary)
            destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            destination.execute("PRAGMA journal_mode=DELETE")
            destination.close()
            destination = None
            source.close()
            source = None

            with temporary.open("rb") as database_file:
                os.fsync(database_file.fileno())
            for sidecar in _sidecars(target):
                sidecar.unlink(missing_ok=True)
            _fsync_directory(directory)
            os.replace(temporary, target)
            target.chmod(0o600)
            _fsync_directory(directory)

            with sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True) as migrated:
                _require_integrity(migrated, target)
            _unlink_database(legacy)
            if name == "memory":
                _unlink_database(directory / "memory.pre-v2.sqlite")
            _fsync_directory(directory)
            return target
        finally:
            for connection in (destination, source):
                if connection is not None:
                    connection.close()
            _unlink_database(temporary)
