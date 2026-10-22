"""SQLite connection, versioned migrations and online backup for the interaction log."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

CURRENT_VERSION: int = 1

_SCHEMA_DIR = Path(__file__).resolve().parent
_MIGRATION_FILES: dict[int, str] = {1: "schema.sql"}


def connect(path: str) -> sqlite3.Connection:
    """Open a connection. File databases use WAL; foreign keys are always on; rows are sqlite3.Row."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order and return the resulting user_version."""
    version = _user_version(conn)
    for target in range(version + 1, CURRENT_VERSION + 1):
        sql = (_SCHEMA_DIR / _MIGRATION_FILES[target]).read_text(encoding="utf-8")
        script = f"BEGIN;\n{sql}\nPRAGMA user_version = {target};\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        version = target
    return _user_version(conn)


def backup(conn: sqlite3.Connection, dest_path: str) -> None:
    """Copy the whole database to dest_path with SQLite's online backup API."""
    with closing(sqlite3.connect(dest_path)) as dest:
        conn.backup(dest)
