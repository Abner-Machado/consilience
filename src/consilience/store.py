"""SQLite index of notes and their embeddings.

One row per note. Embeddings are stored as raw float32 bytes; there is no vector
database, no external service. A few thousand notes fit comfortably in memory for
the pairwise comparison, which is all this tool needs.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    key   TEXT PRIMARY KEY,
    rel   TEXT NOT NULL,
    title TEXT NOT NULL,
    mtime REAL NOT NULL,
    links TEXT NOT NULL,
    dim   INTEGER NOT NULL,
    vec   BLOB NOT NULL
);
"""


@dataclass
class Record:
    key: str
    rel: str
    title: str
    links: frozenset[str]
    vec: np.ndarray


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def mtimes(conn: sqlite3.Connection) -> dict[str, float]:
    return {key: mtime for key, mtime in conn.execute("SELECT key, mtime FROM notes")}


def upsert(
    conn: sqlite3.Connection,
    key: str,
    rel: str,
    title: str,
    mtime: float,
    links: frozenset[str],
    vec: np.ndarray,
) -> None:
    vec = np.ascontiguousarray(vec, dtype=np.float32)
    conn.execute(
        "INSERT INTO notes (key, rel, title, mtime, links, dim, vec) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET "
        "rel=excluded.rel, title=excluded.title, mtime=excluded.mtime, "
        "links=excluded.links, dim=excluded.dim, vec=excluded.vec",
        (key, rel, title, mtime, json.dumps(sorted(links)), vec.shape[0], vec.tobytes()),
    )


def prune(conn: sqlite3.Connection, present: set[str]) -> int:
    stale = [key for (key,) in conn.execute("SELECT key FROM notes") if key not in present]
    conn.executemany("DELETE FROM notes WHERE key = ?", ((key,) for key in stale))
    return len(stale)


def load(conn: sqlite3.Connection) -> list[Record]:
    records = []
    for key, rel, title, links, dim, vec in conn.execute(
        "SELECT key, rel, title, links, dim, vec FROM notes ORDER BY rel"
    ):
        records.append(
            Record(
                key=key,
                rel=rel,
                title=title,
                links=frozenset(json.loads(links)),
                vec=np.frombuffer(vec, dtype=np.float32, count=dim),
            )
        )
    return records
