"""Indexing and the one thing this tool exists to do: suggest missing links.

A suggestion is a pair of notes whose contents are semantically close but which do
not link to each other. Pairs that are already linked, in either direction, are
never suggested — the point is to surface connections you have *not* made yet.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config
from .embed import Embedder
from .store import Record, load, mtimes, prune, upsert
from .vault import iter_notes


class EngineError(RuntimeError):
    pass


@dataclass
class IndexResult:
    embedded: int
    total: int
    pruned: int


@dataclass
class Suggestion:
    a_title: str
    a_rel: str
    b_title: str
    b_rel: str
    score: float


def index(
    vault: Path,
    config: Config,
    conn: sqlite3.Connection,
    embedder: Embedder,
    reindex: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> IndexResult:
    if reindex:
        conn.execute("DELETE FROM notes")

    notes = list(iter_notes(vault, config.max_chars))
    known = mtimes(conn)
    changed = [n for n in notes if known.get(n.key) != n.mtime]

    for done, note in enumerate(changed, start=1):
        payload = f"{note.title}\n{note.text}"
        vec = embedder.embed([payload])[0]
        upsert(conn, note.key, note.rel, note.title, note.mtime, note.links, vec)
        if on_progress:
            on_progress(done, len(changed))

    pruned = prune(conn, {n.key for n in notes})
    conn.commit()
    return IndexResult(embedded=len(changed), total=len(notes), pruned=pruned)


def _matrix(records: list[Record]) -> np.ndarray:
    dims = {r.vec.shape[0] for r in records}
    if len(dims) > 1:
        raise EngineError(
            "The index mixes embedding dimensions, which usually means the model "
            "changed. Rebuild it with `consilience index --reindex`."
        )
    matrix = np.vstack([r.vec for r in records]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def suggest(
    records: list[Record],
    top: int = 20,
    threshold: float = 0.75,
    focus: str | None = None,
) -> list[Suggestion]:
    if len(records) < 2:
        return []

    matrix = _matrix(records)
    similarity = matrix @ matrix.T
    index_of = {r.key: i for i, r in enumerate(records)}

    linked: set[tuple[int, int]] = set()
    for i, record in enumerate(records):
        for target in record.links:
            j = index_of.get(target)
            if j is not None and j != i:
                linked.add((min(i, j), max(i, j)))

    if focus is not None:
        f = index_of.get(focus)
        if f is None:
            raise EngineError(f"Note '{focus}' is not in the index.")
        pairs = [(min(f, j), max(f, j)) for j in range(len(records)) if j != f]
    else:
        rows, cols = np.triu_indices(len(records), k=1)
        pairs = list(zip(rows.tolist(), cols.tolist()))

    scored = [
        (float(similarity[i, j]), i, j)
        for i, j in pairs
        if (i, j) not in linked and similarity[i, j] >= threshold
    ]
    scored.sort(reverse=True)

    out = []
    for score, i, j in scored[:top]:
        a, b = records[i], records[j]
        out.append(Suggestion(a.title, a.rel, b.title, b.rel, round(score, 4)))
    return out


def load_records(conn: sqlite3.Connection) -> list[Record]:
    return load(conn)
