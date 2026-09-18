"""The SQLite index and the incremental indexing loop built on top of it.

A fake embedder counts its calls, so these tests prove that unchanged notes are
not embedded again, that deleted notes leave the index, and that ``--reindex``
really starts from scratch.
"""

import os
from pathlib import Path

import numpy as np

from consilience.config import Config
from consilience.engine import index
from consilience.store import connect, load, mtimes, prune, upsert


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return [np.array([float(len(t)), 1.0], dtype=np.float32) for t in texts]


def test_upsert_roundtrip_keeps_vector_and_links(tmp_path: Path):
    conn = connect(tmp_path / "idx" / "notes.db")
    vec = np.array([0.5, -1.25, 3.0], dtype=np.float64)  # stored as float32
    upsert(conn, "a", "a.md", "A", 1.0, frozenset({"b", "c"}), vec)

    (record,) = load(conn)
    assert record.key == "a"
    assert record.rel == "a.md"
    assert record.title == "A"
    assert record.links == frozenset({"b", "c"})
    assert record.vec.dtype == np.float32
    assert record.vec.tolist() == [0.5, -1.25, 3.0]
    assert mtimes(conn) == {"a": 1.0}


def test_upsert_replaces_existing_row(tmp_path: Path):
    conn = connect(tmp_path / "notes.db")
    upsert(conn, "a", "old/a.md", "Old", 1.0, frozenset(), np.array([1.0, 0.0]))
    upsert(conn, "a", "new/a.md", "New", 2.0, frozenset({"z"}), np.array([0.0, 1.0]))

    (record,) = load(conn)
    assert record.rel == "new/a.md"
    assert record.title == "New"
    assert record.links == frozenset({"z"})
    assert record.vec.tolist() == [0.0, 1.0]
    assert mtimes(conn) == {"a": 2.0}


def test_prune_removes_only_absent_keys(tmp_path: Path):
    conn = connect(tmp_path / "notes.db")
    for key in ("a", "b", "c"):
        upsert(conn, key, f"{key}.md", key, 1.0, frozenset(), np.array([1.0, 0.0]))

    assert prune(conn, {"a", "c"}) == 1
    assert sorted(r.key for r in load(conn)) == ["a", "c"]
    assert prune(conn, {"a", "c"}) == 0


def test_load_orders_by_relative_path(tmp_path: Path):
    conn = connect(tmp_path / "notes.db")
    upsert(conn, "z", "b/z.md", "Z", 1.0, frozenset(), np.array([1.0]))
    upsert(conn, "m", "a/m.md", "M", 1.0, frozenset(), np.array([1.0]))
    assert [r.rel for r in load(conn)] == ["a/m.md", "b/z.md"]


def _write(path: Path, text: str, mtime: float) -> None:
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_index_only_embeds_changed_notes(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write(vault / "one.md", "# One\nlinks to [[two]]", 100.0)
    _write(vault / "two.md", "# Two\nbody", 100.0)
    conn = connect(tmp_path / "notes.db")
    config = Config(max_chars=1000)
    embedder = FakeEmbedder()

    first = index(vault, config, conn, embedder)
    assert (first.embedded, first.total, first.pruned) == (2, 2, 0)
    assert embedder.calls == 2

    # Nothing changed: no note is sent to the embedder again.
    second = index(vault, config, conn, embedder)
    assert (second.embedded, second.total, second.pruned) == (0, 2, 0)
    assert embedder.calls == 2

    # Touch one note: only that one is re-embedded, and its links are refreshed.
    _write(vault / "one.md", "# One\nnow links to nothing", 200.0)
    third = index(vault, config, conn, embedder)
    assert third.embedded == 1
    assert embedder.calls == 3
    by_key = {r.key: r for r in load(conn)}
    assert by_key["one"].links == frozenset()


def test_index_prunes_deleted_notes_and_reindex_starts_over(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write(vault / "one.md", "# One", 100.0)
    _write(vault / "two.md", "# Two", 100.0)
    conn = connect(tmp_path / "notes.db")
    config = Config(max_chars=1000)
    embedder = FakeEmbedder()
    index(vault, config, conn, embedder)

    (vault / "two.md").unlink()
    result = index(vault, config, conn, embedder)
    assert (result.embedded, result.total, result.pruned) == (0, 1, 1)
    assert [r.key for r in load(conn)] == ["one"]

    result = index(vault, config, conn, embedder, reindex=True)
    assert result.embedded == 1
    assert embedder.calls == 3


def test_index_reports_progress(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    for name in ("a", "b", "c"):
        _write(vault / f"{name}.md", f"# {name}", 100.0)
    conn = connect(tmp_path / "notes.db")
    config = Config(max_chars=1000)
    seen = []

    index(vault, config, conn, FakeEmbedder(), on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
