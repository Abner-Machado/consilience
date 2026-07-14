from pathlib import Path

from consilience.vault import iter_notes, link_targets, note_key


def test_note_key_normalises():
    assert note_key("My Note.md") == "my note"
    assert note_key("folder/My Note") == "my note"
    assert note_key("  Spaced  ") == "spaced"


def test_link_targets_strip_alias_and_anchor():
    text = "See [[Alpha]], [[Beta|the beta]] and [[Gamma#section]] plus [[Delta#^blk]]."
    assert link_targets(text) == {"alpha", "beta", "gamma", "delta"}


def test_link_targets_empty():
    assert link_targets("no links here") == set()


def test_iter_notes_reads_title_links_and_skips_dotfolders(tmp_path: Path):
    (tmp_path / "one.md").write_text("# First\nlinks to [[two]]", encoding="utf-8")
    (tmp_path / "two.md").write_text("plain body, no heading", encoding="utf-8")
    hidden = tmp_path / ".trash"
    hidden.mkdir()
    (hidden / "gone.md").write_text("# Ignored", encoding="utf-8")

    notes = {n.key: n for n in iter_notes(tmp_path, max_chars=1000)}

    assert set(notes) == {"one", "two"}
    assert notes["one"].title == "First"
    assert notes["one"].links == frozenset({"two"})
    assert notes["two"].title == "two"  # falls back to the file stem
