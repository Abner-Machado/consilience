"""Reading a folder of Markdown notes.

A note is identified by its file stem, lower-cased, the same way Obsidian resolves
``[[wikilinks]]``. That keeps link resolution predictable without needing Obsidian
itself or its REST API running.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# [[Target]], [[Target|alias]], [[Target#heading]], [[Target#^block]]
_WIKILINK = re.compile(r"\[\[([^\]]+?)\]\]")
_H1 = re.compile(r"^\s{0,3}#\s+(.+?)\s*$", re.MULTILINE)


def note_key(name: str) -> str:
    """Normalise a note name or link target to a comparable key."""
    name = name.strip().split("/")[-1].split("\\")[-1]
    if name.lower().endswith(".md"):
        name = name[:-3]
    return name.lower()


def link_targets(text: str) -> set[str]:
    """Extract the wikilink targets referenced in a note body."""
    targets: set[str] = set()
    for raw in _WIKILINK.findall(text):
        target = raw.split("|", 1)[0]  # drop display alias
        target = target.split("#", 1)[0]  # drop heading / block anchor
        target = target.strip()
        if target:
            targets.add(note_key(target))
    return targets


@dataclass(frozen=True)
class Note:
    key: str
    rel: str  # path relative to the vault, for display
    path: Path
    title: str
    text: str
    mtime: float
    links: frozenset[str]


def _title_of(text: str, stem: str) -> str:
    match = _H1.search(text)
    return match.group(1).strip() if match else stem


def iter_notes(vault: Path, max_chars: int) -> Iterator[Note]:
    """Yield every Markdown note in the vault, skipping dot-folders."""
    vault = vault.resolve()
    for path in sorted(vault.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(vault).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        stem = path.stem
        yield Note(
            key=note_key(stem),
            rel=str(path.relative_to(vault)).replace("\\", "/"),
            path=path,
            title=_title_of(text, stem),
            text=text[:max_chars],
            mtime=path.stat().st_mtime,
            links=frozenset(link_targets(text)),
        )
