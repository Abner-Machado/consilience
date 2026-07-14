"""MCP server.

Exposes the missing-link finder to any MCP client (Claude Code, Cursor, and
others) over stdio. Each call re-indexes incrementally so results reflect the
current state of the vault without a separate step.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import Config, index_path
from .embed import Embedder, OllamaError
from .engine import EngineError, Suggestion, index, load_records, suggest
from .store import connect
from .vault import note_key

mcp = FastMCP("consilience")


def _refresh(vault: Path, config: Config):
    conn = connect(index_path(vault))
    with Embedder(config) as embedder:
        index(vault, config, conn, embedder)
    return conn


def _format(results: list[Suggestion]) -> str:
    if not results:
        return "No missing links above the threshold."
    lines = [f"{len(results)} missing link(s):", ""]
    for s in results:
        lines.append(f"- {s.score:.3f}  {s.a_title} <-> {s.b_title}")
        lines.append(f"    {s.a_rel} | {s.b_rel}")
    return "\n".join(lines)


@mcp.tool()
def suggest_links(vault: str, top: int = 20, threshold: float = 0.75) -> str:
    """Find pairs of notes that are related but not linked to each other.

    vault: path to the folder of Markdown notes.
    top: maximum number of suggestions.
    threshold: minimum cosine similarity, from 0 to 1.
    """
    path = Path(vault).expanduser()
    if not path.is_dir():
        return f"Not a directory: {vault}"
    config = Config.from_env()
    try:
        conn = _refresh(path, config)
        try:
            results = suggest(load_records(conn), top=top, threshold=threshold)
        finally:
            conn.close()
    except (OllamaError, EngineError) as exc:
        return str(exc)
    return _format(results)


@mcp.tool()
def related_notes(vault: str, note: str, top: int = 10, threshold: float = 0.5) -> str:
    """List notes related to one note that it does not already link to.

    vault: path to the folder of Markdown notes.
    note: file name of the note (with or without the .md extension).
    top: maximum number of results.
    threshold: minimum cosine similarity, from 0 to 1.
    """
    path = Path(vault).expanduser()
    if not path.is_dir():
        return f"Not a directory: {vault}"
    config = Config.from_env()
    try:
        conn = _refresh(path, config)
        try:
            results = suggest(
                load_records(conn), top=top, threshold=threshold, focus=note_key(note)
            )
        finally:
            conn.close()
    except (OllamaError, EngineError) as exc:
        return str(exc)
    return _format(results)


def run() -> None:
    mcp.run()
