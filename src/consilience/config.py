"""Runtime configuration.

Everything is overridable by environment variable so the MCP server and the CLI
behave the same way without passing flags around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"
# Embedding models have a bounded context. We only need enough of each note to
# capture what it is about, not the whole thing, so we cap the input.
DEFAULT_MAX_CHARS = 8000


@dataclass(frozen=True)
class Config:
    model: str = DEFAULT_MODEL
    ollama_host: str = DEFAULT_HOST
    max_chars: int = DEFAULT_MAX_CHARS

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model=os.environ.get("CONSILIENCE_MODEL", DEFAULT_MODEL),
            ollama_host=os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/"),
            max_chars=int(os.environ.get("CONSILIENCE_MAX_CHARS", DEFAULT_MAX_CHARS)),
        )


def index_path(vault: Path) -> Path:
    """Where the SQLite index lives for a given vault.

    Kept inside the vault under a dot-folder so it travels with the notes and is
    trivially git-ignored, never mixed in with the notes themselves.
    """
    return vault / ".consilience" / "index.db"
