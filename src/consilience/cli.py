"""Command line interface.

Two verbs do the work: ``index`` reads the vault and needs Ollama; ``suggest``
reads the index and runs fully offline. Keeping them separate means you can query
suggestions without a model loaded once the vault has been indexed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, index_path
from .embed import Embedder, OllamaError
from .engine import EngineError, index, load_records, suggest
from .store import connect
from .vault import note_key


def _cmd_index(args: argparse.Namespace) -> int:
    vault = Path(args.vault)
    if not vault.is_dir():
        print(f"Not a directory: {vault}", file=sys.stderr)
        return 2

    config = Config.from_env()
    conn = connect(index_path(vault))
    try:
        with Embedder(config) as embedder:
            def progress(done: int, total: int) -> None:
                print(f"\r  embedding {done}/{total}", end="", flush=True)

            result = index(vault, config, conn, embedder, reindex=args.reindex,
                           on_progress=progress)
    except OllamaError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if result.embedded:
        print()
    print(
        f"Indexed {result.total} notes "
        f"({result.embedded} new or changed, {result.pruned} removed)."
    )
    return 0


def _cmd_suggest(args: argparse.Namespace) -> int:
    vault = Path(args.vault)
    conn = connect(index_path(vault))
    try:
        records = load_records(conn)
    finally:
        conn.close()

    if not records:
        print(f"No index found. Run `consilience index {vault}` first.",
              file=sys.stderr)
        return 1

    focus = note_key(args.note) if args.note else None
    try:
        results = suggest(records, top=args.top, threshold=args.threshold, focus=focus)
    except EngineError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([s.__dict__ for s in results], indent=2, ensure_ascii=False))
        return 0

    if not results:
        print("No missing links above the threshold. Try a lower --threshold.")
        return 0

    print(f"{len(results)} missing link(s):\n")
    for s in results:
        print(f"  {s.score:.3f}  {s.a_title}  <->  {s.b_title}")
        print(f"          {s.a_rel}   |   {s.b_rel}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import run

    run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="consilience",
        description="Find the links your notes are missing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="scan the vault and embed changed notes")
    p_index.add_argument("vault", help="path to the notes folder")
    p_index.add_argument("--reindex", action="store_true",
                         help="rebuild from scratch (use after changing the model)")
    p_index.set_defaults(func=_cmd_index)

    p_suggest = sub.add_parser("suggest", help="list unlinked but related notes")
    p_suggest.add_argument("vault", help="path to the notes folder")
    p_suggest.add_argument("--top", type=int, default=20, help="max suggestions (default 20)")
    p_suggest.add_argument("--threshold", type=float, default=0.75,
                           help="minimum cosine similarity, 0-1 (default 0.75)")
    p_suggest.add_argument("--note", help="only suggestions involving this note")
    p_suggest.add_argument("--json", action="store_true", help="machine-readable output")
    p_suggest.set_defaults(func=_cmd_suggest)

    p_serve = sub.add_parser("serve", help="run the MCP server over stdio")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
