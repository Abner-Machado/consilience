# Specification

A short, deliberate spec written before the code. It is the contract the
implementation is measured against.

## Problem

A note-taker accumulates thousands of notes. Many pairs of notes are about the
same thing but were never linked, because the two notes were written weeks apart
and the connection was never noticed. Existing tools find links that are
*broken* or notes that are *orphaned*. None of them answer the harder question:
which notes *should* link to each other and do not?

## Goal

Given a folder of Markdown notes, produce a ranked list of note pairs that are
semantically close but have no wikilink between them.

## Non-goals

- Full-text or keyword search (well covered elsewhere).
- Finding broken links or orphans (the official Obsidian CLI does this).
- Editing the vault. Consilience only reads and reports.
- Any network call beyond a local Ollama instance.

## Design

- **Note identity.** A note is keyed by its file stem, lower-cased — the same
  rule Obsidian uses to resolve `[[wikilinks]]`.
- **Embeddings.** Each note's title and body (capped) are embedded by a local
  Ollama model. No cloud, no API key.
- **Storage.** One SQLite row per note: metadata, its outgoing links, and the
  raw embedding bytes. Re-indexing only touches notes whose mtime changed.
- **Suggestion.** Cosine similarity over the note-embedding matrix. A pair is
  suggested when its similarity clears a threshold and neither note links to the
  other. Links are treated as undirected for exclusion.

## Interfaces

- `consilience index <vault> [--reindex]` — reads the vault, needs Ollama.
- `consilience suggest <vault> [--top N] [--threshold T] [--note NAME] [--json]`
  — reads the index, runs offline.
- `consilience serve` — the same capability as MCP tools `suggest_links` and
  `related_notes`.

## Failure modes (must be legible, never a stack trace)

- Ollama unreachable → tell the reader to start it.
- Model not pulled → tell the reader the exact `ollama pull` command.
- Empty index on `suggest` → tell the reader to `index` first.
- Mixed embedding dimensions (model changed) → tell the reader to `--reindex`.

## Known limits (stated, not hidden)

- Pairwise comparison is O(n²). Comfortable to a few thousand notes.
- Long notes are truncated before embedding; very long notes are judged on their
  opening.
- Two notes sharing a file stem collide on their key; the last one wins.
