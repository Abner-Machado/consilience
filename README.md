# Consilience

**Find the links your notes are missing.**

Obsidian's CLI finds the links that are *broken*. Consilience finds the ones that
are *missing* — pairs of notes that are clearly about the same thing but were
never linked, because you wrote them three weeks apart and never noticed.

It runs entirely on your machine. No cloud, no API key, no account. Embeddings
come from a local [Ollama](https://ollama.com) model, and the index is a single
SQLite file that lives next to your notes.

---

## Why I built this

I keep a large vault, and every few months I stumble on two notes that obviously
belong together and think "how did I never link these?" Search doesn't help with
that — search finds notes when you already know what you're looking for. The
connections I was missing were exactly the ones I wasn't searching for.

So the tool does one thing. It reads the vault, embeds each note locally, and
tells me which pairs are close in meaning but have no link between them. Nothing
else. It doesn't edit my notes, it doesn't phone home, and it stays out of the
way until I ask it a question.

## Install

From source (Python 3.10+):

```bash
git clone https://github.com/Abner-Machado/consilience
cd consilience
pipx install .   # or: pip install .
```

You also need Ollama with an embedding model:

```bash
ollama pull nomic-embed-text
```

## Use it

Index the vault (this is the step that needs Ollama), then ask for suggestions
(this step runs offline):

```bash
consilience index ~/notes
consilience suggest ~/notes
```

Example:

```
7 missing link(s):

  0.891  Spaced repetition  <->  Why I forget things
          learning/spaced-repetition.md   |   journal/2025-forgetting.md
  0.864  Stoicism           <->  Dealing with setbacks
          philosophy/stoicism.md          |   journal/setbacks.md
```

Useful flags:

```bash
consilience suggest ~/notes --threshold 0.82   # stricter matches only
consilience suggest ~/notes --note "Stoicism"  # just this note's missing links
consilience suggest ~/notes --json             # for scripts
```

Re-running `index` only re-embeds notes that changed, so it stays fast.

## Use it from an MCP client

Consilience speaks the Model Context Protocol, so an assistant can ask for
suggestions directly. Point your client at it:

```json
{
  "mcpServers": {
    "consilience": {
      "command": "consilience",
      "args": ["serve"]
    }
  }
}
```

Two tools are exposed: `suggest_links` (missing links across the whole vault) and
`related_notes` (missing links for one note). Both re-index incrementally before
answering.

## How it works

1. Every `.md` file is read; its title, body, and outgoing `[[wikilinks]]` are
   parsed. Notes are keyed by file stem, the way Obsidian resolves links.
2. Each note is embedded by the local model and stored in SQLite as raw float32
   bytes. No vector database.
3. On `suggest`, note embeddings are compared with cosine similarity. A pair is
   returned when it clears the threshold and neither note already links to the
   other, in either direction.

The full design is in [SPEC.md](SPEC.md).

## Development

Install the package with its test dependencies and run the suite:

```bash
pip install -e ".[dev]"
pytest
```

The tests in `tests/` cover the vault parser and the suggestion engine, and run
fully offline — no Ollama or network access is needed.

## Where it fits

- **Obsidian's official CLI** and plugins like *Vault Inspector* handle broken
  links and orphans. Consilience does not — it is the complementary half.
- **Smart Connections** shows related notes while you read one note. Consilience
  audits the whole vault at once and works from the command line or an MCP
  client, with no plugin and no Obsidian instance running.

## Limits

Stated plainly, because they matter:

- Comparison is O(n²). It's comfortable up to a few thousand notes; a very large
  vault will be slow.
- Long notes are truncated before embedding, so a very long note is judged on its
  opening.
- Two notes that share a file name collide on their key; the last one wins.

## License

MIT. See [LICENSE](LICENSE).
