# Hermes Local RAG

Private, fully local hybrid text and visual memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent), implemented through the official `MemoryProvider` extension point.

## Features

- EmbeddingGemma 300M through LiteRT/XNNPACK, with configurable 128/256/512/768-dimensional Matryoshka embeddings (512d default)
- SQLite + FTS5 + vector similarity, score thresholds, freshness/importance boosts, source diversity, and bounded context injection
- Physical profile/user namespaces before retrieval
- Episodic memory, extractive session summaries, durable-memory review queue, provenance, deletion, migrations, and backups
- Infinite retention by default; independent episodic and summary TTL settings
- Redacted Hermes session backfill without reading internal `state.db` tables
- Project-root-confined text-file indexing with secret and prompt-injection rejection
- Optional CLIP ViT-B/32 ONNX visual index in a separate 512d vector space
- No model server, vector database server, GUI, or Hermes core patch

## Install

Prerequisites: an existing Hermes Agent installation, `uv`, Hugging Face authentication, and accepted Gemma Terms for the gated EmbeddingGemma repositories.

```bash
git clone https://github.com/condr-at/hermes-local-rag.git
cd hermes-local-rag
python3 install.py
```

The installer:

1. installs pinned runtime dependencies into the Hermes venv;
2. copies the plugin to `$HERMES_HOME/plugins/local_rag`;
3. downloads model artifacts to `$HERMES_HOME/models`;
4. activates it with `hermes config set memory.provider local_rag`.

It never deletes `$HERMES_HOME/local-rag`, where indexes and configuration live.

Verify after installation or a Hermes update:

```bash
python3 install.py --check
```

Then restart the gateway from a separate shell, or start a new Desktop session.

## Configuration

`~/.hermes/local-rag/config.json`:

```json
{
  "embedding_dimensions": 512,
  "episodic_ttl_days": null,
  "summary_ttl_days": null
}
```

`null` means infinite retention. Changing `embedding_dimensions` requires explicit reindexing; startup fails loudly rather than mixing incompatible vectors.

## Backfill

```bash
PYTHONPATH="$HOME/.hermes/plugins" "$HOME/.hermes/hermes-agent/venv/bin/python" -m local_rag.backfill
```

Backfill asks Hermes for an official redacted JSONL export, routes records by profile and platform user, indexes user messages plus extractive summaries, and deletes the temporary export.

## Maintenance

The provider exposes tools for status, search, deletion, candidate review, file/session ingestion, pruning, and visual search. A standalone CLI is also installed as `hermes-local-rag`.

See [`local_rag/README.md`](local_rag/README.md) for architecture, policies, tools, and recovery details.

## Privacy and licensing

No indexed data, model weights, credentials, or session exports belong in this repository. Model weights are downloaded directly from their upstream repositories and retain their own terms. EmbeddingGemma is governed by Gemma Terms. Plugin code is MIT licensed.
