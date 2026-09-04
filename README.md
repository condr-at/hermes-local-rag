# Hermes Local RAG

[![Hermes Local RAG — private local memory that keeps the useful card and bins the noise](https://raw.githubusercontent.com/condr-at/hermes-local-rag/main/assets/branding/local-rag-preview-1600x900.webp)](https://github.com/condr-at/hermes-local-rag)

Private, fully local hybrid text and visual memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent), implemented through the official `MemoryProvider` extension point.

## Features

- EmbeddingGemma 300M through LiteRT/XNNPACK, with configurable 128/256/512/768-dimensional Matryoshka embeddings (512d default)
- SQLite + FTS5 + vector similarity, score thresholds, freshness/importance boosts, source diversity, and bounded context injection
- Physical profile/user namespaces before retrieval
- Explicit contextual admission through `local_rag_remember`, with normalized atomic items instead of raw turns
- Selective historical backfill with pre-LLM filtering, structured extraction, resumable preview, human review, and explicit apply
- Independent scope, durability, status, confidence, importance, provenance, deduplication, deletion, migrations, and backups
- Project- and session-aware retrieval boundaries; built-in Markdown memory is mirrored as global durable context
- Project-root-confined text-file indexing with secret and prompt-injection rejection
- Optional CLIP ViT-B/32 ONNX visual index in a separate 512d vector space
- Button-driven setup inside Hermes Dashboard; no model server, vector database server, or Hermes core patch

## Install without Terminal

Prerequisite: an existing Hermes Agent installation.

1. Open **Hermes Dashboard → Plugins → Install**.
2. Install `condr-at/hermes-local-rag/local_rag`.
3. Open the new **Local RAG** tab.
4. Click **Install dependencies**.
5. Open and accept the required Gemma Terms, create a read-only Hugging Face token, paste it into the password field, and click **Sign in**.
6. Choose **Text only** or **Text + visual**, then click **Download selected models**.
7. Optionally create a historical preview, edit and accept individual candidates, then explicitly apply the reviewed subset.
8. Save the memory settings and click **Activate Local RAG**.
9. Click **Run health check**.

The dashboard never displays or logs the Hugging Face token. Model downloads and setup subprocesses use fixed argument lists and a controlled environment rather than a user shell. Gemma Terms acceptance is explicit and cannot be bypassed by the installer.

The setup wizard:

1. installs pinned runtime dependencies into the Hermes venv;
2. downloads selected model artifacts to `$HERMES_HOME/models`;
3. writes configuration atomically;
4. activates `memory.provider=local_rag`;
5. checks runtime imports, models, the database, and provider availability.

It never deletes `$HERMES_HOME/local-rag`, where indexes and configuration live. Backfill is off by default and requires explicit confirmation.

### Terminal fallback

The technical installer remains available for development and recovery:

```bash
git clone https://github.com/condr-at/hermes-local-rag.git
cd hermes-local-rag
python3 install.py
```

Verify after installation or a Hermes update:

```bash
python3 install.py --check
```

Then restart the gateway, or start a new Desktop session.

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

Raw transcript backfill is disabled. Historical sessions pass through role filtering, credential and prompt-injection rejection, bounded structured extraction, and strict validation. Preview writes only a mode-`0600` review artifact; every candidate starts rejected, and only explicitly accepted items can be applied. Namespace, project, session, timestamps, and source-message provenance come from the trusted export rather than the model.

The Dashboard owns the normal preview → edit/accept → apply flow. The equivalent recovery CLI is:

```bash
hermes local_rag preview --plan /path/to/review.json
# Edit only accepted flags/text after reviewing the artifact.
hermes local_rag apply --plan /path/to/review.json
```

Preview is resumable by session and never writes the memory database. Re-applying a reviewed plan is idempotent through scope-aware exact deduplication.

## Maintenance

The provider exposes tools for status, search, deletion, candidate review, file/session ingestion, pruning, and visual search. A standalone CLI is also installed as `hermes-local-rag`.

See [`local_rag/README.md`](local_rag/README.md) for architecture, policies, tools, and recovery details.

## Privacy and licensing

No indexed data, model weights, credentials, or session exports belong in this repository. Model weights are downloaded directly from their upstream repositories and retain their own terms. EmbeddingGemma is governed by Gemma Terms. Plugin code is MIT licensed.
