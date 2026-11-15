# Hermes Local RAG

> **1.6.1:** managed-runtime dependency recovery and bootstrap-aware subprocesses after Hermes updates.
> Read [CHANGELOG.md](CHANGELOG.md) for changes and limitations, and
> [UPGRADE.md](UPGRADE.md) for the shared-inference and curated-image migration introduced in 1.4.0.
> See the [synthetic benchmark and its limitations](evaluation/README.md).

[![Hermes Local RAG — private local memory that keeps the useful card and bins the noise](https://raw.githubusercontent.com/condr-at/hermes-local-rag/main/assets/branding/local-rag-preview-1600x900.webp)](https://github.com/condr-at/hermes-local-rag)

Private, fully local hybrid text and visual memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent), implemented through the official `MemoryProvider` extension point.

## Features

- EmbeddingGemma 300M through LiteRT/XNNPACK, with configurable 128/256/512/768-dimensional Matryoshka embeddings (512d default)
- SQLite + FTS5 + vector similarity, score thresholds, freshness/importance boosts, source diversity, and bounded context injection
- Physical profile/user namespaces before retrieval
- Explicit contextual admission through `local_rag_remember`, with normalized atomic items instead of raw turns
- In-place `local_rag_update` corrections with exact-text conflict detection, stable identity, immutable scope/provenance, and transactional rollback
- Selective historical backfill with pre-LLM filtering, structured extraction, resumable preview, human review, and explicit apply
- Independent scope, durability, status, confidence, importance, provenance, deduplication, deletion, migrations, and backups
- Project- and session-aware retrieval boundaries; built-in Markdown memory is mirrored as global durable context
- Project-root-confined text-file indexing with secret and prompt-injection rejection
- Optional CLIP ViT-B/32 ONNX visual index in a separate 512d vector space
- Button-driven setup inside Hermes Dashboard; no vector database server or Hermes core patch. One private local inference service starts on the first embedding request (see UPGRADE.md).

## Install

Published package: [hermes-local-rag on PyPI](https://pypi.org/project/hermes-local-rag/)

For managed Hermes installations, use the native installer so dependencies remain part of Hermes' managed environment:

```bash
hermes plugins install condr-at/hermes-local-rag/local_rag --yes-deps --enable
hermes config set memory.provider local_rag
```

Restart existing Hermes processes after installation. Do not install packages into an obsolete `hermes-agent/venv` or the bare bootstrap interpreter.

For **traditional, unmanaged virtualenvs only**, packaged-provider discovery also supports installing into the same Python environment that runs Hermes:

> Upgrading from 1.3.2 or older: stop every Hermes Desktop and gateway process first. Version 1.4.0 performs a one-time, integrity-checked migration from `memory.sqlite` / `visual.sqlite` to backup-safe `.db` files and removes the legacy files after success.

```bash
python -m pip install hermes-local-rag==1.6.1
hermes config set memory.provider local_rag
```

The package exposes `local_rag -> local_rag:register` through the official `hermes_agent.memory_providers` entry-point group. Model download and initial configuration are still required; the Dashboard plugin flow below is the recommended guided setup.

### Install without Terminal

Prerequisite: an existing Hermes Agent installation.

1. Open **Hermes Dashboard → Plugins → Install**.
2. Install `condr-at/hermes-local-rag/local_rag`.
3. Open the new **Local RAG** tab.
4. Click **Install dependencies**, then restart Hermes and its dashboard before continuing. This uses native `hermes plugins enable local_rag` dependency admission; it does not repair the running process.
5. Open and accept the required Gemma Terms, create a read-only Hugging Face token, paste it into the password field, and click **Sign in**.
6. Choose **Text only** or **Text + visual**, then click **Download selected models**.
7. Optionally create a historical preview, edit and accept individual candidates, then explicitly apply the reviewed subset.
8. Save the memory settings and click **Activate Local RAG**.
9. Click **Run health check**.

The dashboard never displays or logs the Hugging Face token. Model downloads and setup subprocesses use fixed argument lists and a controlled environment rather than a user shell. Gemma Terms acceptance is explicit and cannot be bypassed by the installer.

The setup wizard:

1. asks Hermes to admit the manifest dependencies into its managed runtime;
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

The installer copies the plugin and runs native `hermes plugins enable local_rag`.
It does not use the terminal Python to install packages into Hermes. Restart Hermes
and its dashboard, then finish model terms, authentication, downloads, and provider
activation in the **Local RAG** tab. `--skip-models` remains accepted for scripted
code-only setup; model downloads now always happen in the restarted dashboard.
`--check` checks installation artifacts only; success does not establish runtime health.
Native `hermes memory status` can exit zero for an unavailable provider. Use the
dashboard health check to verify dependency imports and provider availability.
Python children re-enter the native Hermes bootstrap to activate selected dependencies;
standalone virtual environments retain their own interpreter. No service restart is automatic.

Use `--home /path/to/profile` (or `HERMES_HOME`) to select the installation profile.
All native commands pin that profile, including the default, instead of following
the sticky active profile. `HERMES_PYTHON` is no longer used.
If an already-enabled plugin still has missing dependencies after restart, run
`HERMES_HOME=/path/to/profile hermes --profile NAME pm install`, then retry
`plugins enable local_rag` with the same home/profile and restart again. Use
`NAME=default` for a root home.

## Configuration

`~/.hermes/local-rag/config.json`:

```json
{
  "embedding_dimensions": 512,
  "episodic_ttl_days": null,
  "summary_ttl_days": null,
  "debug_enabled": false
}
```

`null` means infinite retention. Changing `embedding_dimensions` requires explicit reindexing; startup fails loudly rather than mixing incompatible vectors.

`debug_enabled` is a **Local RAG plugin setting**, off by default. Enable **Append Local RAG diagnostics to chat responses** in the plugin's settings and save configuration. An already-running provider picks up this flag on the next turn without reloading models.

The native final-output hook adds a compact Russian Markdown footer: the provider's prefetch payload, explicit search results returned to the agent, confirmed saves/corrections/deletions, image/document operations, Markdown mirroring, and observed rejections or failures. Retrieval does not prove that the model used a fact or inspected image pixels. A missing save is not labelled a deliberate rejection; an ambiguous storage refusal remains ambiguous.

No extra LLM runs, application changes, transcript indexing, or separate diagnostic log are involved. The bounded in-memory trace is cleared on completion, session end/switch, or shutdown. **The footer exposes memory excerpts in the current chat and may be retained by that chat's normal history.** The flag applies to this Hermes profile; leave it off for chats where those excerpts should not be shown. Another plugin's first-winning output transform can suppress the footer; interrupted turns may not have a final footer.

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
