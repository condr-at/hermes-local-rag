# Hermes Local RAG

User-local `MemoryProvider` for private hybrid recall. It lives outside the Hermes checkout and survives normal updates.

## Storage and isolation

- Text: `~/.hermes/local-rag/memory.db` (SQLite WAL + FTS5 + local vectors)
- Curated images: `~/.hermes/local-rag/curated-images.db` and managed `images/` originals.
- Legacy `visual.db` is preserved, not automatically promoted into curated recall.
- Version 1.4.0 shares one private inference daemon per profile, started on demand.
  See [upgrade and recovery details](https://github.com/condr-at/hermes-local-rag/blob/main/UPGRADE.md).
- Namespace: `<profile>:<platform-user>`, with Desktop/CLI mapped to `<profile>:local`
- Model weights are shared read-only; records are always filtered by namespace before ranking.

## Models

- Text: EmbeddingGemma 300M LiteRT `seq512`, 512-dimensional Matryoshka output by default; configurable to 128/256/512/768.
- Visual: quantized CLIP ViT-B/32 ONNX, 512 dimensions. Loaded lazily only for visual tools.

The 512-token artifact is intentional. Normal chunks are at most 384 words with 64-word overlap; context injection has a separate 3,200-character hard budget. A 2048-token model would increase interactive work without improving these bounded chunks.

Model weights are not distributed by this plugin. EmbeddingGemma remains subject to Gemma Terms; CLIP artifacts retain their upstream license.

## Dashboard setup

After installing the plugin from **Hermes Dashboard → Plugins**, open the **Local RAG** tab. The wizard installs dependencies, accepts a read-only Hugging Face token through a local password field, downloads the selected models, saves retention and embedding settings, activates the provider, and performs a health check. No Terminal is required for the normal path.

The token is passed directly to `huggingface_hub.login`, is never placed in a subprocess argument, response, config file, or setup log, and remains governed by Hugging Face's credential storage. Gemma Terms must still be accepted by the user in the browser.

## Ingestion

- Raw user turns and session summaries are not automatically ingested.
- `~/.hermes/local-rag/config.json` can set `episodic_ttl_days` and `summary_ttl_days` to positive day counts; `null` means no expiry.
- Explicit policy-validated atomic memory writes are admitted directly; historical backfill requires review.
- Approved candidates and built-in Hermes memory writes become durable records.
- Assistant claims and raw tool output are never promoted as user facts.
- Secret-like content, `.env`, private keys, unsupported/binary files, oversized files, and paths outside the current project root are rejected before embedding.

## Retrieval

Hybrid ranking combines semantic similarity, FTS5 exact matches, importance, freshness, and a durable-memory boost. Low-scoring matches are omitted. Context injection includes provenance, limits repeated sources, and labels all retrieved text as untrusted data rather than instructions.

## Agent tools

`local_rag_remember`, `local_rag_search`, `local_rag_status`, `local_rag_forget`, `local_rag_review`, `local_rag_approve`, `local_rag_reject`, `local_rag_index_file`, `local_rag_prune`, `local_rag_index_image`, `local_rag_search_images`, `local_rag_forget_image`.

## Maintenance CLI

From the Hermes checkout:

```sh
PYTHONPATH="$HOME/.hermes/plugins" venv/bin/python -m local_rag.cli status
PYTHONPATH="$HOME/.hermes/plugins" venv/bin/python -m local_rag.cli --namespace default:local search "query"
```

Raw transcript backfill is disabled. Historical sessions must first undergo selective semantic extraction into normalized memory items; raw user turns, assistant output, and extractive summaries are never indexed automatically.

## Recovery

Hermes includes `local-rag/` because it lives inside `HERMES_HOME`; no external `backup_paths()` entry is needed. Active stores use the `.db` suffix, so Hermes snapshots them with SQLite's backup API instead of copying a live WAL database as an ordinary file. Schema upgrades create `memory.pre-v2.db` before the first v2 migration. A changed embedding dimension stops startup with an explicit reindex error instead of corrupting existing vectors.

When upgrading from a release that used `memory.sqlite` or `visual.sqlite`, stop every Hermes Desktop and gateway process before starting the new version. Startup then migrates committed main-file and WAL data into `.db`, verifies integrity, durably installs the new database, and removes the legacy database and sidecars. Old plugin processes must not remain alive during this one-time filename migration because they do not participate in the new migration lock.
