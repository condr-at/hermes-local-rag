# Changelog

## 1.6.1 — 2026-11-16

- Declare runtime dependencies in the directory plugin manifest as well as wheel metadata, so native Hermes dependency admission can restore them after an update.
- Replace hardcoded legacy virtualenv installation with native plugin enable/admission. Re-bootstrap Python children when running inside managed Hermes; standalone Python environments keep their ordinary launch behavior.
- Use the host launcher for backfill export and profile-specific commands, and the active dependency environment for model downloads and health probes.
- Keep installer staging and backups outside the plugin discovery directory; archive the legacy `local_rag.previous` copy instead of loading an old duplicate plugin.
- Clarify that installer `--check` verifies artifacts only, and that running Hermes processes need a restart after dependency admission.
- Add dependency, directory-import, bootstrap subprocess, and backup regression tests; run CI on Python 3.11 and 3.14.

## 1.6.0 — 2026-09-13

- Added an off-by-default memory diagnostics checkbox in Local RAG's own settings, not the host application.
- Completed responses can show provider prefetch, explicit retrieval, confirmed memory mutations, document/image operations, Markdown mirroring, and observed refusals or failures.
- Diagnostic output is deterministic Markdown, requires no additional model calls, and never turns raw chat into a memory corpus.
- Profile/session-isolated, bounded, escaped turn-local traces; no collection while disabled, no repeated forgotten content, and no model reload when toggling diagnostics.
- Expanded credential-header screening and re-screened legacy excerpts before displaying diagnostics. Duplicate image results and zero-chunk imports retain their uncertainty instead of claiming a new write.
- Connected exclusive memory-provider discovery to Hermes' native lifecycle hooks without modifying Hermes core. Added real-loader regression coverage and an isolated live-agent smoke test.


## 1.5.0

### Memory safety

- Added `local_rag_update`: correct a retrieved semantic item in place using its ID and exact `expected_text`. Identity, source and scope remain unchanged; omitted optional semantic fields are preserved.
- Validate and embed replacements before the write transaction, then recheck the original row under the transaction. Conflicts, embedding failures and SQLite failures preserve the existing fact and its FTS entry.
- Reject hidden duplicate collisions rather than rewriting or exposing another scope's item.
- Apply visibility checks to deletion. The standalone CLI derives project authority from its actual working directory and cannot acquire session authority from caller-supplied environment variables.
- Share semantic-field validation between saves and corrections, including metadata screening and finite-number checks.

### Retrieval and image evidence

- Default text semantic admission threshold is now 0.3, applied before rank fusion. This calibration is not a general answer-accuracy guarantee.
- Curated images can carry optional pixel-derived descriptions as retrieval hints. Captions and OCR are untrusted; exact visual claims require inspecting the selected image.
- Clarify evidence guidance: retrieved candidates and paths are not proof that the model inspected their pixels.

### Verification and boundaries

- Added deterministic regression coverage for atomic updates, conflicts, rollback, validation, scoped deletion, relevance and publication privacy.
- Local native-agent acceptance covered a nine-turn memory lifecycle and seven source-identity/table/multi-source visual scenarios. These are targeted checks, not an independent general production-readiness benchmark. Private native transcripts and memory databases are not distributed.
- `expected_text` is content compare-and-swap, not a monotonic revision token: an A→B→A change completed before preflight is not detectable by this API.
- The package remains beta. No automatic raw-dialogue or image-attachment ingestion was added.

### Upgrade

Back up local memory using SQLite's backup API, install `hermes-local-rag==1.5.0` in the environment running Hermes, and restart the relevant Hermes processes when convenient to activate the code. If using the separately installed Dashboard plugin, update that plugin as well. No model swap or Hermes core patch is required. For upgrades from 1.3.2 or older, follow the migration instructions in `UPGRADE.md` before starting new writers.
