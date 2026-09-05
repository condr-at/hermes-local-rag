# Shared local inference and curated image memory — 1.4.0

Publishing this release does not install or activate it in an existing Hermes
session. Upgrade deliberately; no launchctl, gateway restart, Desktop restart,
process supervisor or automatic model download is added. The first embedding request
starts the shared daemon on demand, with cross-process startup and lifetime locks.
Discovery and status checks do not start it. A healthy daemon is never restarted.

## Running the service deliberately

Use the same reviewed checkout/installed package and local model artifacts as the
clients. From an environment with this package's dependencies:

```sh
python -m local_rag.inference --home "$HERMES_HOME"
```

If `HERMES_HOME` is not set, omit `--home` to use the default `~/.hermes`.
The process runs in the foreground; startup prints a JSON `ready` line.
Ctrl-C/SIGTERM stops **this service only**. Stop/start it deliberately, never with
scripts that stop/relaunch Desktop or the gateway. A second service for the same
home fails before replacing the socket/token. A crash releases the OS flock;
the next deliberate start or embedding request can recover the stale socket and
rotate the token. Provider shutdown does not stop the shared daemon: it remains
resident even after Desktop and gateway exit, until explicitly stopped.

Desktop and gateway providers, CLI indexing/backfill, and compatibility embedder
factories use `InferenceClient`, not resident model copies. There is no local
fallback when the daemon is absent. Provider shutdown closes its database handles,
not the shared service. Standalone daemon/storage imports do not require Hermes
core; `local_rag.provider` does, and is imported lazily by the package entry point.

### Resource and security contracts

- One lazy EmbeddingGemma instance (768d output truncated and renormalized for
  128/256/512/768d clients), one lazy CLIP instance with its text/vision ONNX
  sessions. A single worker serializes both, including mixed client concurrency.
- Two bounded admission classes (32 queued requests each) share a priority heap:
  interactive queries before background indexing, FIFO within each class. Queue
  overflow is explicit. Indexing cannot consume interactive queue capacity.
- 48 concurrent connection handlers maximum; 256 KB protocol frame cap, 32K text
  input cap, 2s initial receive deadline, default 30s inference deadline, maximum
  120s accepted. Interactive priority is **not preemption** of a native model call.
- Expired queued work is skipped. A native LiteRT/ONNX call cannot safely be
  cancelled mid-execution: a timed-out caller returns, but the existing worker
  finishes that call. Shutdown retains the flock until the worker really exits;
  a wedged native call can therefore delay shutdown. There is no automatic
  replacement worker/model or restart loop. Sustained interactive traffic can
  starve indexing; fairness is not yet implemented.
- Socket and rotating bearer token under `$HERMES_HOME/local-rag/.cache/`, private
  directory 0700 and files/socket 0600. Native backup excludes `.cache`, avoiding
  socket errors and restoration of live credentials/locks. Same Unix user is the
  security principal; this is not a sandbox against malicious same-user processes.
- Profile paths must fit the platform Unix-socket path limit (104 bytes on macOS);
  too-long paths fail explicitly rather than silently sharing another profile.
- Existing installed weights are read from the original model locations
  (`HERMES_EMBEDDINGGEMMA_DIR`/default `~/.hermes/models/embeddinggemma-litert`,
  `~/.hermes/models/clip-onnx`). Separate per-profile weight discovery/download UI
  is not part of this change. Model objects, runtime sockets, and memories are
  separated by active profile even when immutable model files are shared.

## Curated images, not attachment capture

`local_rag_index_image` now requires explicit `decision`, `reason`, and `scope`:

```json
{
  "path": "/current/project/approved-logo.png",
  "decision": "save",
  "reason": "Approved brand reference for future design work",
  "scope": "project",
  "group": "brand-logo",
  "description": "Approved ACME NORTH logo, revision B"
}
```

`decision=skip` stores no image. Missing approval metadata is rejected. Transient
authentication/error screenshots should normally be skipped. The provider never
captures attachments automatically or saves raw conversation turns. Source paths
remain confined to the active project root; arbitrary conversation attachment
paths outside that boundary are **not yet admitted**. There is no new temporary
attachment store. Non-primary agent contexts cannot save/delete curated images.

The approved original bytes are copied into `$HERMES_HOME/local-rag/images/` under
a SHA-256 filename (0700 directory, 0600 files). Original-file disappearance does
not break recall. Actual decoded format and size/pixel limits are checked. Image
metadata and all derivatives live together in `curated-images.db`, not detached
text-memory records. Relative filenames allow restoration to a different home.

- Byte deduplication is global within the profile; logical deduplication is per
  user namespace, scope, scope identity, version group and hash.
- Changed bytes with the same explicit `group` create a new version and deactivate
  the earlier version, atomically in SQLite. Without a group, the hash is the group.
- Default image search and unified prefetch return active versions only. Explicit
  `local_rag_search_images(include_history=true)` also returns earlier versions,
  still constrained by the same scope. Saving identical bytes is idempotent and
  does not reactivate an older version. There is no separate activation UI yet.
- Global/project/session visibility is checked before scoring both visual and
  OCR/description embeddings. Namespace boundaries always apply.
- `local_rag_forget_image` removes the image row **and its OCR, description and
  both vectors and the co-transactional original BLOB**. Managed files are removed only when their last logical reference
  is deleted. Deleting an active version does not resurrect an older one. Existing
  backup archives are historical snapshots and are not retroactively erased.
  Deletion checks the canonical runtime project/session, not tool-supplied scope.
  Row/derivative deletion commits before blob cleanup. A durable GC queue retries
  failed unlinks; each cleanup rechecks references under `BEGIN IMMEDIATE` so a
  concurrent process saving the same bytes cannot lose its blob.
- Legacy `visual.db` path-only rows are preserved, but not automatically promoted
  into curated recall: automatic copying would contradict selective admission.
  The old dashboard still concerns text memory/setup; no curated-image gallery or
  bulk migration UI is included. Re-save selected legacy images explicitly.

### Bounded reconciliation of managed originals

Image retrieval in primary contexts checks **one already-curated, visible row per
60 seconds per profile**, with the throttle and oldest-checked ordering in SQLite
(shared by Desktop/gateway processes). There is no directory admission scan,
attachment capture, source-file watcher, new cron, or automatic model reindex.
Secondary/read-only contexts do not reconcile. The approved external source is a
snapshot input: editing/deleting it does not change the durable managed original;
explicitly save it again to retain an additional version.

- Changed managed bytes are rehashed, decoded, locally re-OCRed and re-embedded
  from a private temporary snapshot, then moved to their new content-addressed
  filename. The same logical row keeps its trusted namespace/scope/group/active
  bit; derivatives of overwritten bytes are replaced, not retained as false
  history. Explicit saves of changed sources still create version history.
- Missing/invalid/symlink managed originals remove the associated row and all
  derivatives. Older inactive versions are never automatically reactivated.
- Unchanged bytes do not repeat OCR/embedding. Unrelated files are never admitted
  or deleted. Maintenance processes only rows visible to the current runtime
  scope; shared bytes remain until the last logical reference is gone.
- Pending or failed reindex cannot recall unchecked changed pixels: the bounded
  top retrieval candidates are verified against their stored hash before return.
- A maintenance pass may include one OCR/model job with existing inference/OCR
  timeouts. It is synchronous, not a guarantee of sub-second prefetch latency.
  Large collections converge over successive retrievals, not in one full scan.

OCR, description, approval reason and group use the text ingestion policy's
secret/injection blocking patterns; short image labels remain allowed. Blocked
saves leave no image row or newly created managed blob, and blocked changed rows
are removed. No blocked OCR is embedded. OCR is imperfect: unreadable/unrecognized
secrets cannot be detected, and unavailable OCR remains explicitly reported.

## OCR and retrieval

Local OCR uses installed Tesseract (`tesseract IMAGE stdout`, default installed
language) or macOS Vision via the bundled `ocr.swift` and installed Swift command
line tools. It has a 30s subprocess timeout and records an explicit unavailable/
failed status. Swift compiler cache is under the local RAG `.cache`; there is no
network OCR/model service. Other platforms need Tesseract and appropriate language
packs; install separately. No OCR engine is fabricated or silently emulated.

OCR text plus an explicitly supplied description is embedded with EmbeddingGemma.
Image retrieval combines CLIP text/image cosine, OCR/description semantic cosine
and lexical term coverage, so entities in table cells are searchable beyond CLIP's
visual semantics. It does **not** reconstruct a table schema, guarantee exact cell
coordinates, or guarantee OCR accuracy/language coverage. Descriptions are supplied
context, not an automatically generated vision-model caption. No caption API is
called.

`local_rag_search` and automatic prefetch merge text and image ranked lists by
reciprocal rank (not by comparing incompatible raw CLIP/text scores). Image results
retain their managed `path`, `kind=image`, scope, version and derivative metadata.
Use `local_rag_forget_image` for these IDs: image IDs and text IDs are separate.

### Native vision integration: verified boundary

Read-only inspection of the installed Hermes source found:

- `agent/memory_provider.py`: `prefetch(...) -> str`, `handle_tool_call(...) -> str`.
- `agent/memory_manager.py:525–545`: automatic recall calls `.strip()` on provider
  output and joins text strings. This hook cannot deliver pixel content blocks.
- `tools/vision_tools.py:774+`: the native `vision_analyze` fast path returns a
  multimodal tool-result envelope for supported provider/model combinations.
- `agent/tool_dispatch_helpers.py:347+`: multimodal envelopes are dicts, not JSON
  strings. General tool dispatch can handle them, but that is not a multimodal
  **prefetch** contract.

Therefore this plugin returns paths and tells the agent to use native
`vision_analyze` to load pixels. It does **not** claim automatic pixel injection.
No core changes or private-envelope hacks were made. Automatic image prefetch
would need a supported generic multimodal recall API. An actual external vision
model request was not made as part of repository verification.

## Backup and remaining operational limits

Native `hermes_cli.backup.run_backup` was exercised against a temporary home while
the test inference service was live. Its zip contained original managed images
and a SQLite snapshot of `curated-images.db`, excluded runtime/cache files, and
restored searchable OCR plus valid relocated image paths. No production backup or
production memory mutation was performed.

### Self-contained image snapshots — no backup quiesce required

`curated-images.db` now stores verified original bytes in the same row/transaction
as scope, history, hash, OCR and vectors. Native SQLite backup is self-contained
while saves/deletes/reconciliation continue; **backup never requires quiescing**.
`images/` remains user-visible and is still included in native archives. Its files
can reflect a different instant from the DB (or be absent). The DB is authoritative
**only during explicit recovery**, not an excuse to resurrect normal user deletes.
A concurrent deletion may cause native backup to report a skipped managed file /
"Backup incomplete"; the successfully snapshotted image DB remains recoverable.
This is not a global atomic snapshot across all Hermes databases.

On first opening an older DB, migration imports only managed files whose format,
filename and SHA256 agree with the row. The entire backfill rolls back if any
legacy original is missing, tampered or a symlink. Supply that intact original
and retry; an old archive lacking both original bytes and file cannot be repaired.
The self-contained guarantee applies after this migration succeeds, not to old
archives or a DB that has never been opened by the upgraded plugin.

### Exact recovery route (including DB-only native snapshots)

Restore/import the native archive into an **inactive destination home**, or place
its `local-rag/curated-images.db` at that relative path in a new destination home.
Before opening that restored home in Desktop/gateway/provider, run from this repo:

```sh
python -m local_rag.cli \
  --home /absolute/path/to/restored-home restore-images
```

This owner-operated command covers **all namespaces/scopes/history** in the chosen
DB; `--namespace` does not restrict recovery. It verifies all originals before
writing, then uses private 0600 staging, fsync and atomic replacement. DB originals
replace missing or differing managed copies, without OCR, embedding, model startup,
or raw dialogue ingestion. It never adopts extra archive files into the index.
If filesystem IO fails partway, already materialized files are complete and safe;
fix the error and rerun (idempotent). Do not use this recovery command as routine
maintenance: normal managed-file deletion still removes its scoped row and original
BLOB during bounded reconciliation. Start the restored provider **after** recovery,
otherwise normal missing-file reconciliation can intentionally remove those rows.
This destination-only recovery ordering is not a backup quiesce requirement.

### Explicit derivative refresh

After installing OCR or changing embedding models/dimensions, refresh each approved
image ID (including history IDs) with its actual namespace and relevant scope key:

```sh
python -m local_rag.cli \
  --home /absolute/path/to/home --namespace 'actual-namespace' \
  reindex-image 42 --project /absolute/project/path --description 'Updated caption'
# For session scope use --session SESSION_ID; global needs neither scope option.
# Omit --description to retain the caption. Repeat for each affected ID.
```

Reindex checks visibility and original integrity, regenerates OCR/vectors, and
atomically updates derivatives while preserving ID, original, creation time and
active/history state. Failure leaves the old row intact. It does not revive a
missing/edited managed file; use ordinary reconciliation for edits and explicit
recovery only for restored snapshots. Identical-byte save remains idempotent;
reindex is the deliberate update path. Dimension mismatches remain explicit until
those rows are reindexed. Unified retrieval isolates branch failures: working text
results survive CLIP/image outages, with `warnings` in search JSON and a degraded
notice in prefetch (and conversely image results survive text branch failures).

A hard process crash may leave a private cache staging file or complete unreferenced
managed file; no arbitrary-directory orphan scan adopts it. Caches are excluded
from backup; failed ordinary writes clean staging before returning. DB is 0600,
managed/cache directories 0700. SQLite secure-delete is enabled for current row
removal, but this is not forensic erasure of old archives, filesystem snapshots or
storage media. Committed deletes/replacements retain the durable scoped GC queue.

## Reproducible verification (repository only)

For a separate development environment with a read-only Hermes core checkout:

```sh
UV_PROJECT_ENVIRONMENT=.venv-upgrade uv sync --group dev
# PYTHONPATH must not contain another Python version's site-packages.
PYTHONPATH=/path/to/read-only/hermes-agent .venv-upgrade/bin/python -m pytest --basetemp=.test-tmp
PYTHONPATH=/path/to/read-only/hermes-agent .venv-upgrade/bin/python -m pytest -s local_rag/tests/test_shared_e2e.py
```

Image-gap regression verification also ran on the actual Hermes Python 3.11.15:

```sh
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/venv/bin/python -m pytest local_rag/tests -q -ra
```

`test_image_reconciliation.py` covers policy rejection without retained blobs,
changed/missing/invalid managed images, stale-recall suppression, scope authority,
persistent bounded maintenance, inactive version preservation, failed commits,
failed fsync/unlink, and a real separate-process save between delete and GC.

Native integration tests use temporary homes only. Installed-weight tests skip
when weights are absent; OCR tests skip when no local engine is installed. The
subprocess tests exercise singleton contention, concurrent clients, crash recovery,
token rotation, profile separation and SIGTERM cleanup. Tests with controlled
backend vectors exercise queue saturation, reserved interactive capacity, priority
and deadlines; separate tests exercise real LiteRT, ONNX and Apple Vision OCR.
