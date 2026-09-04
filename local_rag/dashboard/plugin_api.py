"""Dashboard setup API for Hermes Local RAG.

Long operations run in a single background job. The API reports only observed
filesystem/process state; it never treats starting a command as success.
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, SecretStr

from local_rag.backfill import plan_key, validate_plan_payload
from local_rag.policy import IngestDecision, classify_text

router = APIRouter()
_LOCK = threading.Lock()
_PLAN_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_PROGRESS_LOCK = _LOCK
_PROGRESS = _JOBS
_ACTIVE_KINDS: set[str] = set()

TEXT_REPO = "litert-community/embeddinggemma-300m"
TOKENIZER_REPO = "google/embeddinggemma-300m"
VISUAL_REPO = "Xenova/clip-vit-base-patch32"
TEXT_FILES = ["embeddinggemma-300M_seq512_mixed-precision.tflite"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
VISUAL_FILES = ["onnx/vision_model_quantized.onnx", "onnx/text_model_quantized.onnx", "tokenizer.json", "tokenizer_config.json", "config.json", "preprocessor_config.json"]
DEPENDENCIES = ["numpy>=2.0,<3", "tokenizers>=0.22,<1", "Pillow>=11,<13", "onnxruntime>=1.20,<2", "ai-edge-litert>=2.1,<3", "huggingface-hub>=0.34,<2", "jsonschema>=4,<5"]


def _home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser().resolve()


def _python() -> Path:
    configured = os.environ.get("HERMES_PYTHON")
    return Path(configured) if configured else _home() / "hermes-agent" / "venv" / "bin" / "python"


def _hf() -> Path:
    return _python().parent / "hf"


def _clean_env(*, include_plugin: bool = False) -> dict[str, str]:
    env = {key: os.environ[key] for key in ("HOME", "PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    env["HERMES_HOME"] = str(_home())
    if include_plugin:
        env["PYTHONPATH"] = str(_home() / "plugins")
    return env


def _run(command: list[str], job: dict[str, Any]) -> None:
    job["detail"] = "Running " + Path(command[0]).name + " " + " ".join(command[1:3])
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_clean_env())
    assert proc.stdout is not None
    for _line in proc.stdout:
        pass  # Never expose command output; it may contain tokens or user IDs.
    code = proc.wait()
    if code:
        raise RuntimeError(f"Command exited with status {code}. See job log for details.")


def _start(kind: str, worker) -> dict[str, str]:
    with _LOCK:
        if any(j["state"] == "running" for j in _JOBS.values()):
            raise HTTPException(409, "Another setup operation is already running")
        while len(_JOBS) >= 20:
            oldest = next(iter(_JOBS))
            if _JOBS[oldest]["state"] == "running":
                break
            _JOBS.pop(oldest)
        job_id = uuid.uuid4().hex
        job: dict[str, Any] = {"id": job_id, "kind": kind, "state": "running", "detail": "Starting…", "log": ""}
        _JOBS[job_id] = job

    def run() -> None:
        try:
            worker(job)
            with _LOCK:
                job.update(state="complete", detail="Completed successfully")
        except Exception as exc:
            with _LOCK:
                job.update(state="failed", detail=str(exc))

    threading.Thread(target=run, name=f"local-rag-{kind}", daemon=True).start()
    return {"job_id": job_id}


def _model_state() -> tuple[bool, bool]:
    models = _home() / "models"
    text = models / "embeddinggemma-litert"
    visual = models / "clip-onnx"
    return ((text / TEXT_FILES[0]).is_file() and all((text / f).is_file() for f in TOKENIZER_FILES), all((visual / f).is_file() for f in VISUAL_FILES))


def _auth_status() -> dict[str, Any]:
    hf = _hf()
    if not hf.is_file():
        return {"available": False, "authenticated": False, "guidance": "Install dependencies first, then sign in with the Hugging Face CLI."}
    try:
        result = subprocess.run([str(hf), "auth", "whoami"], capture_output=True, text=True, timeout=20, env=_clean_env())
        authenticated = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        authenticated = False
    return {"available": True, "authenticated": authenticated, "guidance": "Paste a read-only Hugging Face token below. It is handled locally and never returned or logged."}


def _status() -> dict[str, Any]:
    home = _home()
    text, visual = _model_state()
    config_path = home / "local-rag" / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    hermes = shutil.which("hermes")
    provider = None
    if hermes:
        try:
            check = subprocess.run([hermes, "config", "get", "memory.provider"], capture_output=True, text=True, timeout=20, env=_clean_env())
            if check.returncode == 0:
                provider = check.stdout.strip().strip('"')
        except (subprocess.TimeoutExpired, OSError):
            provider = None
    deps = False
    if _python().is_file():
        try:
            check = subprocess.run([str(_python()), "-c", "import numpy, tokenizers, PIL, onnxruntime, huggingface_hub"], capture_output=True, timeout=30, env=_clean_env())
            deps = check.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            # A path can exist but still be an invalid or stale executable.
            deps = False
    auth = _auth_status()
    db_path = home / "local-rag" / "memory.sqlite"
    database: dict[str, Any] = {"exists": db_path.is_file(), "healthy": None, "embedding_dimensions": None, "memory_count": None}
    if db_path.is_file():
        try:
            with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                database["healthy"] = "memories" in tables
                if "memories" in tables:
                    database["memory_count"] = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                if "meta" in tables:
                    row = conn.execute("SELECT value FROM meta WHERE key='embedding_dimensions'").fetchone()
                    database["embedding_dimensions"] = int(row[0]) if row else None
        except (sqlite3.Error, ValueError, OSError):
            database["healthy"] = False
    return {
        "platform": {"system": __import__("platform").system(), "machine": __import__("platform").machine()},
        "hermes_found": bool(hermes), "python_found": _python().is_file(),
        "dependencies_installed": deps, "hf": auth, "text_model_downloaded": text,
        "visual_model_downloaded": visual, "provider": provider, "active": provider == "local_rag",
        "config": {"embedding_dimensions": config.get("embedding_dimensions", 512), "episodic_ttl_days": config.get("episodic_ttl_days"), "summary_ttl_days": config.get("summary_ttl_days"), "visual_enabled": config.get("visual_enabled", False)},
        "database": database, "disk": {"free_bytes": shutil.disk_usage(home).free},
    }


class DownloadRequest(BaseModel):
    mode: str = "text"


class ConfigRequest(BaseModel):
    embedding_dimensions: int = 512
    retention: str = "forever"
    retention_days: float | None = None

class SetupModelsRequest(BaseModel):
    visual: bool = False
    terms_accepted: bool = False

class SetupConfigRequest(BaseModel):
    embedding_dimensions: int = 512
    episodic_ttl_days: float | None = None
    summary_ttl_days: float | None = None
    visual_enabled: bool = False

class ConfirmRequest(BaseModel):
    confirm: bool = False


class BackfillReviewRequest(BaseModel):
    revision: str
    accepted_indices: list[int]
    edits: dict[int, str] = Field(default_factory=dict)


class BackfillApplyRequest(BaseModel):
    confirm: bool = False
    revision: str


class AuthRequest(BaseModel):
    token: SecretStr


@router.get("/status")
@router.get("/setup/status")
def status():
    return _status()


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job")
        return dict(job)

@router.get("/setup/progress")
def setup_progress(job_id: str | None = None):
    with _LOCK:
        if job_id is not None:
            if job_id not in _JOBS:
                raise HTTPException(404, "Unknown job")
            return dict(_JOBS[job_id])
        return {"jobs": [dict(job) for job in list(_JOBS.values())[-20:]]}


@router.post("/setup/auth")
def setup_auth(request: AuthRequest):
    token = request.token.get_secret_value().strip()
    if not token.startswith("hf_") or len(token) < 10:
        raise HTTPException(422, "A valid Hugging Face access token is required")
    try:
        from huggingface_hub import HfApi, login

        identity = HfApi().whoami(token=token)
        role = ((identity.get("auth") or {}).get("accessToken") or {}).get("role")
        if role != "read":
            raise HTTPException(422, "Use a classic read-only Hugging Face token")
        login(token=token, add_to_git_credential=False)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "Hugging Face sign-in failed; verify the token and accepted model terms") from exc
    return {"authenticated": True}


@router.post("/setup/dependencies", status_code=202)
def install_dependencies():
    if not _python().is_file():
        raise HTTPException(409, "Hermes Python environment was not found")
    return _start("dependencies", lambda job: _run([str(_python()), "-m", "pip", "install", *DEPENDENCIES], job))


def download(request: DownloadRequest):
    if request.mode not in {"text", "visual"}:
        raise HTTPException(422, "mode must be text or visual")
    hf = _hf()
    if not hf.is_file():
        raise HTTPException(409, "Hugging Face CLI is unavailable; install dependencies first")

    def worker(job):
        text_dir = _home() / "models" / "embeddinggemma-litert"
        text_dir.mkdir(parents=True, exist_ok=True)
        _run([str(hf), "download", TEXT_REPO, *TEXT_FILES, "--local-dir", str(text_dir)], job)
        _run([str(hf), "download", TOKENIZER_REPO, *TOKENIZER_FILES, "--local-dir", str(text_dir)], job)
        if request.mode == "visual":
            visual_dir = _home() / "models" / "clip-onnx"
            visual_dir.mkdir(parents=True, exist_ok=True)
            _run([str(hf), "download", VISUAL_REPO, *VISUAL_FILES, "--local-dir", str(visual_dir)], job)
    return _start("download", worker)

@router.post("/setup/models", status_code=202)
def setup_models(request: SetupModelsRequest):
    if not request.terms_accepted:
        raise HTTPException(400, "Gemma Terms acceptance must be confirmed")
    return download(DownloadRequest(mode="visual" if request.visual else "text"))


def save_config(request: ConfigRequest):
    if request.embedding_dimensions not in {128, 256, 512, 768}:
        raise HTTPException(422, "embedding_dimensions must be 128, 256, 512, or 768")
    if request.retention not in {"forever", "custom"}:
        raise HTTPException(422, "retention must be forever or custom")
    days = None if request.retention == "forever" else request.retention_days
    if days is not None and days <= 0:
        raise HTTPException(422, "retention_days must be positive")
    if request.retention == "custom" and days is None:
        raise HTTPException(422, "retention_days is required for custom retention")
    path = _home() / "local-rag" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"embedding_dimensions": request.embedding_dimensions, "episodic_ttl_days": days, "summary_ttl_days": days}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"saved": True, "config": payload}

@router.post("/setup/config")
def setup_config(request: SetupConfigRequest):
    if request.embedding_dimensions not in {128, 256, 512, 768}:
        raise HTTPException(422, "embedding_dimensions must be one of 128, 256, 512, 768")
    if any(value is not None and value <= 0 for value in (request.episodic_ttl_days, request.summary_ttl_days)):
        raise HTTPException(422, "TTLs must be null or positive")
    path = _home() / "local-rag" / "config.json"
    db = _home() / "local-rag" / "memory.sqlite"
    if db.is_file():
        try:
            with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as conn:
                count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                row = conn.execute("SELECT value FROM meta WHERE key='embedding_dimensions'").fetchone()
            if count and (not row or int(row[0]) != request.embedding_dimensions):
                raise HTTPException(409, "Existing embeddings use another or unknown dimension; explicit reindex is required")
        except (sqlite3.Error, ValueError) as exc:
            raise HTTPException(409, "Could not verify existing embedding dimensions; repair or reindex is required") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = request.model_dump()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return {"saved": True, "config": payload, "reindex_performed": False}


def _backfill_plan_path() -> Path:
    return _home() / "local-rag" / "backfill-plan.json"


def _read_backfill_plan() -> dict[str, Any]:
    path = _backfill_plan_path()
    if not path.is_file() or path.stat().st_size > 20_000_000:
        raise HTTPException(404, "No selective backfill preview exists")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(409, "Selective backfill preview is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("items"), list):
        raise HTTPException(409, "Selective backfill preview is invalid")
    try:
        validate_plan_payload(payload, plan_key(_home(), create=False))
    except (OSError, ValueError) as exc:
        raise HTTPException(409, "Selective backfill preview failed trust validation") from exc
    return payload


def _plan_revision(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_during_backfill_preview() -> None:
    with _LOCK:
        if any(job["state"] == "running" and job["kind"] == "backfill-preview" for job in _JOBS.values()):
            raise HTTPException(409, "Backfill preview is running; retry after it completes")


@router.post("/setup/backfill/preview", status_code=202)
@router.post("/setup/backfill", status_code=202)
def setup_backfill_preview(request: ConfirmRequest):
    if not request.confirm:
        raise HTTPException(400, "Backfill preview requires explicit confirmation")
    hermes = shutil.which("hermes")
    if not hermes:
        raise HTTPException(409, "Hermes CLI was not found on PATH")

    def worker(job):
        folder = _home() / "local-rag"
        folder.mkdir(parents=True, exist_ok=True)
        plan = _backfill_plan_path()
        _run([hermes, "local_rag", "preview", "--plan", str(plan), "--home", str(_home())], job)
        if not plan.is_file():
            raise RuntimeError("Backfill preview command did not create a review plan")
        os.chmod(plan, 0o600)
    with _PLAN_LOCK:
        return _start("backfill-preview", worker)


@router.get("/setup/backfill/plan")
def get_backfill_plan():
    with _PLAN_LOCK:
        _reject_during_backfill_preview()
        payload = _read_backfill_plan()
        return {"created_at": payload.get("created_at"), "revision": _plan_revision(payload), "items": payload["items"]}


@router.put("/setup/backfill/plan")
def review_backfill_plan(request: BackfillReviewRequest):
    with _PLAN_LOCK:
        _reject_during_backfill_preview()
        payload = _read_backfill_plan()
        if request.revision != _plan_revision(payload):
            raise HTTPException(409, "Backfill plan changed; refresh before reviewing")
        accepted = set(request.accepted_indices)
        if any(not isinstance(index, int) or index < 0 or index >= len(payload["items"]) for index in accepted):
            raise HTTPException(422, "accepted_indices contains an unknown candidate")
        if any(index < 0 or index >= len(payload["items"]) for index in request.edits):
            raise HTTPException(422, "edits contains an unknown candidate")
        for index, text in request.edits.items():
            cleaned = " ".join(text.split())
            if len(cleaned) > 2000 or classify_text(cleaned) is not IngestDecision.INDEX:
                raise HTTPException(422, "edited candidate text is invalid or unsafe")
            payload["items"][index]["text"] = cleaned
        for index, item in enumerate(payload["items"]):
            if not isinstance(item, dict):
                raise HTTPException(409, "Selective backfill preview is invalid")
            item["accepted"] = index in accepted
        path = _backfill_plan_path()
        _write_private_json(path, payload)
        return {"reviewed": len(payload["items"]), "accepted": len(accepted), "revision": _plan_revision(payload)}


@router.post("/setup/backfill/apply", status_code=202)
def apply_backfill_plan(request: BackfillApplyRequest):
    if not request.confirm:
        raise HTTPException(400, "Backfill apply requires explicit confirmation")
    hermes = shutil.which("hermes")
    if not hermes:
        raise HTTPException(409, "Hermes CLI was not found on PATH")
    snapshot: Path | None = None
    def worker(job):
        assert snapshot is not None
        try:
            _run([hermes, "local_rag", "apply", "--plan", str(snapshot), "--home", str(_home())], job)
        finally:
            snapshot.unlink(missing_ok=True)

    with _PLAN_LOCK:
        _reject_during_backfill_preview()
        payload = _read_backfill_plan()
        if request.revision != _plan_revision(payload):
            raise HTTPException(409, "Backfill plan changed; refresh before applying")
        snapshot = _backfill_plan_path().with_name(f".backfill-apply-{uuid.uuid4().hex}.json")
        descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        try:
            return _start("backfill-apply", worker)
        except Exception:
            snapshot.unlink(missing_ok=True)
            raise


def activate():
    hermes = shutil.which("hermes")
    if not hermes:
        raise HTTPException(409, "Hermes CLI was not found on PATH")
    result = subprocess.run([hermes, "config", "set", "memory.provider", "local_rag"], capture_output=True, text=True, timeout=30, env=_clean_env())
    if result.returncode:
        raise HTTPException(500, "Activation failed; check Hermes configuration permissions")
    return {"active": True, "restart_required": True}

@router.post("/setup/activate")
def setup_activate(request: ConfirmRequest):
    if not request.confirm:
        raise HTTPException(400, "Activation requires explicit confirmation")
    return activate()


@router.post("/health")
def health():
    state = _status()
    problems = []
    if not state["dependencies_installed"]: problems.append("Runtime dependencies are unavailable")
    if not state["text_model_downloaded"]: problems.append("Text model files are incomplete")
    if not state["active"]: problems.append("memory.provider is not local_rag")
    provider_ok = False
    if not problems:
        env = _clean_env(include_plugin=True)
        try:
            check = subprocess.run([str(_python()), "-c", "from local_rag import LocalRagProvider; assert LocalRagProvider().is_available()"], capture_output=True, text=True, env=env, timeout=90)
            provider_ok = check.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            provider_ok = False
        if not provider_ok:
            problems.append("Provider availability check failed; inspect local Dashboard logs")
    return {"healthy": not problems and provider_ok, "provider_available": provider_ok, "problems": problems, "restart_required_after_activation": True}
