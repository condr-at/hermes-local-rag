from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import pytest
import huggingface_hub
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_rag.dashboard import plugin_api as api


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    home = tmp_path / "hermes"
    home.mkdir()
    python = home / "hermes-agent" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch(mode=0o755)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PYTHON", str(python))
    with api._LOCK:
        api._JOBS.clear()
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def wait(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        state = client.get("/setup/progress", params={"job_id": job_id}).json()
        if state["state"] in {"complete", "failed"}:
            return state
        time.sleep(0.01)
    raise AssertionError("setup job did not finish")


def test_status_projects_config_and_reports_disk_and_database(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Path(api.os.environ["HERMES_HOME"])
    folder = home / "local-rag"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"embedding_dimensions": 256, "token": "secret", "user_id": "private"}))
    with sqlite3.connect(folder / "memory.sqlite") as conn:
        conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES('embedding_dimensions','256')")
    monkeypatch.setattr(api, "_auth_status", lambda: {"available": False, "authenticated": False})
    monkeypatch.setattr(api.shutil, "which", lambda _: None)
    monkeypatch.setattr(api.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"returncode": 1, "stdout": "", "stderr": ""})())
    response = client.get("/setup/status")
    assert response.status_code == 200
    body = response.json()
    assert body["database"]["healthy"] is True
    assert body["disk"]["free_bytes"] > 0
    assert body["platform"]["system"]
    assert "secret" not in response.text and "private" not in response.text


def test_config_validates_ttls_and_never_reindexes(client: TestClient) -> None:
    assert client.post("/setup/config", json={"embedding_dimensions": 64}).status_code == 422
    assert client.post("/setup/config", json={"embedding_dimensions": 128, "episodic_ttl_days": 0}).status_code == 422
    response = client.post("/setup/config", json={"embedding_dimensions": 768, "episodic_ttl_days": None, "summary_ttl_days": 2.5, "visual_enabled": True})
    assert response.status_code == 200
    assert response.json()["reindex_performed"] is False
    path = Path(api.os.environ["HERMES_HOME"]) / "local-rag" / "config.json"
    assert json.loads(path.read_text())["embedding_dimensions"] == 768
    assert json.loads(path.read_text())["visual_enabled"] is True
    assert path.stat().st_mode & 0o777 == 0o600


def test_dimension_change_with_existing_data_requires_explicit_reindex(client: TestClient) -> None:
    db = Path(api.os.environ["HERMES_HOME"]) / "local-rag" / "memory.sqlite"
    db.parent.mkdir()
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO memories DEFAULT VALUES")
        conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES('embedding_dimensions','512')")
    response = client.post("/setup/config", json={"embedding_dimensions": 256})
    assert response.status_code == 409
    assert "reindex" in response.json()["detail"].lower()


def test_existing_data_without_dimension_metadata_fails_closed(client: TestClient) -> None:
    db = Path(api.os.environ["HERMES_HOME"]) / "local-rag" / "memory.sqlite"
    db.parent.mkdir()
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO memories DEFAULT VALUES")
    response = client.post("/setup/config", json={"embedding_dimensions": 512})
    assert response.status_code == 409
    assert "repair or reindex" in response.json()["detail"].lower()


def test_models_are_fixed_allowlist_and_visual_opt_in(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hf = tmp_path / "hf"
    hf.touch()
    monkeypatch.setattr(api, "_hf", lambda: hf)
    calls: list[list[str]] = []
    monkeypatch.setattr(api, "_run", lambda argv, job: calls.append(argv))
    assert client.post("/setup/models", json={"visual": False, "terms_accepted": False}).status_code == 400
    response = client.post("/setup/models", json={"visual": False, "terms_accepted": True})
    assert response.status_code == 202
    assert wait(client, response.json()["job_id"])["state"] == "complete"
    assert {call[2] for call in calls} == {api.TEXT_REPO, api.TOKENIZER_REPO}
    assert api.VISUAL_REPO not in {call[2] for call in calls}


def test_selective_backfill_preview_review_and_apply_are_separate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = Path(api.os.environ["HERMES_HOME"])
    commands: list[list[str]] = []
    monkeypatch.setattr(api.shutil, "which", lambda _: "/managed/hermes")

    def fake_run(command: list[str], job: dict) -> None:
        commands.append(command)
        if command[1:3] == ["local_rag", "preview"]:
            from local_rag.backfill import _seal_item, plan_key
            plan = Path(command[command.index("--plan") + 1])
            candidate = {
                "text": "The project uses SQLite.", "scope": "project", "subject": "database",
                "durability": "stable", "importance": 0.8, "confidence": 0.9, "tags": [],
                "namespace": "default:local", "project": "/work/project", "accepted": False,
                "provenance": {"origin": "historical-extraction", "session_id": "s1", "message_ids": [1]},
            }
            _seal_item(candidate, plan_key(home, create=True))
            plan.write_text(json.dumps({
                "schema_version": 1, "created_at": 1, "items": [candidate],
            }), encoding="utf-8")

    monkeypatch.setattr(api, "_run", fake_run)
    assert client.post("/setup/backfill/preview", json={"confirm": False}).status_code == 400
    preview = client.post("/setup/backfill/preview", json={"confirm": True})
    assert preview.status_code == 202
    assert wait(client, preview.json()["job_id"])["state"] == "complete"
    assert not list((home / "local-rag").glob(".backfill-export-*.jsonl"))
    assert not (home / "local-rag" / "memory.sqlite").exists()

    plan = client.get("/setup/backfill/plan")
    assert plan.status_code == 200
    assert plan.json()["items"][0]["accepted"] is False
    review = client.put("/setup/backfill/plan", json={
        "revision": plan.json()["revision"], "accepted_indices": [0],
        "edits": {"0": "The edited project memory remains reviewable."},
    })
    assert review.status_code == 200
    assert review.json()["reviewed"] == 1 and review.json()["accepted"] == 1
    assert client.put("/setup/backfill/plan", json={
        "revision": plan.json()["revision"], "accepted_indices": [], "edits": {},
    }).status_code == 409
    assert client.get("/setup/backfill/plan").json()["items"][0]["text"] == "The edited project memory remains reviewable."
    assert client.post("/setup/backfill/apply", json={"confirm": False, "revision": review.json()["revision"]}).status_code == 400
    apply = client.post("/setup/backfill/apply", json={"confirm": True, "revision": review.json()["revision"]})
    assert apply.status_code == 202
    assert wait(client, apply.json()["job_id"])["state"] == "complete"
    assert commands[-1][1:3] == ["local_rag", "apply"]
    assert ".backfill-apply-" in commands[-1][commands[-1].index("--plan") + 1]


def test_backfill_plan_operations_are_blocked_while_preview_runs(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(api.shutil, "which", lambda _: "/managed/hermes")

    def blocked_run(_command: list[str], _job: dict) -> None:
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(api, "_run", blocked_run)
    response = client.post("/setup/backfill/preview", json={"confirm": True})
    assert response.status_code == 202 and started.wait(timeout=2)
    assert client.get("/setup/backfill/plan").status_code == 409
    assert client.put("/setup/backfill/plan", json={"revision": "stale", "accepted_indices": [], "edits": {}}).status_code == 409
    assert client.post("/setup/backfill/apply", json={"confirm": True, "revision": "stale"}).status_code == 409
    release.set()
    assert wait(client, response.json()["job_id"])["state"] == "failed"


def test_activation_requires_confirmation(client: TestClient) -> None:
    assert client.post("/setup/activate", json={"confirm": False}).status_code == 400


def test_dependency_command_uses_managed_python_without_shell(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(api, "_run", lambda argv, job: calls.append(argv))
    response = client.post("/setup/dependencies")
    assert response.status_code == 202
    assert wait(client, response.json()["job_id"])["state"] == "complete"
    assert calls[0][:4] == [str(api._python()), "-m", "pip", "install"]
    assert calls[0][-len(api.DEPENDENCIES):] == api.DEPENDENCIES


def test_dashboard_auth_never_returns_or_logs_token(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", lambda self, token: {"auth": {"accessToken": {"role": "read"}}})
    monkeypatch.setattr(huggingface_hub, "login", lambda token, add_to_git_credential: captured.append(token))
    token = "hf_private-test-token-123"
    response = client.post("/setup/auth", json={"token": token})
    assert response.status_code == 200
    assert captured == [token]
    assert token not in response.text
    assert token not in json.dumps(api._JOBS)


def test_dashboard_auth_rejects_write_token(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", lambda self, token: {"auth": {"accessToken": {"role": "write"}}})
    assert client.post("/setup/auth", json={"token": "hf_write-token-123"}).status_code == 422


def test_legacy_mutating_routes_are_not_exposed(client: TestClient) -> None:
    assert client.post("/download", json={"mode": "text"}).status_code == 404
    assert client.put("/config", json={"embedding_dimensions": 256}).status_code == 404
    assert client.post("/backfill").status_code == 404
    assert client.post("/activate").status_code == 404


def test_health_never_returns_subprocess_output(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_status", lambda: {"dependencies_installed": True, "text_model_downloaded": True, "active": True})
    secret_output = "/Users/private/path token=hf_should_never_escape"
    monkeypatch.setattr(api.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"returncode": 1, "stdout": secret_output, "stderr": secret_output})())
    response = client.post("/health")
    assert response.status_code == 200
    assert response.json()["healthy"] is False
    assert secret_output not in response.text
    assert response.json()["problems"] == ["Provider availability check failed; inspect local Dashboard logs"]


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("python", 90), OSError("private path")])
def test_health_sanitizes_provider_launch_failures(client: TestClient, monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    monkeypatch.setattr(api, "_status", lambda: {"dependencies_installed": True, "text_model_downloaded": True, "active": True})

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(api.subprocess, "run", fail)
    response = client.post("/health")
    assert response.status_code == 200
    assert response.json()["healthy"] is False
    assert "private path" not in response.text


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("probe", 20), OSError("private probe path")])
def test_status_probes_fail_safely(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: Exception) -> None:
    hf = tmp_path / "hf"
    hf.touch()
    monkeypatch.setattr(api, "_hf", lambda: hf)
    monkeypatch.setattr(api.shutil, "which", lambda _: "/managed/hermes")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(api.subprocess, "run", fail)
    status = client.get("/setup/status")
    health = client.post("/health")
    assert status.status_code == 200
    assert health.status_code == 200
    assert status.json()["hf"]["authenticated"] is False
    assert health.json()["healthy"] is False
    assert "private probe path" not in status.text + health.text


def test_completed_job_history_is_bounded(client: TestClient) -> None:
    with api._LOCK:
        api._JOBS.update({str(index): {"id": str(index), "kind": "test", "state": "complete"} for index in range(20)})
    result = api._start("test", lambda job: None)
    wait(client, result["job_id"])
    with api._LOCK:
        assert len(api._JOBS) <= 20


def test_dashboard_ui_uses_protected_setup_contract() -> None:
    script = (Path(__file__).parents[1] / "dashboard" / "dist" / "index.js").read_text(encoding="utf-8")
    for route in ("/setup/status", "/setup/progress", "/setup/auth", "/setup/dependencies", "/setup/models", "/setup/config", "/setup/activate", "/setup/backfill/preview", "/setup/backfill/plan", "/setup/backfill/apply"):
        assert route in script
    assert "Raw transcripts are deleted after extraction" in script
    assert "Apply accepted memories" in script
    assert "Sign in from a terminal" not in script
    assert "hf auth login" not in script
    assert 'terms_accepted: acceptedTerms' in script
    assert 'visual_enabled: mode === "visual"' in script
    assert 'type: "radio"' in script
    assert 'name: "local-rag-model-mode"' in script
    assert 'name: "local-rag-retention"' in script
    assert "ChoiceButton" not in script
    source = (Path(__file__).parents[1] / "dashboard" / "src" / "index.js").read_text(encoding="utf-8")
    assert source == script


def test_dashboard_manifest_loads_setup_assets() -> None:
    dashboard = Path(__file__).parents[1] / "dashboard"
    manifest = json.loads((dashboard / "manifest.json").read_text(encoding="utf-8"))
    script = (dashboard / "dist" / "index.js").read_text(encoding="utf-8")

    assert manifest["name"] == "local_rag"
    assert manifest["tab"]["path"] == "/local-rag"
    assert manifest["entry"] == "dist/index.js"
    assert manifest["css"] == "dist/style.css"
    assert (dashboard / manifest["entry"]).is_file()
    assert (dashboard / manifest["css"]).is_file()
    assert f'register("{manifest["name"]}", Page)' in script
    assert f'const API = "/api/plugins/{manifest["name"]}"' in script
