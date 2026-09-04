from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .embedder import get_shared_embedder
from .backfill import apply_plan_to_store, build_plan, plan_key
from .config import LocalRagConfig
from .database import canonical_database_path
from .service import LocalRagService
from .sessions import import_session_jsonl
from .store import MemoryStore


MEMORY_ITEMS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "maxItems": 50,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "kind", "scope", "subject", "durability", "importance", "confidence", "tags"],
                "properties": {
                    "text": {"type": "string", "minLength": 12, "maxLength": 2000},
                    "kind": {"type": "string", "enum": ["fact", "preference", "decision", "constraint", "skip"]},
                    "scope": {"type": "string", "enum": ["global", "project", "session"]},
                    "subject": {"type": "string", "minLength": 1, "maxLength": 200},
                    "durability": {"type": "string", "enum": ["transient", "ongoing", "stable"]},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "maxItems": 16, "items": {"type": "string", "maxLength": 64}},
                },
            },
        }
    },
}


def register_backfill_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="backfill_command", required=True)
    preview = sub.add_parser("preview", help="Extract a review plan without writing memory")
    preview.add_argument("--plan", required=True)
    preview.add_argument("--home", default=str(Path.home() / ".hermes"))
    apply = sub.add_parser("apply", help="Apply only explicitly accepted review items")
    apply.add_argument("--plan", required=True)
    apply.add_argument("--home", default=str(Path.home() / ".hermes"))


def _structured_model(llm):
    def model(messages: list[dict[str, str]]) -> str:
        result = llm.complete_structured(
            instructions=messages[0]["content"],
            input=[{"type": "text", "text": messages[1]["content"]}],
            json_schema=MEMORY_ITEMS_SCHEMA,
            schema_name="historical_memory_items",
            temperature=0,
            max_tokens=4000,
            timeout=120,
            purpose="selective_historical_memory_backfill",
        )
        if not isinstance(result.parsed, dict) or not isinstance(result.parsed.get("items"), list):
            raise ValueError("Structured extractor returned no validated items")
        return json.dumps(result.parsed["items"], ensure_ascii=False)
    return model


def _auxiliary_structured_model():
    """Use Hermes active provider/auth without creating an agent or exposing tools."""
    from agent.auxiliary_client import call_llm
    from jsonschema import ValidationError, validate

    def model(messages: list[dict[str, str]]) -> str:
        current = list(messages)
        last_error: Exception | None = None
        for attempt in range(2):
            response = call_llm(
                messages=current,
                temperature=0,
                max_tokens=4000,
                timeout=120,
                extra_body={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "historical_memory_items", "schema": MEMORY_ITEMS_SCHEMA, "strict": False},
                    }
                },
            )
            content = response.choices[0].message.content
            try:
                if not isinstance(content, str):
                    raise ValueError("Structured extractor returned no text")
                parsed = json.loads(content)
                validate(instance=parsed, schema=MEMORY_ITEMS_SCHEMA)
                return json.dumps(parsed["items"], ensure_ascii=False)
            except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as exc:
                last_error = exc
                if attempt == 0:
                    current = [
                        {**messages[0], "content": messages[0]["content"] + " Your previous response failed schema validation; obey every required field and limit."},
                        messages[1],
                    ]
        raise ValueError("Structured extractor failed schema validation twice") from last_error
    return model


def _build_from_trusted_host_export(args: argparse.Namespace, *, model) -> dict:
    home = Path(args.home).expanduser().resolve()
    directory = home / "local-rag"
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, export_name = tempfile.mkstemp(prefix=".backfill-export-", suffix=".jsonl", dir=directory)
    export = Path(export_name)
    os.fchmod(descriptor, 0o600)
    os.close(descriptor)
    executable = (home / "hermes-agent" / "hermes").resolve()
    python = home / "hermes-agent" / "venv" / "bin" / "python"
    if not executable.is_file() or not python.is_file():
        export.unlink(missing_ok=True)
        raise RuntimeError("Managed Hermes runtime was not found")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    try:
        subprocess.run([str(python), str(executable), "sessions", "export", str(export)], check=True, env=environment)
        return build_plan(export, Path(args.plan), model=model, signing_key=plan_key(home, create=True))
    finally:
        export.unlink(missing_ok=True)


def backfill_command(args: argparse.Namespace, *, llm) -> int:
    if args.backfill_command == "preview":
        result = _build_from_trusted_host_export(args, model=_structured_model(llm))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.backfill_command == "apply":
        result = apply_plan_to_store(Path(args.plan), hermes_home=Path(args.home))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    raise ValueError(f"Unsupported backfill command: {args.backfill_command}")


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Hermes memory-provider CLI discovery entrypoint."""
    register_backfill_cli(parser)


def local_rag_command(args: argparse.Namespace) -> int:
    if args.backfill_command == "preview":
        result = _build_from_trusted_host_export(args, model=_auxiliary_structured_model())
    elif args.backfill_command == "apply":
        result = apply_plan_to_store(Path(args.plan), hermes_home=Path(args.home))
    else:
        raise ValueError(f"Unsupported backfill command: {args.backfill_command}")
    print(json.dumps(result, ensure_ascii=False))
    return 0


def backfill_main() -> int:
    parser = argparse.ArgumentParser(prog="hermes-local-rag-backfill")
    register_backfill_cli(parser)
    return local_rag_command(parser.parse_args())


def main() -> int:
    parser = argparse.ArgumentParser(prog="local-rag", description="Inspect and maintain Hermes local RAG")
    parser.add_argument("--home", default=str(Path.home() / ".hermes"))
    parser.add_argument("--namespace", default="default:local")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    forget = sub.add_parser("forget")
    forget.add_argument("id", type=int)
    review = sub.add_parser("review")
    approve = sub.add_parser("approve")
    approve.add_argument("id", type=int)
    reject = sub.add_parser("reject")
    reject.add_argument("id", type=int)
    index = sub.add_parser("index-file")
    index.add_argument("path")
    index.add_argument("--root", required=True)
    sessions = sub.add_parser("import-sessions")
    sessions.add_argument("path")
    sessions.add_argument("--root", required=True)
    sub.add_parser("prune")
    backfill = sub.add_parser("backfill")
    register_backfill_cli(backfill)
    args = parser.parse_args()

    if args.command == "backfill":
        return local_rag_command(args)

    db = canonical_database_path(Path(args.home).expanduser() / "local-rag", "memory")
    config = LocalRagConfig.load(args.home)
    store = MemoryStore(db, dimensions=config.embedding_dimensions)
    try:
        if args.command == "status":
            output = {"database": str(db), "namespaces": store.namespaces()}
        elif args.command == "forget":
            output = {"removed": store.delete(args.namespace, args.id)}
        elif args.command == "review":
            output = {"candidates": store.list_candidates(args.namespace)}
        elif args.command == "reject":
            output = {"rejected": store.reject(args.namespace, args.id)}
        elif args.command == "prune":
            output = {"removed": store.prune(args.namespace)}
        else:
            embedder = get_shared_embedder(config.embedding_dimensions)
            roots = [Path(getattr(args, "root", Path.cwd())).expanduser().resolve()]
            service = LocalRagService(store=store, embedder=embedder, namespace=args.namespace, allowed_roots=roots)
            if args.command == "search":
                output = {"results": [item.__dict__ for item in service.search(args.query, limit=args.limit)]}
            elif args.command == "approve":
                candidates = {item["id"]: item for item in store.list_candidates(args.namespace)}
                item = candidates.get(args.id)
                output = {"promoted": bool(item) and store.promote(args.namespace, args.id, embedder.embed_document(item["text"]))}
            elif args.command == "index-file":
                output = {"chunks": service.index_path(args.path)}
            elif args.command == "import-sessions":
                output = {"chunks": import_session_jsonl(args.path, service)}
            else:
                raise RuntimeError(f"Unsupported command: {args.command}")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
