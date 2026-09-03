from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embedder import get_shared_embedder
from .config import LocalRagConfig
from .service import LocalRagService
from .sessions import import_session_jsonl
from .store import MemoryStore


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
    args = parser.parse_args()

    db = Path(args.home).expanduser() / "local-rag" / "memory.sqlite"
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
