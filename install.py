#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


TEXT_REPO = "litert-community/embeddinggemma-300m"
TEXT_FILES = ["embeddinggemma-300M_seq512_mixed-precision.tflite"]
TOKENIZER_REPO = "google/embeddinggemma-300m"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
VISUAL_REPO = "Xenova/clip-vit-base-patch32"
VISUAL_FILES = [
    "onnx/vision_model_quantized.onnx",
    "onnx/text_model_quantized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "preprocessor_config.json",
]


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Hermes Local RAG without touching the Hermes checkout")
    parser.add_argument("--home", default=os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    parser.add_argument("--skip-models", action="store_true", help="Install code and dependencies without downloading model artifacts")
    parser.add_argument("--check", action="store_true", help="Only verify an existing installation")
    args = parser.parse_args()

    home = Path(args.home).expanduser().resolve()
    checkout = home / "hermes-agent"
    python = Path(os.environ.get("HERMES_PYTHON", checkout / "venv" / "bin" / "python"))
    hermes = shutil.which("hermes")
    if not python.is_file() or not hermes:
        raise SystemExit("Hermes installation not found. Install Hermes Agent first or set HERMES_HOME/HERMES_PYTHON.")

    plugin_target = home / "plugins" / "local_rag"
    text_target = home / "models" / "embeddinggemma-litert"
    visual_target = home / "models" / "clip-onnx"
    required = [plugin_target / "plugin.yaml", text_target / TEXT_FILES[0], visual_target / VISUAL_FILES[0], visual_target / VISUAL_FILES[1]]
    if args.check:
        missing = [str(path) for path in required if not path.exists()]
        run([hermes, "memory", "status"])
        if missing:
            raise SystemExit("Missing installation artifacts:\n" + "\n".join(missing))
        env = {**os.environ, "PYTHONPATH": str(home / "plugins")}
        run([str(python), "-c", "from local_rag import LocalRagProvider; assert LocalRagProvider().is_available()"], env=env)
        print("Hermes Local RAG installation is healthy.")
        return 0

    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required because Hermes venvs may not include pip")
    repository = Path(__file__).resolve().parent
    run([uv, "pip", "install", "--python", str(python), str(repository)])

    plugin_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = plugin_target.with_name("local_rag.installing")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(repository / "local_rag", temporary, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if plugin_target.exists():
        backup = plugin_target.with_name("local_rag.previous")
        shutil.rmtree(backup, ignore_errors=True)
        plugin_target.replace(backup)
    temporary.replace(plugin_target)

    if not args.skip_models:
        hf = python.parent / "hf"
        if not hf.is_file():
            raise SystemExit("Hugging Face CLI was not installed into the Hermes venv")
        text_target.mkdir(parents=True, exist_ok=True)
        visual_target.mkdir(parents=True, exist_ok=True)
        try:
            run([str(hf), "download", TEXT_REPO, *TEXT_FILES, "--local-dir", str(text_target)])
            run([str(hf), "download", TOKENIZER_REPO, *TOKENIZER_FILES, "--local-dir", str(text_target)])
            run([str(hf), "download", VISUAL_REPO, *VISUAL_FILES, "--local-dir", str(visual_target)])
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "Model download failed. EmbeddingGemma is gated: accept Gemma Terms on Hugging Face and run `hf auth login`, then rerun install.py."
            ) from exc

    run([hermes, "config", "set", "memory.provider", "local_rag"])
    print("Installed. Restart the Hermes gateway or start a new Desktop session, then run: python install.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
