#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from local_rag.runtime import hermes_command


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
    parser.add_argument("--skip-models", action="store_true", help="Code-only setup; model downloads are completed in the restarted dashboard")
    parser.add_argument("--check", action="store_true", help="Check installation artifacts only (not runtime health)")
    args = parser.parse_args()

    home = Path(args.home).expanduser().resolve()
    try:
        launcher = hermes_command()
    except FileNotFoundError:
        raise SystemExit("Hermes launcher not found on PATH. Install Hermes Agent first.") from None
    # Pin even the default profile: Hermes otherwise follows active_profile.
    profile = home.name if home.parent.name == "profiles" else "default"
    command = [*launcher, "--profile", profile]
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env["HERMES_HOME"] = str(home)

    plugin_target = home / "plugins" / "local_rag"
    text_target = home / "models" / "embeddinggemma-litert"
    visual_target = home / "models" / "clip-onnx"
    required = [plugin_target / "plugin.yaml", text_target / TEXT_FILES[0], visual_target / VISUAL_FILES[0], visual_target / VISUAL_FILES[1]]
    if args.check:
        missing = [str(path) for path in required if not path.exists()]
        # Native memory status is informational and exits zero when unavailable.
        if missing:
            raise SystemExit("Missing installation artifacts:\n" + "\n".join(missing))
        print("Installation artifacts present. Run the Local RAG dashboard health check after restarting Hermes.")
        return 0

    repository = Path(__file__).resolve().parent
    plugin_target.parent.mkdir(parents=True, exist_ok=True)
    # Backups and staging must stay outside Hermes' plugin discovery tree.
    upgrades = home / "local-rag-upgrades"
    upgrades.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="install-", dir=upgrades))
    temporary = backup / "staging"
    shutil.copytree(repository / "local_rag", temporary, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    legacy = plugin_target.with_name("local_rag.previous")
    if legacy.exists():
        legacy.replace(backup / "legacy-plugin")
    if plugin_target.exists():
        plugin_target.replace(backup / "plugin")
    temporary.replace(plugin_target)

    run([*command, "plugins", "enable", "local_rag"], env=env)
    print("Plugin copied and native enable completed. Restart Hermes and its dashboard to use the admitted runtime.")
    if not args.skip_models:
        print("Open Local RAG in the restarted dashboard to accept model terms, sign in, download models, and activate the provider.")
    print("If dependencies remain unavailable after restart, run `hermes pm install` for this profile, then retry enable and restart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
