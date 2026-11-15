"""The directory plugin must retain the wheel's dependency contract."""
from pathlib import Path
import json
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_directory_plugin_declares_all_package_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    # JSON flow sequences are valid YAML and keep this check independent of Hermes.
    manifest = (ROOT / "local_rag" / "plugin.yaml").read_text()
    declaration = next((line.partition(":")[2].strip() for line in manifest.splitlines()
                        if line.startswith("python_dependencies:")), "[]")
    assert json.loads(declaration) == project["project"]["dependencies"]


def test_inference_imports_under_directory_loader_namespace():
    import subprocess
    import sys
    code = """
import importlib.util, importlib, sys
from pathlib import Path
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('_native_rag', root / '__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
importlib.import_module('_native_rag.inference')
assert 'local_rag' not in sys.modules
"""
    subprocess.run([sys.executable, '-I', '-S', '-c', code, str(ROOT / 'local_rag')], check=True, timeout=15)
