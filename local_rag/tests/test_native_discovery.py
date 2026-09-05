from pathlib import Path

import pytest


def test_native_directory_loader_discovers_provider():
    memory = pytest.importorskip("plugins.memory", reason="Requires installed Hermes core")
    _is_memory_provider_dir = memory._is_memory_provider_dir
    _load_provider_from_dir = memory._load_provider_from_dir
    root = Path(__file__).resolve().parents[1]
    assert _is_memory_provider_dir(root)
    provider = _load_provider_from_dir(root)
    assert provider is not None
    assert provider.name == 'local_rag'
