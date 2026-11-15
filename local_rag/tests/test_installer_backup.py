"""Installer backups must not be discoverable as another local_rag plugin."""
import importlib.util
import sys
from pathlib import Path


def test_reinstall_archives_both_old_plugin_and_legacy_backup(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('rag_backup_install', Path(__file__).parents[2] / 'install.py')
    assert spec is not None and spec.loader is not None
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    home = tmp_path / 'home'
    for name, marker in [('local_rag', 'current'), ('local_rag.previous', 'legacy')]:
        directory = home / 'plugins' / name
        directory.mkdir(parents=True)
        (directory / 'plugin.yaml').write_text('name: local_rag\nversion: old\n')
        (directory / 'marker.txt').write_text(marker)
    monkeypatch.setattr(sys, 'argv', ['install.py', '--home', str(home), '--skip-models'])
    monkeypatch.setattr(installer.shutil, 'which', lambda _: '/managed/hermes')
    monkeypatch.setattr(installer, 'run', lambda *a, **k: None)
    assert installer.main() == 0
    assert list((home / 'plugins').glob('*/plugin.yaml')) == [home / 'plugins/local_rag/plugin.yaml']
    assert {p.read_text() for p in (home / 'local-rag-upgrades').rglob('marker.txt')} == {'current', 'legacy'}
