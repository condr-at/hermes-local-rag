"""Dashboard/installer wiring; real child execution lives in test_runtime.py."""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from local_rag import cli
from local_rag.dashboard import plugin_api as api


def test_dashboard_uses_host_interpreter(monkeypatch):
    monkeypatch.setenv('HERMES_PYTHON', '/obsolete/venv/bin/python')
    assert api._python() == Path(sys.executable)
    assert api._hf()[:2] == [sys.executable, "-c"]
    assert "huggingface_hub.cli.hf" in api._hf()[2]


@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_export_uses_launcher_and_requested_home(tmp_path, monkeypatch, named, failure):
    home = tmp_path / 'profiles' / 'work' if named else tmp_path / 'root'
    monkeypatch.setenv('HERMES_HOME', '/wrong/profile')
    monkeypatch.setenv('PYTHONPATH', '/wrong/python')
    monkeypatch.setattr('shutil.which', lambda name: '/managed/hermes')
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if failure:
            raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(cli.subprocess, 'run', run)
    monkeypatch.setattr(cli, 'build_plan', lambda *a, **k: {'ok': True})
    args = argparse.Namespace(home=str(home), plan=str(tmp_path / 'plan.json'))
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            cli._build_from_trusted_host_export(args, model=None)
    else:
        assert cli._build_from_trusted_host_export(args, model=None) == {'ok': True}
    argv, kwargs = calls[0]
    assert argv[:5] == ['/managed/hermes', '--profile', 'work' if named else 'default', 'sessions', 'export']
    assert kwargs['env']['HERMES_HOME'] == str(home)
    assert 'PYTHONPATH' not in kwargs['env']
    assert not list((home / 'local-rag').glob('.backfill-export-*'))


def test_installer_admits_dependencies_without_host_pip(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location('rag_install', Path(__file__).parents[2] / 'install.py')
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    home = tmp_path / 'profiles' / 'test'
    monkeypatch.setattr(sys, 'argv', ['install.py', '--home', str(home)])
    monkeypatch.setattr(installer.shutil, 'which', lambda name: '/managed/hermes' if name == 'hermes' else None)
    calls = []
    monkeypatch.setattr(installer, 'run', lambda argv, **kwargs: calls.append((argv, kwargs)))
    assert installer.main() == 0
    assert (home / 'plugins/local_rag/plugin.yaml').is_file()
    assert calls == [(['/managed/hermes', '--profile', 'test', 'plugins', 'enable', 'local_rag'], {'env': calls[0][1]['env']})]
    assert calls[0][1]['env']['HERMES_HOME'] == str(home)
    assert 'PYTHONPATH' not in calls[0][1]['env']
    assert 'restart' in capsys.readouterr().out.lower()


def test_installer_check_is_explicitly_artifact_only(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location('rag_install_check', Path(__file__).parents[2] / 'install.py')
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    home = tmp_path / 'root'
    for relative in ['plugins/local_rag/plugin.yaml',
                     'models/embeddinggemma-litert/' + installer.TEXT_FILES[0],
                     *['models/clip-onnx/' + name for name in installer.VISUAL_FILES[:2]]]:
        path = home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(sys, 'argv', ['install.py', '--home', str(home), '--check'])
    monkeypatch.setattr(installer.shutil, 'which', lambda _: '/managed/hermes')
    calls = []
    monkeypatch.setattr(installer, 'run', lambda argv, **kwargs: calls.append((argv, kwargs)))
    assert installer.main() == 0
    assert calls == []
    assert 'healthy' not in capsys.readouterr().out
