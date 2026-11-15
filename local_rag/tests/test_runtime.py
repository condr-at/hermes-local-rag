"""Real child execution: dependency injection is not a virtual environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from local_rag import runtime


@pytest.mark.parametrize('mode', ['-c', '-m'])
def test_bootstrapped_child_sees_dependency_with_bare_python(tmp_path, monkeypatch, mode):
    core = tmp_path / 'core'
    deps = tmp_path / 'selected-dependencies'
    core.mkdir()
    deps.mkdir()
    (deps / 'injected_dependency.py').write_text('VALUE = "visible"\n')
    (deps / 'probe.py').write_text('import injected_dependency, json, sys; print(json.dumps([injected_dependency.VALUE, sys.argv[1:]]))')
    bootstrap = core / 'hermes_bootstrap.py'
    bootstrap.write_text(f'import sys\nsys.path.insert(0, {str(deps)!r})\n')
    monkeypatch.setitem(sys.modules, 'hermes_bootstrap', types.SimpleNamespace(__file__=str(bootstrap)))
    bare = sys._base_executable
    monkeypatch.setattr(sys, 'executable', bare)
    env = {k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'PYTHONHOME'}}
    missing = subprocess.run([bare, '-I', '-c', 'import injected_dependency'], env=env, capture_output=True)
    assert missing.returncode != 0
    target = 'import probe' if mode == '-c' else 'probe'
    result = subprocess.run(runtime.python_command(mode, target, 'a b', 'quote\'value'), cwd=tmp_path,
                            env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ['visible', ['a b', "quote'value"]]
    assert str(deps) not in repr(runtime.python_command(mode, target))


def test_standalone_retains_interpreter(monkeypatch):
    monkeypatch.delitem(sys.modules, 'hermes_bootstrap', raising=False)
    assert runtime.python_command('-m', 'some_module', '--arg') == [sys.executable, '-m', 'some_module', '--arg']


def test_cli_prefers_public_sdk(monkeypatch):
    module = types.ModuleType('hermes_cli.chat_launch')
    module.hermes_command = lambda args: ['/sdk/hermes', *args]
    monkeypatch.setitem(sys.modules, 'hermes_cli.chat_launch', module)
    assert runtime.hermes_command(['memory', 'status']) == ['/sdk/hermes', 'memory', 'status']


@pytest.fixture
def native_tmp():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='rag-native-', dir='/tmp') as directory:
        yield Path(directory)


@pytest.mark.parametrize("proof", ["imports", "daemon"])
def test_fresh_native_host_and_child(native_tmp, proof):
    """Opt-in integration against actual Hermes core, with synthetic selected state.

    No install, production profile, model loading, or production state writes.
    """
    tmp_path = native_tmp
    core_value = os.environ.get('LOCAL_RAG_TEST_HERMES_CORE')
    if not core_value:
        pytest.skip('Set LOCAL_RAG_TEST_HERMES_CORE for native bootstrap integration')
    import hashlib
    core = Path(core_value).resolve()
    home = tmp_path / 'home'
    key = hashlib.sha256(str(core).encode()).hexdigest()[:16]
    state = home / 'installs' / key
    venv = state / 'environments' / 'synthetic' / 'venv'
    site = venv / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    site.mkdir(parents=True)
    (venv / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    (state / 'facts.json').write_text(json.dumps({'packages': {'venv': {'environment': str(venv)}}}))
    (site / 'injected_dependency.py').write_text('VALUE = "native-visible"\n')
    # Test fixture only: borrow installed test dependencies, never production state.
    import sysconfig
    (site / 'test-dependencies.pth').write_text(sysconfig.get_path('purelib') + '\n')
    repo = Path(__file__).resolve().parents[2]
    script = f'''
import sys, subprocess, json
sys.path.insert(0, {str(core)!r})
import hermes_bootstrap
sys.path.insert(0, {str(repo)!r})
import injected_dependency
from local_rag.runtime import python_command, hf_command
assert sys.prefix == sys.base_prefix
from local_rag.dashboard import plugin_api
code = 'import injected_dependency; from local_rag.dashboard import plugin_api; print(injected_dependency.VALUE)'
result = subprocess.run(python_command('-c', code), check=True, capture_output=True, text=True)
assert result.stdout.strip() == 'native-visible'
subprocess.run(hf_command('--help'), check=True, capture_output=True, text=True)
plugin_api._hermes_launcher = lambda: None
plugin_api._auth_status = lambda: dict(available=False, authenticated=False)
assert plugin_api._status()['dependencies_installed'] is True
if {proof!r} == 'daemon':
    from local_rag.inference import InferenceClient
    import os, signal
    from pathlib import Path
    client = InferenceClient(Path({str(home)!r}), startup_timeout=5)
    try:
        client.ensure_started()
    except RuntimeError:
        log = (Path({str(home)!r}) / 'local-rag/.cache/inference.log').read_text()
        if 'self.listener.bind' in log and 'PermissionError: [Errno 1] Operation not permitted' in log:
            print('native-daemon-socket-denied')
            raise SystemExit(0)
        raise
    else:
        pid = client.request('ping')['pid']
        os.kill(pid, signal.SIGTERM)
print('native-parent-and-child-imports-pass')
'''
    env = {k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}}
    env.update(HOME=str(home), HERMES_HOME=str(home), HERMES_DISABLE_LAZY_INSTALLS='1')
    result = subprocess.run([sys._base_executable, '-I', '-c', script], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    if result.stdout.strip() == 'native-daemon-socket-denied':
        pytest.skip('Native daemon launched, but sandbox denied Unix socket bind (EPERM)')
    assert result.stdout.strip() == 'native-parent-and-child-imports-pass'
