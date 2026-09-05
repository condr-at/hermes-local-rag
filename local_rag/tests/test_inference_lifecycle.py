"""Real daemon lifecycle tests; never load models or touch the user's profile."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from local_rag.inference import InferenceClient


@pytest.fixture
def homes():
    with tempfile.TemporaryDirectory(prefix='rag-life-', dir='/tmp') as root:
        paths = [Path(root) / 'a', Path(root) / 'b']
        yield paths
        for home in paths:
            try:
                pid = InferenceClient(home, timeout=.2).request('ping')['pid']
                os.kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while (home / 'local-rag/.cache/i.sock').exists() and time.monotonic() < deadline:
                    time.sleep(.02)
            except (OSError, RuntimeError, ValueError):
                pass


def demand(home):
    # Invalid dimensions are rejected before any real backend/model is loaded.
    client = InferenceClient(home, dimensions=1)
    with pytest.raises(RuntimeError, match='Invalid dimensions'):
        client.embed_query('lifecycle probe')
    return client.request('ping')['pid']


def test_actual_client_race_and_exit_do_not_stop_daemon(homes):
    code = '''
import json, sys
from local_rag.inference import InferenceClient
c = InferenceClient(sys.argv[1], dimensions=1)
try:
    c.embed_query('lifecycle probe')
except RuntimeError as exc:
    assert str(exc) == 'Invalid dimensions', str(exc)
print(json.dumps(c.request('ping')))
'''
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    clients = [subprocess.Popen([sys.executable, '-c', code, str(homes[0])],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
               for _ in range(2)]
    try:
        results = [p.communicate(timeout=15) for p in clients]
        assert [p.returncode for p in clients] == [0, 0], results
        pids = [json.loads(out)['pid'] for out, _ in results]
        assert pids[0] == pids[1]
        assert InferenceClient(homes[0]).request('ping')['pid'] == pids[0]
    finally:
        for p in clients:
            if p.poll() is None:
                p.kill()
            p.wait()


def test_ping_and_client_construction_are_read_only(homes):
    client = InferenceClient(homes[0])
    with pytest.raises(FileNotFoundError):
        client.request('ping')
    assert not homes[0].exists()


def test_crash_recovers_only_on_next_demand_and_profiles_are_isolated(homes):
    first = demand(homes[0])
    other = demand(homes[1])
    assert first != other
    os.kill(first, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            InferenceClient(homes[0], timeout=.1).request('ping')
        except (OSError, ValueError):
            break
        time.sleep(.02)
    time.sleep(.1)
    with pytest.raises((OSError, ValueError)):
        InferenceClient(homes[0]).request('ping')
    assert InferenceClient(homes[1]).request('ping')['pid'] == other
    assert demand(homes[0]) not in {first, other}


def test_failed_readiness_is_bounded_and_backoff_prevents_spawn_storm(homes, monkeypatch):
    import local_rag.inference as inference
    real_popen = subprocess.Popen
    children = []
    def unready(argv, **kwargs):
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(inference.subprocess, 'Popen', unready)
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            InferenceClient(homes[0], startup_timeout=.2).embed_query('secret input')
        assert time.monotonic() - start < 1.5
        for _ in range(3):
            with pytest.raises(RuntimeError, match='recently failed'):
                InferenceClient(homes[0], startup_timeout=.2).embed_query('secret input')
        assert len(children) == 1
        directory = homes[0] / 'local-rag/.cache'
        for name in ('startup.lock', 'owner.lock', 'inference.log'):
            assert (directory / name).stat().st_mode & 0o777 == 0o600
        assert 'secret input' not in (directory / 'inference.log').read_text()
    finally:
        for child in children:
            child.kill()
            child.wait(timeout=5)


def test_failure_backoff_starts_after_readiness_deadline(homes, monkeypatch):
    import local_rag.inference as inference
    real_popen = subprocess.Popen
    children = []
    def unready(argv, **kwargs):
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(inference.subprocess, 'Popen', unready)
    try:
        with pytest.raises(TimeoutError):
            InferenceClient(homes[0], startup_timeout=5.1).embed_query('probe')
        with pytest.raises(RuntimeError, match='recently failed'):
            InferenceClient(homes[0], startup_timeout=.1).embed_query('probe')
        assert len(children) == 1
    finally:
        for child in children:
            child.kill()
            child.wait(timeout=5)


def test_pending_launch_survives_cooldown_and_concurrent_clients(homes):
    # Each client substitutes only the daemon command: real exec and inherited
    # descriptors still run, but the child never reaches server.start().
    code = '''
import json, subprocess, sys
from pathlib import Path
from local_rag.inference import InferenceClient
real_popen = subprocess.Popen
children = []
def sleeping(argv, **kwargs):
    child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
    children.append(child)
    with (Path(sys.argv[1]) / 'launches').open('a') as log:
        log.write(str(child.pid) + '\\n')
    return child
subprocess.Popen = sleeping
try:
    try:
        InferenceClient(sys.argv[1], startup_timeout=.2).ensure_started()
    except (RuntimeError, TimeoutError) as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
    else:
        raise AssertionError('sleeping daemon cannot be ready')
    sys.stdin.readline()
finally:
    for child in children:
        child.kill()
        child.wait(timeout=5)
'''
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    clients = []
    def start():
        child = subprocess.Popen([sys.executable, '-c', code, str(homes[0])],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, env=env)
        clients.append(child)
        return child
    try:
        import select
        first = start()
        assert first.stdout is not None
        assert select.select([first.stdout], [], [], 5)[0]
        assert 'deadline' in json.loads(first.stdout.readline())['error']
        launches = homes[0] / 'launches'
        initial = launches.read_text().splitlines()
        assert len(initial) == 1
        time.sleep(5.1)  # Real cooldown expiry, not a mocked clock.
        contenders = [start() for _ in range(4)]
        results = [p.communicate('\n', timeout=5) for p in contenders]
        assert all(p.returncode == 0 for p in contenders), results
        assert launches.read_text().splitlines() == initial
        os.kill(int(initial[0]), 0)  # No timeout-triggered kill/restart.
        first.communicate('\n', timeout=5)
        assert first.returncode == 0
        # Death releases launch ownership; next demand may start a replacement.
        replacement = start()
        out, err = replacement.communicate('\n', timeout=5)
        assert replacement.returncode == 0, (out, err)
        assert len(launches.read_text().splitlines()) == 2
    finally:
        for child in clients:
            if child.poll() is None:
                try:
                    child.communicate('\n', timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait(timeout=5)


def test_startup_lock_wait_is_bounded(homes):
    import fcntl
    directory = homes[0] / 'local-rag/.cache'
    directory.mkdir(parents=True, mode=0o700)
    path = directory / 'startup.lock'
    path.touch(mode=0o600)
    with path.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            InferenceClient(homes[0], startup_timeout=.15).embed_query('probe')
        assert time.monotonic() - start < 1
    assert not (directory / 'inference.log').exists()


def test_live_unauthorized_service_is_not_replaced(homes):
    pid = demand(homes[0])
    directory = homes[0] / 'local-rag/.cache'
    token_path = directory / 'token'
    token = token_path.read_text()
    inode = (directory / 'i.sock').stat().st_ino
    try:
        token_path.write_text('wrong-token')
        with pytest.raises(RuntimeError, match='Unauthorized'):
            InferenceClient(homes[0]).embed_query('probe')
        assert token_path.read_text() == 'wrong-token'
        assert (directory / 'i.sock').stat().st_ino == inode
        token_path.unlink()
        with pytest.raises(PermissionError, match='credentials'):
            demand(homes[0])
        assert not token_path.exists()
    finally:
        token_path.touch(mode=0o600)
        token_path.write_text(token)
    assert InferenceClient(homes[0]).request('ping')['pid'] == pid


def test_lifetime_owner_lock_blocks_replacement_without_socket(homes):
    import fcntl
    directory = homes[0] / 'local-rag/.cache'
    directory.mkdir(parents=True, mode=0o700)
    path = directory / 'owner.lock'
    path.touch(mode=0o600)
    with path.open('r+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match='owns profile'):
            InferenceClient(homes[0]).embed_query('probe')
    assert not (directory / 'inference.log').exists()


def test_insecure_directory_and_symlink_lock_are_not_repaired(homes):
    directory = homes[0] / 'local-rag/.cache'
    directory.mkdir(parents=True, mode=0o755)
    directory.chmod(0o755)
    with pytest.raises(PermissionError, match='directory'):
        InferenceClient(homes[0]).embed_query('probe')
    assert directory.stat().st_mode & 0o777 == 0o755
    directory.chmod(0o700)
    target = homes[0] / 'untouched'
    target.write_text('do not overwrite')
    (directory / 'startup.lock').symlink_to(target)
    with pytest.raises(OSError):
        InferenceClient(homes[0]).embed_query('probe')
    assert target.read_text() == 'do not overwrite'
    assert not (directory / 'inference.log').exists()


@pytest.mark.parametrize('kind', ['hardlink', 'symlink', 'permissions'])
def test_direct_server_rejects_unsafe_token_without_changing_victim(homes, kind):
    directory = homes[0] / 'local-rag/.cache'
    directory.mkdir(parents=True, mode=0o700)
    victim = homes[0] / 'victim'
    victim.write_text('keep this secret intact')
    victim.chmod(0o600 if kind != 'permissions' else 0o644)
    token = directory / 'token'
    if kind == 'symlink':
        token.symlink_to(victim)
    elif kind == 'hardlink':
        os.link(victim, token)
    else:
        victim.rename(token)
        victim = token
    before = victim.stat()
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    child = subprocess.Popen(
        [sys.executable, '-m', 'local_rag.inference', '--home', str(homes[0])],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        import select
        assert child.stdout is not None and child.stderr is not None
        assert select.select([child.stdout, child.stderr], [], [], 5)[0]
        assert victim.read_text() == 'keep this secret intact'
        assert victim.stat().st_mode == before.st_mode
        out, err = child.communicate(timeout=5)
        assert child.returncode != 0, (out, err)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_insecure_token_is_not_read_or_repaired(homes):
    demand(homes[0])
    token_path = homes[0] / 'local-rag/.cache/token'
    token_path.chmod(0o644)
    try:
        with pytest.raises(PermissionError, match='token'):
            InferenceClient(homes[0]).embed_query('probe')
        assert token_path.stat().st_mode & 0o777 == 0o644
    finally:
        token_path.chmod(0o600)
