import importlib.util
import threading
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def home():
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".it-") as name:
        yield Path(name)
from concurrent.futures import ThreadPoolExecutor


@pytest.mark.parametrize('character', ['я', '😀'])
def test_protocol_accepts_unicode_at_text_limit(home, character):
    from local_rag.inference import InferenceServer, InferenceClient
    received = []
    def backend(op, value, **kwargs):
        received.append(value)
        return [1.0]
    server = InferenceServer(home, backend=backend)
    server.start()
    try:
        value = character * 32000
        assert InferenceClient(home).embed_document(value) == [1.0]
        assert received == [value]
    finally:
        server.close()


def test_private_singleton_service(home):
    tmp_path = home
    assert importlib.util.find_spec('local_rag.inference') is not None, 'shared inference service missing'
    from local_rag.inference import InferenceServer, InferenceClient
    calls = []
    def backend(op, value, **kwargs):
        calls.append((op, value))
        return [1.0, 0.0]
    server = InferenceServer(tmp_path, backend=backend)
    server.start()
    try:
        assert server.socket_path.stat().st_mode & 0o777 == 0o600
        other = InferenceServer(tmp_path, backend=backend)
        import pytest
        with pytest.raises(RuntimeError, match='already running'):
            other.start()
        client = InferenceClient(tmp_path, dimensions=2)
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(client.embed_query, map(str, range(12)))) == [[1., 0.]] * 12
        assert len(calls) == 12
    finally:
        server.close()
    assert not server.socket_path.exists()


def test_interactive_capacity_reserved_and_priority_deadlines(home):
    from local_rag.inference import InferenceClient, InferenceServer
    import time
    entered, release = threading.Event(), threading.Event()
    order = []
    def backend(op, value, **kwargs):
        order.append(value)
        if value == 'blocking':
            entered.set()
            assert release.wait(5)
        return [1., 0.]
    server = InferenceServer(home, backend=backend, queue_size=1)
    server.start()
    client = InferenceClient(home, timeout=3)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            first = pool.submit(client.embed_document, 'blocking')
            assert entered.wait(2)
            pending = pool.submit(client.embed_document, 'background')
            deadline = time.monotonic() + 2
            while server.jobs.qsize() != 1 and time.monotonic() < deadline:
                time.sleep(.005)
            interactive = pool.submit(client.embed_query, 'interactive')
            deadline = time.monotonic() + 1
            while server.jobs.qsize() != 2 and not interactive.done() and time.monotonic() < deadline:
                time.sleep(.005)
            try:
                assert not interactive.done(), 'indexing saturation rejected interactive recall'
                with pytest.raises(RuntimeError, match='queue full'):
                    client.embed_document('overflow')
            finally:
                release.set()
            assert first.result() == pending.result() == interactive.result() == [1., 0.]
        assert order == ['blocking', 'interactive', 'background']
    finally:
        release.set()
        server.close()


def test_auth_and_deadline_do_not_create_replacement_worker(home):
    from local_rag.inference import InferenceClient, InferenceServer, _send, _receive
    import socket
    entered, release = threading.Event(), threading.Event()
    calls = []
    def backend(op, value, **kwargs):
        calls.append(value)
        if value == 'slow':
            entered.set()
            release.wait(5)
        return [1.]
    server = InferenceServer(home, backend=backend)
    server.start()
    try:
        with socket.socket(socket.AF_UNIX) as conn:
            conn.connect(str(server.socket_path))
            _send(conn, {'token': 'wrong', 'op': 'query', 'value': 'secret'})
            assert 'Unauthorized' in _receive(conn)['error']
        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(InferenceClient(home, timeout=.2).embed_query, 'slow')
            assert entered.wait(2)
            with pytest.raises(RuntimeError, match='deadline'):
                InferenceClient(home, timeout=.05).embed_query('expired')
            with pytest.raises(RuntimeError, match='deadline'):
                slow.result(timeout=2)
            assert InferenceClient(home).request('ping')['pid'] > 0
            release.set()
        assert InferenceClient(home).embed_query('after') == [1.]
        assert calls == ['slow', 'after']
    finally:
        release.set()
        server.close()


def test_compatibility_factories_also_return_clients(home, monkeypatch):
    from local_rag.embedder import get_shared_embedder
    from local_rag.visual import get_shared_visual_embedder
    from local_rag.inference import InferenceClient
    monkeypatch.setenv('HERMES_HOME', str(home))
    assert isinstance(get_shared_embedder(), InferenceClient)
    assert isinstance(get_shared_visual_embedder(), InferenceClient)
    assert get_shared_embedder().directory == home / 'local-rag' / '.cache'


def test_real_models_shared_through_service(home):
    from local_rag.inference import InferenceServer, InferenceClient
    from local_rag.embedder import default_model_path
    from PIL import Image
    import math
    if not default_model_path().is_file() or not (Path.home() / '.hermes/models/clip-onnx/onnx/vision_model_quantized.onnx').is_file():
        pytest.skip('Local model weights not installed')
    server = InferenceServer(home)
    server.start()
    try:
        text_client = InferenceClient(home, dimensions=128)
        v = text_client.embed_query('approved red company logo')
        text = server.backend.text
        assert len(v) == 128 and abs(sum(x*x for x in v) - 1) < .001
        assert len(InferenceClient(home, dimensions=256).embed_document('ACME NORTH')) == 256
        assert server.backend.text is text
        path = home / 'red.png'
        Image.new('RGB', (224,224), 'red').save(path)
        assert len(text_client.embed_image(path)) == 512
        visual = server.backend.visual
        assert len(text_client.embed_text('red logo')) == 512
        assert server.backend.visual is visual
    finally:
        server.close()
    assert server.backend.text is None and server.backend.visual is None


def test_cli_defaults_to_active_profile(home, monkeypatch):
    import argparse
    from local_rag.cli import register_backfill_cli
    monkeypatch.setenv('HERMES_HOME', str(home))
    parser = argparse.ArgumentParser()
    register_backfill_cli(parser)
    args = parser.parse_args(['preview', '--plan', 'review.json'])
    assert Path(args.home) == home


def test_provider_uses_profile_client_without_loading_models(home):
    from local_rag import LocalRagProvider
    from local_rag.inference import InferenceClient
    provider = LocalRagProvider()
    provider.initialize('desktop', hermes_home=str(home))
    try:
        assert isinstance(provider._embedder, InferenceClient)
        assert provider._embedder.directory == home / 'local-rag' / '.cache'
    finally:
        provider.shutdown()


def test_crash_recovery_rotates_token_and_keeps_profile_isolation(home):
    import subprocess
    import sys
    import os
    import select
    import socket
    from local_rag.inference import InferenceClient, _send, _receive
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    command = [sys.executable, '-m', 'local_rag.inference', '--home', str(home)]
    def start():
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        assert select.select([child.stdout], [], [], 10)[0]
        assert 'ready' in child.stdout.readline(), child.stderr.read()
        return child
    child = start()
    token = (home / 'local-rag/.cache/token').read_text()
    child.kill()
    child.wait(timeout=10)
    child = start()
    try:
        assert (home / 'local-rag/.cache/token').read_text() != token
        with socket.socket(socket.AF_UNIX) as conn:
            conn.connect(str(home / 'local-rag/.cache/i.sock'))
            _send(conn, dict(token=token, op='ping'))
            assert 'Unauthorized' in _receive(conn)['error']
        with pytest.raises(FileNotFoundError):
            InferenceClient(home / 'other-profile').request('ping')
        assert InferenceClient(home).request('ping')['pid'] == child.pid
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_standalone_service_subprocess_singleton_and_shutdown(home):
    import subprocess
    import sys
    import os
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    command = [sys.executable, '-m', 'local_rag.inference', '--home', str(home)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        import select
        assert select.select([process.stdout], [], [], 10)[0], 'service did not start'
        line = process.stdout.readline()
        assert 'ready' in line, process.stderr.read()
        from local_rag.inference import InferenceClient
        client = InferenceClient(home)
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert set(pool.map(lambda _: client.request('ping')['pid'], range(16))) == {process.pid}
        duplicate = subprocess.run(command, capture_output=True, text=True, env=env, timeout=10)
        assert duplicate.returncode != 0 and 'already running' in duplicate.stderr
        assert client.request('ping')['pid'] == process.pid
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert process.returncode == 0
    assert not (home / 'local-rag/.cache/i.sock').exists()
