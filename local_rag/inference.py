"""Private profile-scoped inference. Foreground service; never manages Hermes processes.

One worker owns both models, lazily loaded once. Clients never fall back to local
model copies. A timed-out native call cannot be interrupted safely: its worker
continues, but bounded queues and deadlines prevent unbounded follow-up work.
"""
from __future__ import annotations

import argparse
import fcntl
import hmac
import json
import math
import os
from pathlib import Path
import queue
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from itertools import count

# Bounded frame accommodates 32K Unicode characters plus protocol metadata.
MAX_FRAME = 256_000


def _private_directory(path):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError('Inference directory must be owned by this user, not a symlink')
    path.chmod(0o700)
    return path


def _receive(conn):
    data = bytearray()
    while b'\n' not in data:
        part = conn.recv(min(4096, MAX_FRAME + 1 - len(data)))
        if not part:
            raise ValueError('Incomplete inference frame')
        data.extend(part)
        if len(data) > MAX_FRAME:
            raise ValueError('Inference frame too large')
    return json.loads(data.split(b'\n', 1)[0])


def _send(conn, value):
    data = json.dumps(value, allow_nan=False, ensure_ascii=False).encode() + b'\n'
    if len(data) > MAX_FRAME:
        raise ValueError('Inference frame too large')
    conn.sendall(data)


class ModelBackend:
    def __init__(self, home):
        self.home = Path(home)
        self.text = self.visual = None

    def __call__(self, op, value, *, dimensions=512, title='none'):
        if op in {'query', 'document'}:
            if self.text is None:
                from .embedder import LiteRTEmbeddingGemma
                self.text = LiteRTEmbeddingGemma(dimensions=768)
            vector = (self.text.embed_query(value) if op == 'query'
                      else self.text.embed_document(value, title=title))[:dimensions]
            norm = math.sqrt(sum(x*x for x in vector))
            return [x/norm for x in vector] if norm else vector
        if op in {'clip_text', 'clip_image'}:
            if self.visual is None:
                from .visual import ClipOnnxEmbedder
                self.visual = ClipOnnxEmbedder()
            return self.visual.embed_text(value) if op == 'clip_text' else self.visual.embed_image(value)
        raise ValueError('Unknown inference operation')


class InferenceServer:
    def __init__(self, home, *, backend=None, queue_size=32, connections=48):
        self.home = Path(home)
        self.directory = self.home / 'local-rag' / '.cache'
        self.socket_path = self.directory / 'i.sock'
        self.backend = backend if backend is not None else ModelBackend(home)
        if queue_size < 1 or connections < 1:
            raise ValueError("Queue and connection bounds must be positive")
        self.jobs = queue.PriorityQueue(maxsize=2 * queue_size)
        self.capacity = {p: threading.BoundedSemaphore(queue_size) for p in ("interactive", "indexing")}
        self.slots = threading.BoundedSemaphore(connections)
        self.sequence = count()
        self.stopping = threading.Event()
        self.lock = self.listener = None
        self.token_owned = False
        self.threads = []
        self.handlers = set()
        self.handlers_lock = threading.Lock()

    def start(self):
        _private_directory(self.directory)
        if len(os.fsencode(self.socket_path)) >= 104:
            raise ValueError('Profile path too long for a Unix socket (104-byte limit)')
        fd = os.open(self.directory / 'owner.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.lock = os.fdopen(fd, 'w')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError('Inference service already running') from None
        try:
            self.token = secrets.token_hex(32)
            token_path = self.directory / 'token'
            with _private_file(token_path) as handle:
                handle.truncate()
                handle.write(self.token)
            self.token_owned = True
            self.socket_path.unlink(missing_ok=True)
            self.listener = socket.socket(socket.AF_UNIX)
            self.listener.bind(str(self.socket_path))
            self.socket_path.chmod(0o600)
            self.listener.listen(32)
            self.listener.settimeout(.2)
            for target in (self._worker, self._accept):
                thread = threading.Thread(target=target, daemon=True)
                self.threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise

    def _accept(self):
        while not self.stopping.is_set():
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not self.slots.acquire(blocking=False):
                conn.close()
                continue
            thread = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            with self.handlers_lock:
                self.handlers.add(thread)
            thread.start()

    def _handle(self, conn):
        try:
            conn.settimeout(2)
            request = _receive(conn)
            if not isinstance(request, dict) or not hmac.compare_digest(str(request.get('token', '')), self.token):
                raise PermissionError('Unauthorized inference request')
            op = request.get('op')
            if op == 'ping':
                _send(conn, {'result': {'pid': os.getpid()}})
                return
            if op not in {'query', 'document', 'clip_text', 'clip_image'}:
                raise ValueError('Unknown inference operation')
            value = request.get('value')
            if not isinstance(value, str) or len(value) > 32000:
                raise ValueError('Invalid inference input')
            dimensions = request.get('dimensions', 512)
            if self.backend.__class__ is ModelBackend and dimensions not in {128, 256, 512, 768}:
                raise ValueError('Invalid dimensions')
            timeout = float(request.get('timeout', 30))
            if not math.isfinite(timeout) or not 0 < timeout <= 120:
                raise ValueError('Invalid timeout')
            priority = request.get('priority', 'interactive')
            if priority not in {'interactive', 'indexing'}:
                raise ValueError('Invalid priority')
            future = Future()
            deadline = time.monotonic() + timeout
            if not self.capacity[priority].acquire(blocking=False):
                raise queue.Full
            try:
                self.jobs.put_nowait((0 if priority == 'interactive' else 1, next(self.sequence), deadline, future, request))
            except BaseException:
                self.capacity[priority].release()
                raise
            try:
                result = future.result(timeout=timeout)
            except TimeoutError:
                future.cancel()
                raise TimeoutError('Inference deadline exceeded') from None
            _send(conn, {'result': result})
        except Exception as exc:
            try:
                _send(conn, {'error': 'Inference queue full' if isinstance(exc, queue.Full) else str(exc)})
            except (OSError, ValueError):
                pass
        finally:
            conn.close()
            self.slots.release()
            with self.handlers_lock:
                self.handlers.discard(threading.current_thread())

    def _worker(self):
        while not self.stopping.is_set():
            try:
                _, _, deadline, future, request = self.jobs.get(timeout=.1)
                self.capacity[request.get("priority", "interactive")].release()
            except queue.Empty:
                continue
            try:
                if time.monotonic() >= deadline:
                    future.cancel()
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = self.backend(request['op'], request['value'], dimensions=request.get('dimensions', 512), title=request.get('title', 'none'))
                except Exception as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self.jobs.task_done()

    def close(self):
        self.stopping.set()
        if self.listener:
            self.listener.close()
        # Hold singleton lock until native inference has really returned.
        for thread in self.threads:
            thread.join()
        while not self.jobs.empty():
            *_, future, request = self.jobs.get_nowait()
            future.cancel()
            self.jobs.task_done()
        with self.handlers_lock:
            handlers = list(self.handlers)
        for thread in handlers:
            thread.join(timeout=3)
        if isinstance(self.backend, ModelBackend):
            self.backend.text = self.backend.visual = None
        if self.lock:
            self.socket_path.unlink(missing_ok=True)
            if self.token_owned:
                (self.directory / 'token').unlink(missing_ok=True)
            self.lock.close()
            self.lock = None


def _private_file(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077 or info.st_nlink != 1):
        os.close(fd)
        raise PermissionError('Insecure inference lifecycle file')
    return os.fdopen(fd, 'r+')


class InferenceClient:
    def __init__(self, home, *, dimensions=512, timeout: float = 30, startup_timeout: float = 10):
        self.home = Path(home).expanduser().resolve()
        self.directory = self.home / 'local-rag' / '.cache'
        self.dimensions = dimensions
        self.timeout = timeout
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError('Startup timeout must be positive and finite')
        self.startup_timeout = startup_timeout

    def _available(self, timeout):
        try:
            self._request('ping', socket_timeout=timeout)
            return True
        except ConnectionRefusedError:
            return False
        except FileNotFoundError:
            # A missing token must not allow replacing a live, unauthenticated
            # listener. Probe transport only; never unlink or repair its files.
            with socket.socket(socket.AF_UNIX) as conn:
                conn.settimeout(timeout)
                try:
                    conn.connect(str(self.directory / 'i.sock'))
                except (FileNotFoundError, ConnectionRefusedError):
                    return False
            # The daemon may have bound between the first and second connect.
            # Authenticate again rather than misclassifying normal startup.
            try:
                self._request('ping', socket_timeout=timeout)
                return True
            except FileNotFoundError:
                raise PermissionError('Live inference socket has no usable credentials') from None
            except ConnectionRefusedError:
                return False

    def ensure_started(self):
        deadline = time.monotonic() + self.startup_timeout

        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError('Inference startup deadline exceeded')
            return min(value, .25)

        if self._available(remaining()):
            return
        # Existing insecure directories are rejected by the read-only probe.
        _private_directory(self.directory)
        with _private_file(self.directory / 'startup.lock') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(.025, remaining()))
            if self._available(remaining()):
                return
            # Shared failure backoff, also covering a starter killed mid-launch.
            lock.seek(0)
            attempted = lock.read().strip()
            if attempted and 0 <= time.monotonic() - float(attempted) < 5:
                raise RuntimeError('Inference startup recently failed; retry on later demand')
            # Lifetime lock is authoritative even if an owner cannot answer ping.
            with _private_file(self.directory / 'owner.lock') as owner:
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError('Inference service owns profile but is not ready') from None
            # Inherit a separate flock across exec. The child owns this guard
            # even before Python imports/main/owner.lock, and even if its starter
            # exits or times out. A timestamp or PID cannot provide that handoff.
            with _private_file(self.directory / 'launch.lock') as launch:
                try:
                    fcntl.flock(launch, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError('Inference launch in progress; service is not ready') from None
                lock.seek(0)
                lock.truncate()
                lock.write(str(time.monotonic()))
                lock.flush()
                try:
                    # No shell, terminal window, inherited pipes, or app-lifetime coupling.
                    with _private_file(self.directory / 'inference.log') as log:
                        log.seek(0)
                        log.truncate()
                        env = dict(os.environ)
                        env.pop('PYTHONPATH', None)
                        child = subprocess.Popen(
                            [sys.executable, '-m', 'local_rag.inference', '--home', str(self.home),
                             '--launch-fd', str(launch.fileno())],
                            cwd=str(Path(__file__).resolve().parent.parent), env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True, close_fds=True, pass_fds=(launch.fileno(),))
                    # Reap our child if this client remains alive. Never terminate it on
                    # client shutdown or readiness timeout; the lifetime flock owns it.
                    threading.Thread(target=child.wait, daemon=True).start()
                    while True:
                        if self._available(remaining()):
                            lock.seek(0)
                            lock.truncate()
                            lock.flush()
                            return
                        if child.poll() is not None:
                            raise RuntimeError('Inference service failed to start; see private inference.log')
                        time.sleep(min(.025, remaining()))
                except BaseException:
                    lock.seek(0)
                    lock.truncate()
                    lock.write(str(time.monotonic()))
                    lock.flush()
                    raise

    def request(self, op, value='', *, priority='interactive', **kwargs):
        if op in {'query', 'document', 'clip_text', 'clip_image'}:
            self.ensure_started()
        return self._request(op, value, priority=priority, **kwargs)

    def _request(self, op, value='', *, priority='interactive', socket_timeout=None, **kwargs):
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError('Insecure inference directory')
        fd = os.open(self.directory / 'token', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd) as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1):
                raise PermissionError('Insecure inference token')
            token = handle.read(256)
        with socket.socket(socket.AF_UNIX) as conn:
            conn.settimeout(self.timeout + 1 if socket_timeout is None else socket_timeout)
            conn.connect(str(self.directory / 'i.sock'))
            _send(conn, dict(token=token, op=op, value=str(value), dimensions=self.dimensions,
                             timeout=self.timeout, priority=priority, **kwargs))
            result = _receive(conn)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result['result']

    def embed_query(self, text):
        return self.request('query', text)

    def embed_document(self, text, *, title='none'):
        return self.request('document', text, priority='indexing', title=title)

    def embed_text(self, text):
        return self.request('clip_text', text)

    def embed_image(self, path):
        return self.request('clip_image', path, priority='indexing')


def main():
    parser = argparse.ArgumentParser(description='Foreground private Local RAG inference; Ctrl-C/SIGTERM to stop')
    parser.add_argument('--home', default=os.environ.get('HERMES_HOME', str(Path.home() / '.hermes')))
    parser.add_argument('--launch-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    server = InferenceServer(Path(args.home).expanduser())
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    try:
        server.start()
    finally:
        # Only hand off after start has acquired the lifetime owner.lock.
        if args.launch_fd is not None:
            os.close(args.launch_fd)
    print(json.dumps({'ready': str(server.socket_path), 'pid': os.getpid()}), flush=True)
    try:
        stopped.wait()
    finally:
        server.close()


if __name__ == '__main__':
    main()
