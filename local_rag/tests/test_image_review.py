"""Regression coverage for OCR policy limits and BLOB-free metadata queries."""
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
import pytest

from local_rag.images import CuratedImages, ImagePolicyError, local_ocr
from local_rag.tests.test_curated_images import Text, Clip


@pytest.mark.parametrize('operation', ['save', 'reindex', 'reconcile'])
@pytest.mark.parametrize('failure', ['nonzero', 'unavailable'])
def test_ocr_failure_blocks_derivatives_and_preserves_previous_record(tmp_path, monkeypatch, operation, failure):
    import sys
    source = tmp_path / 'source.png'
    Image.new('RGB', (30, 20), 'red').save(source)
    images = CuratedImages(tmp_path / 'home', text=Text(), visual=Clip,
                           ocr=lambda _: ('ordinary table', 'fixture'))
    try:
        saved = images.save('owner', source, decision='save', reason='approved',
                            scope='global', group='brand', description='ordinary table')
        before = dict(images.db.execute('SELECT * FROM curated_images').fetchone())
        managed = Path(saved['path'])
        if operation == 'save':
            Image.new('RGB', (30, 20), 'blue').save(source)
        elif operation == 'reconcile':
            Image.new('RGB', (30, 20), 'blue').save(managed)
        managed_bytes = managed.read_bytes()
        processes = []
        if failure == 'nonzero':
            popen = subprocess.Popen
            monkeypatch.setattr('local_rag.images.shutil.which', lambda _: sys.executable)
            def start(command, **kwargs):
                process = popen([sys.executable, '-c',
                                 'import sys; print("password=supersecret123"); sys.exit(1)'], **kwargs)
                processes.append(process)
                return process
            monkeypatch.setattr('local_rag.images.subprocess.Popen', start)
        else:
            monkeypatch.setattr('local_rag.images.shutil.which', lambda _: None)
        images.ocr = local_ocr
        embed = Mock(wraps=images.text.embed_document)
        visual = Mock(wraps=images.visual)
        monkeypatch.setattr(images.text, 'embed_document', embed)
        monkeypatch.setattr(images, 'visual', visual)
        if operation == 'reconcile':
            images.reconcile('owner', force=True)
        else:
            with pytest.raises(RuntimeError, match='OCR'):
                if operation == 'save':
                    images.save('owner', source, decision='save', reason='approved',
                                scope='global', group='brand', description='ordinary table')
                else:
                    images.reindex('owner', saved['id'], description='updated table')
        embed.assert_not_called()
        visual.assert_not_called()
        assert [dict(row) for row in images.db.execute('SELECT * FROM curated_images')] == [before]
        assert list(images.directory.iterdir()) == [managed]
        assert managed.read_bytes() == managed_bytes
        assert not list(images._private_cache().glob('tmp*'))
        assert images.db.execute('SELECT count(*) FROM image_gc').fetchone()[0] == 0
        if failure == 'nonzero':
            assert len(processes) == 1
            assert processes[0].returncode == 1 and processes[0].stdout.closed
        if operation == 'reconcile':
            assert not images.search('owner', 'table', reconcile=False)
        else:
            assert images.search('owner', 'table', reconcile=False)[0]['id'] == saved['id']
    finally:
        images.close()


@pytest.mark.parametrize('suffix', ['password=supersecret123', 'token=secret-token-123', ''])
def test_real_local_ocr_rejects_oversized_output_before_save(tmp_path, monkeypatch, suffix):
    path = tmp_path / 'source.png'
    Image.new('RGB', (30, 20), 'red').save(path)
    output = 'ordinary table cell ' * 2000 + suffix
    monkeypatch.setattr('local_rag.images.shutil.which', lambda _: '/mock/tesseract')

    # Exercise local_ocr itself; only the external OCR process is replaced.
    import io
    class BoundedOutput(io.BytesIO):
        def read(self, size: int | None = -1):
            assert size == 32001, 'OCR must not capture unbounded stdout'
            return super().read(size)
    class Process:
        stdout = BoundedOutput(output.encode())
        killed = False
        def kill(self):
            self.killed = True
        def wait(self, timeout=None):
            return 0
        def poll(self):
            return None
    process = Process()
    monkeypatch.setattr('local_rag.images.subprocess.Popen', lambda *a, **kw: process)
    images = CuratedImages(tmp_path / 'home', text=Text(), visual=Clip, ocr=local_ocr)
    try:
        with pytest.raises(ImagePolicyError):
            images.save('owner', path, decision='save', reason='approved', scope='global')
        assert process.killed and process.stdout.closed
        assert images.count('owner') == 0
        assert not list(images.directory.iterdir())
        assert not list(images._private_cache().glob('tmp*'))
    finally:
        images.close()


@pytest.mark.parametrize('size', [32000, 32001, 10_000_000])
def test_local_ocr_output_limit_with_real_pipe(tmp_path, monkeypatch, size):
    import sys
    popen = subprocess.Popen
    processes = []
    monkeypatch.setattr('local_rag.images.shutil.which', lambda _: sys.executable)
    def start(command, **kwargs):
        process = popen([sys.executable, '-c', f'import sys; sys.stdout.write("x" * {size})'], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr('local_rag.images.subprocess.Popen', start)
    if size > 32000:
        with pytest.raises(ImagePolicyError):
            local_ocr(tmp_path / 'source.png')
    else:
        assert local_ocr(tmp_path / 'source.png') == ('x' * size, 'tesseract')
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed


@pytest.mark.parametrize('operation', ['search', 'reconcile', 'save', 'reindex', 'delete'])
def test_normal_image_operations_do_not_read_original_blob(tmp_path, operation):
    path = tmp_path / 'source.png'
    Image.new('RGB', (30, 20), 'red').save(path)
    images = CuratedImages(tmp_path / 'home', text=Text(), visual=Clip,
                           ocr=lambda _: ('ordinary table', 'fixture'))
    saved = images.save('owner', path, decision='save', reason='approved', scope='global')
    reads = []
    def authorize(action, table, column, *unused):
        if action == sqlite3.SQLITE_READ and table == 'curated_images':
            reads.append(column)
            if column == 'original':
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    images.db.set_authorizer(authorize)
    try:
        if operation == 'search':
            assert images.search('owner', 'table', reconcile=False)[0]['id'] == saved['id']
        elif operation == 'reconcile':
            images.reconcile('owner', force=True)
        elif operation == 'save':
            assert images.save('owner', path, decision='save', reason='approved', scope='global')['id'] == saved['id']
            Image.new('RGB', (30, 20), 'blue').save(path)
            assert images.save('owner', path, decision='save', reason='approved', scope='global')['id'] != saved['id']
        elif operation == 'reindex':
            reindexed = images.reindex('owner', saved['id'])
            assert reindexed is not None and reindexed['id'] == saved['id']
        else:
            assert images.delete('owner', saved['id'])
        assert reads and 'original' not in reads
    finally:
        images.db.set_authorizer(None)
        images.close()
