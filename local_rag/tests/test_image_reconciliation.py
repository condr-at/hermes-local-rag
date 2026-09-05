from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from local_rag.images import CuratedImages
from local_rag.tests.test_curated_images import Text, Clip


def make_images(tmp_path, ocr='Approved reference logo'):
    images = CuratedImages(tmp_path / 'home', text=Text(), visual=lambda: Clip(),
                           ocr=lambda _: (ocr, 'fixture'))
    source = tmp_path / 'approved.png'
    Image.new('RGB', (30, 20), 'red').save(source)
    return images, source


def save(images, source, **kwargs):
    return images.save('owner', source, decision='save', reason='Approved reference',
                       scope='global', **kwargs)


@pytest.mark.parametrize('field', ['ocr', 'description', 'reason', 'group'])
def test_sensitive_images_reject_without_rows_or_blobs(tmp_path, field):
    secret = 'password: do-not-retain-this-secret'
    images, source = make_images(tmp_path, ocr=secret if field == 'ocr' else 'Approved reference')
    kwargs = dict(decision='save', reason='Approved reference', scope='global')
    if field != 'ocr':
        kwargs[field] = secret
    try:
        with pytest.raises(ValueError, match='policy'):
            images.save('owner', source, **kwargs)
        assert images.count('owner') == 0
        assert not list(images.directory.iterdir())
    finally:
        images.close()


def test_reconcile_changed_managed_image_refreshes_hash_and_derivatives(tmp_path):
    import hashlib
    images, source = make_images(tmp_path)
    try:
        saved = save(images, source, group='brand', description='Approved logo')
        Image.new('RGB', (30, 20), 'blue').save(saved['path'])
        changed_bytes = Path(saved['path']).read_bytes()
        calls = []
        images.ocr = lambda path: (calls.append(str(path)) or 'Revised blue logo', 'fixture')
        # The ordinary retrieval path performs bounded maintenance.
        found = images.search('owner', 'Revised blue logo')
        assert len(found) == 1
        current = found[0]
        assert current['sha256'] == hashlib.sha256(changed_bytes).hexdigest()
        assert current['ocr'] == 'Revised blue logo'
        assert current['version_group'] == 'brand' and current['active'] == 1
        assert Path(current['path']).read_bytes() == changed_bytes
        assert not Path(saved['path']).exists()
        assert images.count('owner') == 1
        images.reconcile('owner', force=True)
        assert len(calls) == 1, 'unchanged content must not rerun OCR or embedding'
        assert images.search('owner', 'Revised blue logo')[0]['id'] == current['id']
    finally:
        images.close()


@pytest.mark.parametrize('replacement', ['missing', 'secret', 'invalid', 'symlink'])
def test_reconcile_removes_invalid_managed_image_and_derivatives(tmp_path, replacement):
    images, source = make_images(tmp_path)
    try:
        saved = save(images, source)
        managed = Path(saved['path'])
        if replacement == 'missing':
            managed.unlink()
        elif replacement == 'secret':
            Image.new('RGB', (30, 20), 'blue').save(managed)
            images.ocr = lambda _: ('api_key=never-persist-this', 'fixture')
        elif replacement == 'invalid':
            managed.write_bytes(b'not an image')
        else:
            managed.unlink()
            managed.symlink_to(source)
        images.reconcile('owner', force=True)
        assert images.count('owner') == 0
        assert not list(images.directory.iterdir())
        assert source.exists(), 'never remove external originals/symlink targets'
        assert not images.search('owner', 'Approved reference')
        assert not list((images.directory.parent / '.cache').glob('tmp*'))
    finally:
        images.close()


def test_changed_image_is_not_recalled_while_maintenance_is_throttled(tmp_path):
    images, source = make_images(tmp_path)
    try:
        saved = save(images, source)
        assert images.search('owner', 'Approved reference')
        Image.new('RGB', (30, 20), 'blue').save(saved['path'])
        assert not images.search('owner', 'Approved reference'), 'stale derivative must not expose unchecked changed bytes'
    finally:
        images.close()


def test_failed_save_removes_new_blob_before_releasing_sqlite_writer(tmp_path, monkeypatch):
    images, source = make_images(tmp_path, ocr='password: never-retain')
    unlink = Path.unlink
    locked = []
    def check_lock(path, *args, **kwargs):
        if path.parent == images.directory:
            locked.append(images.db.in_transaction)
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', check_lock)
    try:
        with pytest.raises(ValueError, match='policy'):
            save(images, source)
        # Policy screening now happens in excluded private staging, before any
        # managed file can be copied by native backup; no final-name unlink needed.
        assert locked == []
        assert not list(images.directory.iterdir())
        assert not list((images.directory.parent / '.cache').glob('tmp*'))
        assert images.count('owner') == 0
    finally:
        images.close()


def test_gc_rechecks_references_after_cross_process_save(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    images, source = make_images(tmp_path)
    try:
        saved = save(images, source)
        collect = images._collect_garbage
        def competing_save():
            # A separate process acquires a new reference after DELETE's commit,
            # before GC starts. GC must not trust its pre-commit reference count.
            code = '''from local_rag.images import CuratedImages
from local_rag.tests.test_curated_images import Text, Clip
import sys
images = CuratedImages(sys.argv[1], text=Text(), visual=lambda: Clip(), ocr=lambda _: ('Approved reference', 'fixture'))
images.save('other', sys.argv[2], decision='save', reason='Approved reference', scope='global')
images.close()
'''
            subprocess.run([sys.executable, '-c', code, str(tmp_path / 'home'), str(source)],
                           check=True, timeout=20, env=os.environ.copy())
            collect()
        monkeypatch.setattr(images, '_collect_garbage', competing_save)
        assert images.delete('owner', saved['id'])
        assert images.count('owner') == 0 and images.count('other') == 1
        assert Path(saved['path']).read_bytes() == source.read_bytes()
    finally:
        images.close()


def test_gc_unlink_failure_is_retried_after_reopen(tmp_path, monkeypatch):
    images, source = make_images(tmp_path)
    saved = save(images, source)
    unlink = Path.unlink
    def fail(path, *args, **kwargs):
        if str(path) == saved['path']:
            raise PermissionError('injected unlink failure')
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patcher:
        patcher.setattr(Path, 'unlink', fail)
        assert images.delete('owner', saved['id'])
    assert images.count('owner') == 0
    assert Path(saved['path']).exists()
    images.close()
    reopened = CuratedImages(tmp_path / 'home', text=Text(), visual=lambda: Clip())
    try:
        reopened.reconcile('owner', force=True)
        assert not Path(saved['path']).exists()
        assert reopened.db.execute('SELECT count(*) FROM image_gc').fetchone()[0] == 0
    finally:
        reopened.close()


def test_reconcile_is_bounded_shared_and_never_admits_new_files(tmp_path):
    images, source = make_images(tmp_path)
    other = CuratedImages(tmp_path / 'home', text=Text(), visual=lambda: Clip(),
                          ocr=lambda _: ('Approved reference', 'fixture'))
    try:
        first = save(images, source, group='one')
        second = save(images, source, group='two')
        arbitrary = images.directory / 'not-approved.png'
        Image.new('RGB', (30, 20), 'blue').save(arbitrary)
        Path(first['path']).unlink()
        images.reconcile('owner')
        assert images.count('owner') == 1, 'one logical row maximum per pass'
        other.reconcile('owner')
        assert images.count('owner') == 1, 'rate limit must be persistent, not per instance'
        other.reconcile('owner', force=True)
        assert images.count('owner') == 0
        assert arbitrary.exists(), 'do not scan, admit or garbage collect arbitrary files'
        assert images.db.execute('SELECT count(*) FROM image_checks').fetchone()[0] == 0
    finally:
        other.close()
        images.close()


def test_reindex_preserves_inactive_group_and_scope(tmp_path):
    images, source = make_images(tmp_path)
    try:
        first = images.save('owner', source, decision='save', reason='Approved reference',
                            scope='project', project='/a', group='brand')
        Image.new('RGB', (30, 20), 'blue').save(source)
        active = images.save('owner', source, decision='save', reason='Approved reference',
                             scope='project', project='/a', group='brand')
        Image.new('RGB', (30, 20), 'green').save(first['path'])
        images.reconcile('owner', project='/b', force=True)
        assert images.db.execute('SELECT hash FROM curated_images WHERE id=?', (first['id'],)).fetchone()[0] == first['sha256']
        images.reconcile('owner', project='/a', force=True)
        history = images.search('owner', 'Approved reference', project='/a', include_history=True)
        assert len(history) == 2
        assert next(r for r in history if r['id'] == first['id'])['active'] == 0
        assert [r['id'] for r in images.search('owner', 'Approved reference', project='/a')] == [active['id']]
        assert not images.search('owner', 'Approved reference', project='/b', include_history=True)
    finally:
        images.close()


def test_reconcile_write_failure_leaves_no_new_orphan(tmp_path, monkeypatch):
    images, source = make_images(tmp_path)
    try:
        saved = save(images, source)
        Image.new('RGB', (30, 20), 'blue').save(saved['path'])
        def fail(_):
            raise OSError('injected fsync failure')
        monkeypatch.setattr('local_rag.images.os.fsync', fail)
        images.reconcile('owner', force=True)
        assert images.count('owner') == 1
        assert list(images.directory.iterdir()) == [Path(saved['path'])]
        assert not images.search('owner', 'Approved reference')
    finally:
        images.close()


def test_provider_image_delete_uses_canonical_scope_not_arguments(tmp_path):
    import json
    from local_rag import LocalRagProvider
    from local_rag.config import LocalRagConfig
    home = tmp_path / 'home'
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    LocalRagConfig(visual_enabled=True).save(home)
    with sqlite3.connect(home / 'state.db') as db:
        db.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, git_repo_root TEXT, cwd TEXT)')
        db.execute('INSERT INTO sessions VALUES(?,?,?)', ('s1', str(a), str(a)))
        db.execute('INSERT INTO sessions VALUES(?,?,?)', ('s2', str(b), str(b)))
    source = a / 'logo.png'
    Image.new('RGB', (30, 20), 'red').save(source)
    p = LocalRagProvider(embedder=Text(), visual_embedder=Clip())
    p.initialize('s1', hermes_home=str(home), cwd=str(a))
    p._images.ocr = lambda _: ('Approved reference', 'fixture')
    try:
        saved = json.loads(p.handle_tool_call('local_rag_index_image', dict(path=str(source),
                           decision='save', reason='Approved reference', scope='project')))
        assert 'id' in saved, saved
        p.on_session_switch('s2')
        result = json.loads(p.handle_tool_call('local_rag_forget_image',
                            {'id': saved['id'], 'project': str(a), 'session': 's1'}))
        assert result == {'removed': False}
        assert Path(saved['path']).exists()
        p.on_session_switch('s1')
        assert json.loads(p.handle_tool_call('local_rag_forget_image', {'id': saved['id']}))['removed']
    finally:
        p.shutdown()


class FailCommit:
    def __init__(self, db):
        self.db = db
        self.fail = True

    def __getattr__(self, name):
        return getattr(self.db, name)

    def commit(self):
        if self.fail:
            self.fail = False
            raise sqlite3.OperationalError('injected commit failure')
        return self.db.commit()


def test_reconcile_commit_failure_preserves_row_and_retries(tmp_path):
    images, source = make_images(tmp_path)
    try:
        saved = save(images, source, group='brand')
        Image.new('RGB', (30, 20), 'blue').save(saved['path'])
        images.db = FailCommit(images.db)
        with pytest.raises(sqlite3.OperationalError, match='injected'):
            images.reconcile('owner', force=True)
        row = images.db.execute('SELECT * FROM curated_images').fetchone()
        assert row['hash'] == saved['sha256'] and row['active'] == 1
        assert list(images.directory.iterdir()) == [Path(saved['path'])]
        images.reconcile('owner', force=True)
        found = images.search('owner', 'Approved reference')
        assert len(found) == 1 and found[0]['sha256'] != saved['sha256']
    finally:
        images.close()


def test_duplicate_reindex_never_reactivates_history(tmp_path):
    images, source = make_images(tmp_path)
    try:
        historical = save(images, source, group='brand')
        original_bytes = source.read_bytes()
        Image.new('RGB', (30, 20), 'blue').save(source)
        active = save(images, source, group='brand')
        # First check the old version so the next pass visits the active row.
        images.reconcile('owner', force=True)
        Path(active['path']).write_bytes(original_bytes)
        images.reconcile('owner', force=True)
        assert images.count('owner') == 1
        assert not images.search('owner', 'Approved reference')
        history = images.search('owner', 'Approved reference', include_history=True)
        assert len(history) == 1 and history[0]['id'] == historical['id'] and history[0]['active'] == 0
        assert not Path(active['path']).exists()
    finally:
        images.close()


def test_delete_commit_failure_keeps_row_and_original_bytes(tmp_path):
    images, source = make_images(tmp_path)
    try:
        saved = save(images, source)
        images.db = FailCommit(images.db)
        with pytest.raises(sqlite3.OperationalError, match='injected'):
            images.delete('owner', saved['id'])
        assert images.count('owner') == 1
        assert Path(saved['path']).read_bytes() == source.read_bytes()
        assert images.delete('owner', saved['id'])
        assert not Path(saved['path']).exists()
    finally:
        images.close()


@pytest.mark.parametrize('scope,key', [('project', '/a'), ('session', 's1')])
def test_delete_checks_scope_not_just_namespace(tmp_path, scope, key):
    images, source = make_images(tmp_path)
    try:
        saved = images.save('owner', source, decision='save', reason='Approved reference',
                            scope=scope, **{scope: key})
        assert not images.delete('owner', saved['id'], project='/b', session='s2')
        assert not images.delete('owner', saved['id'])
        assert Path(saved['path']).exists()
        assert images.delete('owner', saved['id'], **{scope: key})
    finally:
        images.close()
