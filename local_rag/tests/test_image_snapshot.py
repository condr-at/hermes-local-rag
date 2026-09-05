"""Deterministic native SQLite snapshot schedules; no model/runtime dependency."""
import sqlite3
from pathlib import Path
import pytest
from PIL import Image
from local_rag.images import CuratedImages
from local_rag.tests.test_curated_images import Text, Clip


def store(home):
    return CuratedImages(home, text=Text(), visual=lambda: Clip(), ocr=lambda _: ('ACME', 'fixture'))


def save(images, path, **kw):
    return images.save('owner', path, decision='save', reason='approved', scope='global', **kw)


def snapshot(images, home):
    target = home / 'local-rag'
    target.mkdir(parents=True)
    with sqlite3.connect(target / 'curated-images.db') as db:
        images.db.backup(db)


@pytest.mark.parametrize('schedule', ['save-after-file-list', 'delete-after-db-snapshot'])
def test_db_only_snapshot_restores_exact_state(tmp_path, schedule):
    source = tmp_path / 'source.png'
    Image.new('RGB', (20, 20), 'red').save(source)
    images = store(tmp_path / 'live')
    restored_home = tmp_path / 'restored'
    if schedule == 'save-after-file-list':
        assert list(images.directory.iterdir()) == []  # native file enumeration
        saved = save(images, source)
        snapshot(images, restored_home)
    else:
        saved = save(images, source)
        snapshot(images, restored_home)
        images.delete('owner', saved['id'])
        assert not Path(saved['path']).exists()
    expected = dict(images.db.execute('SELECT * FROM curated_images').fetchone()) if schedule.startswith('save') else None
    images.close()
    restored = store(restored_home)
    def forbidden(*a, **kw):
        raise AssertionError('restore must not infer')
    restored.ocr = restored.visual = forbidden
    restored.text = None
    assert restored.materialize() == 1
    row = restored.db.execute('SELECT * FROM curated_images').fetchone()
    assert row['original'] == source.read_bytes()
    assert (restored.directory / row['filename']).read_bytes() == source.read_bytes()
    assert row['ocr'] == 'ACME'
    if expected:
        assert dict(row) == expected
    restored.close()


@pytest.mark.parametrize('damage', ['valid', 'missing', 'tampered', 'symlink'])
def test_legacy_migration_is_verified_and_all_or_nothing(tmp_path, damage):
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    images = store(tmp_path / 'home')
    first = save(images, source, group='one')
    save(images, source, group='two')
    images.db.execute('ALTER TABLE curated_images DROP COLUMN original')
    images.db.commit()
    images.close()
    path = Path(first['path'])
    if damage == 'missing':
        path.unlink()
    elif damage == 'tampered':
        Image.new('RGB', (20,20), 'blue').save(path)
    elif damage == 'symlink':
        path.unlink()
        path.symlink_to(source)
    if damage == 'valid':
        images = store(tmp_path / 'home')
        assert all(r[0] == source.read_bytes() for r in images.db.execute('SELECT original FROM curated_images'))
        images.close()
    else:
        with pytest.raises(ValueError, match='verified original'):
            store(tmp_path / 'home')
        with sqlite3.connect(tmp_path / 'home/local-rag/curated-images.db') as db:
            assert 'original' not in {r[1] for r in db.execute('PRAGMA table_info(curated_images)')}
            assert db.execute('SELECT count(*) FROM curated_images').fetchone()[0] == 2


def test_restore_integrity_and_atomic_failure(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    images = store(tmp_path / 'home')
    saved = save(images, source)
    path = Path(saved['path'])
    path.unlink()
    images.db.execute('UPDATE curated_images SET original=?', (b'corrupt',))
    images.db.commit()
    with pytest.raises(ValueError, match='integrity'):
        images.materialize()
    assert not path.exists()
    images.db.execute('UPDATE curated_images SET original=?', (source.read_bytes(),))
    images.db.commit()
    import os
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(OSError):
        images.materialize()
    assert not path.exists()
    assert list((images.directory.parent / '.cache').iterdir()) == []
    images.close()

def test_normal_deletion_does_not_resurrect_and_edit_updates_original(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    images = store(tmp_path / 'home')
    saved = save(images, source)
    Image.new('RGB', (20,20), 'blue').save(saved['path'])
    edited = Path(saved['path']).read_bytes()
    images.reconcile('owner', force=True)
    row = images.db.execute('SELECT * FROM curated_images').fetchone()
    assert row['original'] == edited
    Path(images.directory / row['filename']).unlink()
    images.reconcile('owner', force=True)
    assert images.db.execute('SELECT original FROM curated_images').fetchall() == []
    assert images.materialize() == 0
    images.close()

def test_explicit_reindex_identical_bytes_preserves_identity_and_rolls_back(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    images = store(tmp_path / 'home')
    saved = save(images, source)
    before = dict(images.db.execute('SELECT * FROM curated_images').fetchone())
    images.ocr = lambda _: ('NEW OCR', 'new fixture')
    updated = images.reindex('owner', saved['id'], description='Updated caption')
    after = dict(images.db.execute('SELECT * FROM curated_images').fetchone())
    assert updated['id'] == saved['id'] and updated['ocr'] == 'NEW OCR'
    for field in ('original', 'created', 'active', 'hash', 'version_group', 'scope'):
        assert after[field] == before[field]
    def fail(*a):
        raise RuntimeError('CLIP unavailable')
    images.visual = fail
    with pytest.raises(RuntimeError):
        images.reindex('owner', saved['id'], description='Must roll back')
    assert dict(images.db.execute('SELECT * FROM curated_images').fetchone()) == after
    assert images.reindex('other', saved['id']) is None
    images.close()

def test_unified_search_preserves_text_on_image_outage():
    from local_rag.provider import LocalRagProvider
    from types import SimpleNamespace
    provider = LocalRagProvider(embedder=Text())
    provider._service = SimpleNamespace(search=lambda *a, **kw: [SimpleNamespace(text='working memory')])
    def fail(*a, **kw):
        raise RuntimeError('CLIP unavailable')
    provider._image_search = fail
    assert provider._unified_search('query')[0]['text'] == 'working memory'
    assert provider._retrieval_warnings == ['image retrieval unavailable: RuntimeError']



@pytest.mark.parametrize('command', ['restore-images', 'reindex-image'])
def test_image_maintenance_cli(tmp_path, monkeypatch, capsys, command):
    from local_rag.cli import main
    import sys
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    home = tmp_path / 'home'
    images = store(home)
    saved = save(images, source)
    images.close()
    if command == 'restore-images':
        Path(saved['path']).unlink()
        args = []
    else:
        args = [str(saved['id']), '--description', 'Updated caption']
        from local_rag import cli
        monkeypatch.setattr(cli, 'InferenceClient', lambda *a, **kw: Text())
        monkeypatch.setattr(CuratedImages, '_derivatives', lambda *a: ('NEW', 'fixture', [0.]*512, [1.]*3))
    monkeypatch.setattr(sys, 'argv', ['local-rag', '--home', str(home), '--namespace', 'owner', command] + args)
    assert main() == 0
    assert Path(saved['path']).read_bytes() == source.read_bytes()
    assert 'materialized' in capsys.readouterr().out if command == 'restore-images' else True
    assert not (home / 'local-rag/memory.db').exists()



def test_original_database_and_staging_are_private(tmp_path):
    import stat
    images = store(tmp_path / 'home')
    assert stat.S_IMODE((images.directory.parent / 'curated-images.db').stat().st_mode) == 0o600
    cache = images.directory.parent / '.cache'
    external = tmp_path / 'outside'
    external.mkdir()
    cache.symlink_to(external, target_is_directory=True)
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    with pytest.raises(ValueError, match='symlink'):
        save(images, source)
    assert list(external.iterdir()) == []
    images.close()


@pytest.mark.parametrize('schedule', ['save-after-file-list', 'delete-after-db-snapshot'])
def test_actual_native_backup_with_concurrent_writer(tmp_path, monkeypatch, schedule):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace
    import zipfile
    backup = pytest.importorskip('hermes_cli.backup')
    home = tmp_path / 'live'
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    images = store(home)
    saved = save(images, source) if schedule.startswith('delete') else None
    images.close()
    release = Event()
    def writer():
        assert release.wait(10)
        other = store(home)
        try:
            if saved:
                assert other.delete('owner', saved['id'])
            else:
                save(other, source)
        finally:
            other.close()
    original_copy = backup._safe_copy_db
    monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: home)
    monkeypatch.setattr(backup, 'display_hermes_home', lambda: str(home))
    monkeypatch.setattr(backup, '_collect_memory_provider_external_paths', lambda: [])
    with ThreadPoolExecutor(max_workers=1) as pool:
        mutation = pool.submit(writer)
        def gated_copy(src, dst):
            if schedule.startswith('save'):
                release.set()  # Native file list is already finalized.
                mutation.result(timeout=10)
            result = original_copy(src, dst)
            if schedule.startswith('delete'):
                release.set()  # Native SQLite snapshot is already complete.
                mutation.result(timeout=10)
            return result
        monkeypatch.setattr(backup, '_safe_copy_db', gated_copy)
        archive_path = tmp_path / 'native.zip'
        backup.run_backup(SimpleNamespace(output=str(archive_path)))
        mutation.result(timeout=10)
    restored_home = tmp_path / 'recovered'
    with zipfile.ZipFile(archive_path) as archive:
        assert 'local-rag/curated-images.db' in archive.namelist()
        assert not any(n.startswith('local-rag/images/') for n in archive.namelist())
        archive.extract('local-rag/curated-images.db', restored_home)
    restored = CuratedImages(restored_home, text=None, visual=None, ocr=None)
    assert restored.materialize() == 1
    row = restored.db.execute('SELECT * FROM curated_images').fetchone()
    assert (restored.directory / row['filename']).read_bytes() == source.read_bytes()
    assert row['ocr'] == 'ACME' and len(row['clip']) == 2048
    restored.close()



def test_save_screens_private_snapshot_before_publishing(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    images = store(tmp_path / 'home')
    def ocr(path):
        assert not list(images.directory.iterdir()), 'unapproved image must not enter native file backup'
        assert Path(path).read_bytes() == source.read_bytes()
        return 'ACME', 'fixture'
    images.ocr = ocr
    save(images, source)
    assert images.count('owner') == 1
    images.close()


def test_db_restore_preserves_history_scopes_and_scoped_original_deletion(tmp_path):
    source = tmp_path / 'source.png'
    images = store(tmp_path / 'home')
    Image.new('RGB', (20,20), 'red').save(source)
    historical = save(images, source, group='versions')
    project = images.save('owner', source, decision='save', reason='approved', scope='project', project='/a')
    session = images.save('owner', source, decision='save', reason='approved', scope='session', session='s1')
    Image.new('RGB', (20,20), 'blue').save(source)
    save(images, source, group='versions')
    before = [dict(r) for r in images.db.execute('SELECT * FROM curated_images ORDER BY id')]
    snapshot(images, tmp_path / 'restored')
    restored = CuratedImages(tmp_path / 'restored', text=None, visual=None, ocr=None)
    assert restored.materialize() == 4
    assert [dict(r) for r in restored.db.execute('SELECT * FROM curated_images ORDER BY id')] == before
    assert not images.delete('owner', project['id'], project='/b')
    assert images.reindex('owner', session['id'], session='s2') is None
    assert images.delete('owner', project['id'], project='/a')
    assert images.delete('owner', session['id'], session='s1')
    assert images.delete('owner', historical['id'])
    assert images.db.execute('SELECT original FROM curated_images WHERE hash=?', (historical['sha256'],)).fetchall() == []
    assert not Path(historical['path']).exists()
    restored.close()
    images.close()

def test_reconcile_db_failure_preserves_original_and_derivatives(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (20,20), 'red').save(source)
    images = store(tmp_path / 'home')
    saved = save(images, source)
    before = dict(images.db.execute('SELECT * FROM curated_images').fetchone())
    images.db.execute("CREATE TRIGGER fail_update BEFORE UPDATE ON curated_images BEGIN SELECT RAISE(ABORT, 'forced'); END")
    images.db.commit()
    Image.new('RGB', (20,20), 'blue').save(saved['path'])
    with pytest.raises(sqlite3.IntegrityError, match='forced'):
        images.reconcile('owner', force=True)
    assert dict(images.db.execute('SELECT * FROM curated_images').fetchone()) == before
    assert list(images.directory.iterdir()) == [Path(saved['path'])]
    assert not list((images.directory.parent / '.cache').iterdir())
    images.close()
