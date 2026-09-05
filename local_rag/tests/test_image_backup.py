import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
from PIL import Image
from local_rag.images import CuratedImages
from local_rag.tests.test_curated_images import Text, Clip


def test_native_backup_includes_images_and_derivatives_not_runtime(tmp_path, monkeypatch):
    backup = pytest.importorskip('hermes_cli.backup')
    from local_rag.inference import InferenceServer
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.bk-') as name:
        home = Path(name)
        images = CuratedImages(home, text=Text(), visual=lambda: Clip(), ocr=lambda _: ('ACME NORTH', 'test fixture'))
        source = tmp_path / 'source.png'
        Image.new('RGB', (20, 20), 'red').save(source)
        saved = images.save('owner', source, decision='save', reason='approved logo', scope='global')
        server = InferenceServer(home, backend=lambda *a, **kw: [])
        server.start()
        try:
            monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: home)
            monkeypatch.setattr(backup, 'display_hermes_home', lambda: str(home))
            monkeypatch.setattr(backup, '_collect_memory_provider_external_paths', lambda: [])
            output = tmp_path / 'backup.zip'
            backup.run_backup(SimpleNamespace(output=str(output)))
            with zipfile.ZipFile(output) as archive:
                assert 'local-rag/curated-images.db' in archive.namelist()
                assert 'local-rag/images/' + Path(saved['path']).name in archive.namelist()
                assert not any(p.endswith(('token', 'i.sock', 'owner.lock')) for p in archive.namelist())
                restore = tmp_path / 'restored'
                archive.extractall(restore)
            restored = CuratedImages(restore, text=Text(), visual=lambda: Clip())
            try:
                hits = restored.search('owner', 'ACME NORTH')
                assert hits and Path(hits[0]['path']).exists()
                assert hits[0]['ocr'] == 'ACME NORTH'
            finally:
                restored.close()
        finally:
            server.close()
            images.close()
