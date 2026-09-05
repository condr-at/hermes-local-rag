import importlib.util
import json
from pathlib import Path
from PIL import Image
import pytest
from local_rag.config import LocalRagConfig


def test_installed_local_ocr_reads_table_cell(tmp_path):
    import sys
    import shutil
    from PIL import ImageDraw, ImageFont
    from local_rag.images import local_ocr
    if not shutil.which('tesseract') and not (sys.platform == 'darwin' and shutil.which('swift')):
        pytest.skip('No installed local OCR runtime')
    image = Image.new('RGB', (1000, 200), 'white')
    ImageDraw.Draw(image).text((30, 60), 'ACME NORTH 123', fill='black', font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 48))
    path = tmp_path / 'table.png'
    image.save(path)
    text, status = local_ocr(path)
    assert 'ACME' in text.upper() and '123' in text, (text, status)


class Text:
    dimensions = 3
    def embed_query(self, text):
        return [1., 0., 0.]
    def embed_document(self, text, **kwargs):
        return [1., 0., 0.]


class Clip:
    def embed_image(self, path):
        return [1.] + [0.] * 511
    def embed_text(self, text):
        return [0., 1.] + [0.] * 510


def test_scopes_versions_derivatives_and_shared_blob_deletion(tmp_path):
    from local_rag.images import CuratedImages
    images = CuratedImages(tmp_path / 'home', text=Text(), visual=lambda: Clip(), ocr=lambda _: ('ACME NORTH', 'fixture'))
    red, blue = tmp_path / 'red.png', tmp_path / 'blue.png'
    Image.new('RGB', (30,20), 'red').save(red)
    Image.new('RGB', (30,20), 'blue').save(blue)
    def save(path, **kw):
        return images.save('owner', path, decision='save', reason='approved', **kw)
    try:
        first = save(red, scope='project', project='/a', group='design')
        second = save(blue, scope='project', project='/a', group='design')
        assert [r['id'] for r in images.search('owner', 'ACME', project='/a')] == [second['id']]
        assert len(images.search('owner', 'ACME', project='/a', include_history=True)) == 2
        assert not images.search('owner', 'ACME', project='/b')
        assert not images.search('other', 'ACME', project='/a', include_history=True)
        session = save(red, scope='session', session='s1')
        assert not images.search('owner', 'ACME', session='s2')
        assert [r['id'] for r in images.search('owner', 'ACME', session='s1')] == [session['id']]
        global_image = save(red, scope='global')
        assert images.search('owner', 'ACME', project='/b')[0]['id'] == global_image['id']
        assert not images.delete('other', global_image['id'])
        assert images.delete('owner', first['id'], project='/a')
        assert Path(first['path']).exists(), 'same bytes still referenced by another scope'
        images.delete('owner', session['id'], session='s1')
        images.delete('owner', global_image['id'])
        assert not Path(first['path']).exists()
        images.delete('owner', second['id'], project='/a')
        assert not images.search('owner', 'ACME', project='/a', include_history=True)
    finally:
        images.close()


def test_image_tool_exposes_explicit_version_history():
    from local_rag import LocalRagProvider
    schema = next(s for s in LocalRagProvider(embedder=Text()).get_tool_schemas() if s['name'] == 'local_rag_search_images')
    assert 'include_history' in schema['parameters']['properties']


def test_curated_image_is_durable_searchable_and_deleted(tmp_path):
    from local_rag import LocalRagProvider
    root = tmp_path / 'project'
    root.mkdir()
    original = root / 'logo.png'
    Image.new('RGB', (30, 20), 'red').save(original)
    home = tmp_path / 'home'
    LocalRagConfig(visual_enabled=True).save(home)
    p = LocalRagProvider(embedder=Text(), visual_embedder=Clip())
    p.initialize('s1', hermes_home=str(home), cwd=str(root))
    try:
        rejected = json.loads(p.handle_tool_call('local_rag_index_image', {'path': str(original)}))
        assert 'error' in rejected, 'image save requires explicit curated decision and scope'
        args = dict(path=str(original), decision='save', reason='Approved brand reference', scope='project', group='brand-logo', description='Table cell: Acme North revenue 123')
        saved = json.loads(p.handle_tool_call('local_rag_index_image', args))
        assert 'error' not in saved, saved
        managed = Path(saved['path'])
        assert managed.is_relative_to(home / 'local-rag' / 'images')
        assert managed.read_bytes() == original.read_bytes()
        assert json.loads(p.handle_tool_call('local_rag_status', {}))['images'] == 1
        assert json.loads(p.handle_tool_call('local_rag_index_image', args))['id'] == saved['id']
        original.unlink()
        found = json.loads(p.handle_tool_call('local_rag_search', {'query': 'Acme North'}))['results']
        assert any(r.get('path') == str(managed) for r in found), found
        assert str(managed) in p.prefetch('Acme North')
        assert json.loads(p.handle_tool_call('local_rag_forget_image', {'id': saved['id']}))['removed']
        assert not managed.exists()
        assert not json.loads(p.handle_tool_call('local_rag_search', {'query': 'Acme North'}))['results']
    finally:
        p.shutdown()
