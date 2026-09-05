"""Opt-in-by-installed-weights integration: no downloads, production writes or model API calls."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import pytest
from PIL import Image, ImageDraw, ImageFont
from local_rag import LocalRagProvider
from local_rag.config import LocalRagConfig
from local_rag.embedder import default_model_path
from local_rag.inference import InferenceClient


def test_subprocess_real_ocr_recall_desktop_gateway_and_shutdown():
    if sys.platform != 'darwin' or not default_model_path().is_file() or not (Path.home() / '.hermes/models/clip-onnx/onnx/vision_model_quantized.onnx').is_file():
        pytest.skip('Requires installed local models and macOS OCR')
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.e2-') as name:
        home = Path(name)
        root = home / 'project'
        root.mkdir()
        source = root / 'table.png'
        image = Image.new('RGB', (1000, 300), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 40)
        draw.text((30,30), 'Company             Revenue', font=font, fill='black')
        draw.text((30,130), 'ACME NORTH           123', font=font, fill='black')
        image.save(source)
        LocalRagConfig(visual_enabled=True).save(home)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        process = subprocess.Popen([sys.executable, '-m', 'local_rag.inference', '--home', name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        providers = []
        try:
            assert select.select([process.stdout], [], [], 10)[0]
            ready = process.stdout.readline()
            assert 'ready' in ready, process.stderr.read()
            for session, user in [('desktop', 'owner'), ('gateway', 'owner'), ('foreign', 'other')]:
                provider = LocalRagProvider()
                provider.initialize(session, hermes_home=name, cwd=str(root), user_id=user)
                providers.append(provider)
            saved = json.loads(providers[0].handle_tool_call('local_rag_index_image', dict(path=str(source), decision='save', reason='Approved table reference', scope='project', group='company-table')))
            assert 'error' not in saved, saved
            assert 'ACME' in saved['ocr'].upper(), saved
            assert saved['ocr_status'] in {'apple-vision', 'tesseract'}
            source.unlink()
            result = json.loads(providers[1].handle_tool_call('local_rag_search', {'query': 'ACME NORTH'}))
            assert any(r.get('path') == saved['path'] for r in result['results']), result
            assert saved['path'] in providers[1].prefetch('ACME NORTH')
            assert not json.loads(providers[2].handle_tool_call('local_rag_search', {'query': 'ACME NORTH'}))['results']
            providers[0].shutdown()
            assert InferenceClient(home).request('ping')['pid'] == process.pid
            assert json.loads(providers[1].handle_tool_call('local_rag_forget_image', {'id': saved['id']}))['removed']
            assert not Path(saved['path']).exists()
            assert not providers[1].prefetch('ACME NORTH')
            print(json.dumps({'service_pid': process.pid, 'ocr': saved['ocr'], 'ocr_backend': saved['ocr_status'], 'cross_client_recall': True, 'deletion': True}))
        finally:
            for provider in providers:
                provider.shutdown()
            process.terminate()
            process.wait(timeout=20)
        assert process.returncode == 0
