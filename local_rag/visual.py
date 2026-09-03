from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer


class ClipOnnxEmbedder:
    dimensions = 512

    def __init__(self, model_dir: Path | None = None) -> None:
        root = model_dir or Path.home() / ".hermes" / "models" / "clip-onnx"
        self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
        self._text = ort.InferenceSession(str(root / "onnx" / "text_model_quantized.onnx"), sess_options=options, providers=providers)
        self._vision = ort.InferenceSession(str(root / "onnx" / "vision_model_quantized.onnx"), sess_options=options, providers=providers)
        self._lock = threading.Lock()

    @staticmethod
    def _normalize(vector: np.ndarray) -> list[float]:
        result = vector.astype(np.float32).reshape(-1)
        norm = np.linalg.norm(result)
        if norm:
            result /= norm
        return result.tolist()

    def embed_text(self, text: str) -> list[float]:
        ids = self._tokenizer.encode(text, add_special_tokens=True).ids[:77]
        ids.extend([0] * (77 - len(ids)))
        inputs = np.asarray([ids], dtype=np.int64)
        with self._lock:
            output = self._text.run(["text_embeds"], {"input_ids": inputs})[0][0]
        return self._normalize(output)

    def embed_image(self, path: str | Path) -> list[float]:
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        pixels = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...]
        with self._lock:
            output = self._vision.run(["image_embeds"], {"pixel_values": pixels})[0][0]
        return self._normalize(output)


_shared: ClipOnnxEmbedder | None = None
_shared_lock = threading.Lock()


def get_shared_visual_embedder() -> ClipOnnxEmbedder:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = ClipOnnxEmbedder()
        return _shared
