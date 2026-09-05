from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from tokenizers import Tokenizer


_shared_embedders: dict = {}
_shared_lock = threading.Lock()


def default_model_dir() -> Path:
    return Path(os.environ.get("HERMES_EMBEDDINGGEMMA_DIR", "~/.hermes/models/embeddinggemma-litert")).expanduser()


def default_model_path() -> Path:
    return default_model_dir() / "embeddinggemma-300M_seq512_mixed-precision.tflite"


class LiteRTEmbeddingGemma:
    sequence_length = 512

    def __init__(self, *, dimensions: int = 512, num_threads: int = 4) -> None:
        if dimensions not in {128, 256, 512, 768}:
            raise ValueError("dimensions must be one of 128, 256, 512, 768")
        model_dir = default_model_dir()
        self.dimensions = dimensions
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._interpreter = Interpreter(model_path=str(default_model_path()), num_threads=num_threads)
        self._interpreter.allocate_tensors()
        self._runner = self._interpreter.get_signature_runner("embed_512")
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        return self._embed(f"task: search result | query: {text}")

    def embed_document(self, text: str, *, title: str = "none") -> list[float]:
        return self._embed(f"title: {title} | text: {text}")

    def _embed(self, text: str) -> list[float]:
        ids = self._tokenizer.encode(text, add_special_tokens=True).ids[: self.sequence_length]
        ids.extend([0] * (self.sequence_length - len(ids)))
        inputs = np.asarray([ids], dtype=np.int32)
        with self._lock:
            output = self._runner(text_batch=inputs)["encodings"][0][: self.dimensions]
        norm = float(np.linalg.norm(output))
        if norm:
            output = output / norm
        return output.astype(np.float32, copy=False).tolist()


def get_shared_embedder(dimensions: int = 512):
    """Compatibility factory: only the service may own resident model instances."""
    from .inference import InferenceClient
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser().resolve()
    key = (str(home), dimensions)
    with _shared_lock:
        if key not in _shared_embedders:
            _shared_embedders[key] = InferenceClient(home, dimensions=dimensions)
        return _shared_embedders[key]
