"""Text embedders. HashEmbedder is deterministic and model-free (tests); the sentence
transformer is imported lazily so the package imports without the dependency."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_WORD = re.compile(r"\w+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (n, dim) float32, rows L2-normalised
        ...


def _features(text: str) -> list[str]:
    low = text.lower()
    feats = ["w:" + w for w in _WORD.findall(low)]
    feats.extend("c:" + low[i : i + 3] for i in range(len(low) - 2))
    return feats


class HashEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = int(dim)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in _features(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            # No features (e.g. the empty string): a fixed unit vector, never zero.
            vec[:] = 0.0
            vec[0] = 1.0
            return vec
        return vec / norm

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts]).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        dim: int = 1024,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: runtime only

        self.model_name = model_name
        self.device = device
        self.dim = int(dim)
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(out, dtype=np.float32).reshape(len(texts), self.dim)
