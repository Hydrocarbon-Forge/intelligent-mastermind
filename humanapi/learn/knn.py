"""Exact inner-product kNN index with softmax-weighted token votes (numpy only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KnnResult:
    votes: dict[int, float]  # token id -> pooled neighbour weight; sums to 1 when non-empty
    max_sim: float  # highest similarity among the neighbours, 0.0 for an empty index
    neighbors: list[tuple[int, float]]  # (row id, similarity), best first


class KnnIndex:
    def __init__(self, dim: int, k: int = 16, temperature: float = 0.05) -> None:
        self.dim = int(dim)
        self.k = int(k)
        self.temperature = float(temperature)
        self._n = 0
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._tokens = np.zeros(0, dtype=np.int64)
        self._weights = np.zeros(0, dtype=np.float64)

    def __len__(self) -> int:
        return self._n

    def _grow(self) -> None:
        cap = max(16, 2 * self._matrix.shape[0])
        matrix = np.zeros((cap, self.dim), dtype=np.float32)
        tokens = np.zeros(cap, dtype=np.int64)
        weights = np.zeros(cap, dtype=np.float64)
        matrix[: self._n] = self._matrix[: self._n]
        tokens[: self._n] = self._tokens[: self._n]
        weights[: self._n] = self._weights[: self._n]
        self._matrix, self._tokens, self._weights = matrix, tokens, weights

    def add(self, vec: np.ndarray, token_id: int, weight: float = 1.0) -> int:
        v = np.asarray(vec, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {v.shape}")
        if self._n == self._matrix.shape[0]:
            self._grow()
        row = self._n
        self._matrix[row] = v
        self._tokens[row] = token_id
        self._weights[row] = weight
        self._n += 1
        return row

    def query(self, vec: np.ndarray) -> KnnResult:
        n = self._n
        if n == 0:
            return KnnResult(votes={}, max_sim=0.0, neighbors=[])
        q = np.asarray(vec, dtype=np.float32)
        sims = (self._matrix[:n] @ q).astype(np.float64)
        # Best similarity first; ties broken by lower row id.
        order = np.lexsort((np.arange(n), -sims))[: min(self.k, n)]
        top_sims = sims[order]

        logits = top_sims / self.temperature
        logits = logits - logits.max()
        w = np.exp(logits) * self._weights[order]
        total = w.sum()
        if total > 0:
            w = w / total

        votes: dict[int, float] = {}
        for idx, row in enumerate(order):
            tok = int(self._tokens[row])
            votes[tok] = votes.get(tok, 0.0) + float(w[idx])
        neighbors = [(int(row), float(s)) for row, s in zip(order, top_sims)]
        return KnnResult(votes=votes, max_sim=float(top_sims[0]), neighbors=neighbors)
