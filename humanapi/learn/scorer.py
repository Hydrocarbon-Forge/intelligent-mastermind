"""Log-linear user model: p_you over candidates plus OTHER, learned online with Adagrad and an L2 pull to w0.

Prequential metrics: the prediction made before each labeled update is recorded before the weights change.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

import numpy as np

from humanapi.config import LearningConfig
from humanapi.learn.candidates import assemble, unlisted_candidate
from humanapi.learn.features import N_FEATURES, build_features, initial_weights
from humanapi.learn.knn import KnnResult
from humanapi.learn.ngram import NgramMemory
from humanapi.learn.types import PositionContext, ScoredPosition
from humanapi.text.types import Tokenizer

_METRIC_WINDOW = 200
_ADAGRAD_EPS = 1e-8


class Scorer:
    def __init__(self, cfg: LearningConfig) -> None:
        self.cfg = cfg
        self.w0: np.ndarray = initial_weights()
        self.weights: np.ndarray = self.w0.copy()
        self.n_updates: int = 0
        self._g_sq: np.ndarray = np.zeros(N_FEATURES, dtype=np.float64)
        self._history: deque[tuple[bool, float]] = deque(maxlen=_METRIC_WINDOW)

    def probs(self, feats: np.ndarray) -> np.ndarray:
        s = feats @ self.weights
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    def _gradient(self, feats: np.ndarray, chosen: int, alpha: float) -> np.ndarray:
        p = self.probs(feats)
        l2 = self.cfg.l2_to_init
        return alpha * (p @ feats - feats[chosen]) + l2 * (self.weights - self.w0)

    def _step(self, grad: np.ndarray) -> None:
        self._g_sq += grad * grad
        self.weights = self.weights - self.cfg.learning_rate * grad / (np.sqrt(self._g_sq) + _ADAGRAD_EPS)

    def update(self, feats: np.ndarray, chosen: int, alpha: float) -> float:
        p = self.probs(feats)
        pre = float(p[chosen])
        correct = int(np.argmax(p)) == chosen
        surprisal_bits = -math.log2(max(pre, 1e-12))
        self._step(self._gradient(feats, chosen, alpha))
        self.n_updates += 1
        self._history.append((correct, surprisal_bits))
        return pre

    def penalize(self, feats: np.ndarray, rejected: int, alpha: float = 1.0) -> float:
        p = self.probs(feats)
        p_rej = float(p[rejected])
        coef = alpha * p_rej / max(1.0 - p_rej, 1e-9)
        grad = coef * (feats[rejected] - p @ feats) + self.cfg.l2_to_init * (self.weights - self.w0)
        self._step(grad)
        self.n_updates += 1
        return p_rej

    def snapshot(self) -> np.ndarray:
        return self.weights.copy()

    def restore(self, weights: np.ndarray, n_updates: int = 0) -> None:
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        self.n_updates = int(n_updates)
        self._g_sq = np.zeros(N_FEATURES, dtype=np.float64)

    def reset(self) -> None:
        self.weights = self.w0.copy()
        self.n_updates = 0
        self._g_sq = np.zeros(N_FEATURES, dtype=np.float64)
        self._history.clear()

    def metrics(self) -> dict[str, float]:
        if not self._history:
            return {"pick_accuracy": 0.0, "mean_surprisal_bits": 0.0}
        n = len(self._history)
        return {
            "pick_accuracy": sum(1 for c, _ in self._history if c) / n,
            "mean_surprisal_bits": sum(s for _, s in self._history) / n,
        }


def score_position(
    ctx: PositionContext,
    *,
    tok: Tokenizer,
    ngram: NgramMemory,
    knn: KnnResult | None,
    scorer: Scorer,
    cfg: LearningConfig,
) -> ScoredPosition:
    candidates = assemble(ctx, tok=tok, ngram=ngram, knn=knn, cfg=cfg)
    feats = build_features(ctx, candidates, ngram=ngram, knn=knn, cfg=cfg)
    probs = scorer.probs(feats)
    return ScoredPosition(
        candidates=candidates,
        features=feats,
        probs=probs,
        state=ctx.state,
        p_other=float(probs[-1]),
        base_k=ctx.result.k if ctx.result is not None else 0,
        tail_mass=ctx.result.tail_mass if ctx.result is not None else 1.0,
        retrieval_hit=ctx.retrieval_hit,
        retrieval_max_sim=ctx.retrieval_max_sim,
    )


def with_chosen(
    scored: ScoredPosition,
    token_id: int,
    ctx: PositionContext,
    *,
    tok: Tokenizer,
    ngram: NgramMemory,
    knn: KnnResult | None,
    scorer: Scorer,
    cfg: LearningConfig,
) -> tuple[ScoredPosition, int]:
    ids = [c.token_id for c in scored.candidates]
    if token_id in ids:
        return scored, ids.index(token_id)
    extra = unlisted_candidate(ctx, tok=tok, cfg=cfg, token_id=token_id)
    candidates = list(scored.candidates) + [extra]
    feats = build_features(ctx, candidates, ngram=ngram, knn=knn, cfg=cfg)
    probs = scorer.probs(feats)
    new = replace(
        scored,
        candidates=candidates,
        features=feats,
        probs=probs,
        p_other=float(probs[-1]),
    )
    return new, len(candidates) - 1
