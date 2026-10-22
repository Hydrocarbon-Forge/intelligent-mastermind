"""Feature vectors for candidates and the OTHER pseudo-candidate. Pure numpy, no I/O."""
from __future__ import annotations

import math

import numpy as np

from humanapi.config import LearningConfig
from humanapi.learn.knn import KnnResult
from humanapi.learn.ngram import NgramMemory
from humanapi.learn.types import PositionContext, Candidate
from humanapi.text.types import PosState, TokenClass

_FLOOR = 1e-9
_CLASSES = list(TokenClass)
_STATES = list(PosState)
_END_BUCKETS = 5

FEATURE_NAMES: list[str] = (
    [
        "log_p_base",
        "base_unknown",
        "base_rank",
        "ng1",
        "ng2",
        "ng3",
        "knn_vote",
        "user_unigram",
    ]
    + [f"cls_{c.value}_x_{s.value}" for c in _CLASSES for s in _STATES]
    + ["recent", "prefix_match", "retrieval_overlap"]
    + [f"end_len_{i}" for i in range(_END_BUCKETS)]
    + ["other_log_tail", "other_entropy", "other_knn_max_sim", "other_max_order", "other_bias"]
)
N_FEATURES: int = len(FEATURE_NAMES)
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
IDX_LOG_P_BASE: int = _IDX["log_p_base"]
IDX_OTHER_LOG_TAIL: int = _IDX["other_log_tail"]
_IDX_CLASS_STATE0: int = _IDX[f"cls_{_CLASSES[0].value}_x_{_STATES[0].value}"]
_IDX_END_LEN0: int = _IDX["end_len_0"]


def initial_weights() -> np.ndarray:
    w = np.zeros(N_FEATURES, dtype=np.float64)
    w[IDX_LOG_P_BASE] = 1.0
    w[IDX_OTHER_LOG_TAIL] = 1.0
    return w


def _log_floor(p: float) -> float:
    return math.log(max(p, _FLOOR))


def _length_bucket(n: int) -> int:
    if n == 0:
        return 0
    if n <= 4:
        return 1
    if n <= 19:
        return 2
    if n <= 59:
        return 3
    return 4


def _base_entropy(ctx: PositionContext) -> float:
    if ctx.result is None:
        return 0.0
    h = 0.0
    for entry in ctx.result.top:
        p = math.exp(entry.logprob)
        if p > 0:
            h -= p * math.log(p)
    return h


def build_features(
    ctx: PositionContext,
    candidates: list[Candidate],
    *,
    ngram: NgramMemory,
    knn: KnnResult | None,
    cfg: LearningConfig,
) -> np.ndarray:
    n = len(candidates)
    feats = np.zeros((n + 1, N_FEATURES), dtype=np.float64)
    state_idx = _STATES.index(ctx.state)
    window = cfg.recent_window
    recent = set(ctx.response_ids[-window:]) if window > 0 else set()
    prefix = ctx.typed_prefix
    retrieval = ctx.retrieval_ids
    total_weight = ngram.total_weight
    ng_orders = list(ngram.orders)
    end_bucket = _length_bucket(len(ctx.response_ids))

    for i, cand in enumerate(candidates):
        row = feats[i]
        tid = cand.token_id
        row[IDX_LOG_P_BASE] = _log_floor(cand.p_base)
        row[_IDX["base_unknown"]] = 1.0 if cand.base_unknown else 0.0
        row[_IDX["base_rank"]] = float(cand.base_rank)

        logs = ngram.log_probs(ctx.response_ids, tid)
        by_order = dict(zip(ng_orders, logs))
        row[_IDX["ng1"]] = max(by_order.get(1, 0.0), math.log(_FLOOR))
        row[_IDX["ng2"]] = max(by_order.get(2, 0.0), math.log(_FLOOR))
        row[_IDX["ng3"]] = max(by_order.get(3, 0.0), math.log(_FLOOR))

        row[_IDX["knn_vote"]] = knn.votes.get(tid, 0.0) if knn is not None else 0.0
        row[_IDX["user_unigram"]] = math.log(
            (ngram.unigram_count(tid) + 1.0) / (total_weight + 1000.0)
        )
        row[_IDX_CLASS_STATE0 + _CLASSES.index(cand.token_class) * len(_STATES) + state_idx] = 1.0
        row[_IDX["recent"]] = 1.0 if tid in recent else 0.0
        if prefix:
            text = cand.text[1:] if cand.text.startswith(" ") else cand.text
            row[_IDX["prefix_match"]] = 1.0 if text.startswith(prefix) else 0.0
        row[_IDX["retrieval_overlap"]] = 1.0 if tid in retrieval else 0.0
        if cand.token_class is TokenClass.END:
            row[_IDX_END_LEN0 + end_bucket] = 1.0

    other = feats[n]
    tail_mass = ctx.result.tail_mass if ctx.result is not None else 1.0
    other[IDX_OTHER_LOG_TAIL] = _log_floor(max(tail_mass, cfg.tail_floor))
    other[_IDX["other_entropy"]] = _base_entropy(ctx)
    other[_IDX["other_knn_max_sim"]] = knn.max_sim if knn is not None else 0.0
    other[_IDX["other_max_order"]] = float(ngram.max_order_seen(ctx.response_ids))
    other[_IDX["other_bias"]] = 1.0
    return feats
