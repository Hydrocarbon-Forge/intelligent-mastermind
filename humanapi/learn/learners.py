"""The learners bundle and its single update path.

`Learners.apply` is the only place a logged position changes the learners. The live driver and the
rebuild (humanapi/store/rebuild.py) both call it, so a replayed record has the same effect as a live one.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from humanapi.config import Config
from humanapi.learn.gate import IsotonicCalibrator
from humanapi.learn.knn import KnnIndex
from humanapi.learn.ngram import NgramMemory
from humanapi.learn.scorer import Scorer
from humanapi.learn.types import OTHER_ID, GateGroup, Outcome, PositionRecord
from humanapi.store.log import InteractionLog

log = logging.getLogger(__name__)

_LABELED = (Outcome.PICK, Outcome.TYPED, Outcome.EXPLORE)
_PROBE_SEED_VALUE = 1.0


def _row_of(candidates: Sequence[dict[str, Any]], token_id: int) -> Optional[int]:
    for row, cand in enumerate(candidates):
        if cand["id"] == token_id:
            return row
    return None


def _features(rec: PositionRecord) -> np.ndarray:
    return np.array([c["f"] for c in rec.candidates], dtype=np.float64)


class Learners:
    ngram: NgramMemory
    knn: KnnIndex
    scorer: Scorer
    calibrator: IsotonicCalibrator
    n_labels: int

    def __init__(self, cfg: Config, *, embed_dim: int, vocab_size: int) -> None:
        self.cfg = cfg
        self.embed_dim = int(embed_dim)
        self.ngram = NgramMemory(
            orders=cfg.learning.ngram_orders,
            alpha=cfg.learning.ngram_alpha,
            vocab_size=vocab_size,
        )
        self.knn = KnnIndex(dim=self.embed_dim, k=cfg.learning.knn_k, temperature=cfg.learning.knn_temperature)
        self.scorer = Scorer(cfg.learning)
        self.calibrator = IsotonicCalibrator(refit_every=cfg.gate.calibration_refit_every)
        self.n_labels = 0
        self._last_snapshot_n = -1

    # ---- the single update path

    def learn_memories(self, rec: PositionRecord, response_ids: Sequence[int]) -> None:
        """Steps (a) and (b) only: n-gram and kNN memories of a labeled record. Used by apply and by the
        reset-scorer rebuild, so the memories are never updated anywhere else."""
        if rec.outcome not in _LABELED or rec.chosen_id is None:
            return
        self.ngram.update(list(response_ids[: rec.idx]), rec.chosen_id, rec.sample_weight)
        if rec.feat_vec is not None:
            # Stored vectors are float16 in the log; use the same rounding live so replay is identical.
            vec = np.asarray(rec.feat_vec, dtype=np.float16)
            self.knn.add(vec, rec.chosen_id, rec.sample_weight)

    def apply(self, rec: PositionRecord, response_ids: Sequence[int]) -> Optional[float]:
        """Apply one logged position. Returns the pre-update p_you of the chosen token (the prequential
        prediction) or None when no scorer step happened."""
        if rec.outcome == Outcome.AUTO:
            return None

        if rec.outcome == Outcome.REWOUND:
            if rec.rejected_id is None:
                return None
            feats = _features(rec)
            row = _row_of(rec.candidates, rec.rejected_id)
            if row is None:
                log.warning("rewound record %s: rejected id %s is not among its candidates", rec.id, rec.rejected_id)
                return None
            return self.scorer.penalize(feats, row)

        if rec.outcome not in _LABELED or rec.chosen_id is None:
            return None

        self.learn_memories(rec, response_ids)

        pre: Optional[float] = None
        feats = _features(rec)
        row = _row_of(rec.candidates, rec.chosen_id)
        if row is None:
            log.warning("labeled record %s: chosen id %s is not among its candidates; scorer skipped",
                        rec.id, rec.chosen_id)
        else:
            pre = self.scorer.update(feats, row, rec.sample_weight)

        self._calibrate(rec)
        self.n_labels += 1
        return pre

    def _calibrate(self, rec: PositionRecord) -> None:
        if rec.group_name is None:
            return  # teacher-forced continuation token, not a gate decision
        best: Optional[dict[str, Any]] = None
        for cand in rec.candidates:
            if cand["id"] == OTHER_ID or cand["cls"] == "end":
                continue
            if best is None or cand["p_you"] > best["p_you"]:  # strict: ties keep the lower index
                best = cand
        if best is None:
            return
        hit = rec.chosen_id == best["id"]
        p_top = float(best["p_you"])
        group = GateGroup(rec.group_name)
        if rec.outcome == Outcome.EXPLORE:
            self.calibrator.add(group, p_top, hit, 1.0 / self.cfg.gate.explore_rate)
        elif rec.outcome in (Outcome.PICK, Outcome.TYPED):
            if rec.top_conf is not None and rec.tau is not None and rec.top_conf < rec.tau:
                self.calibrator.add(group, p_top, hit, 1.0)

    # ---- snapshots

    def maybe_snapshot(self, log_: InteractionLog) -> Optional[int]:
        n = self.scorer.n_updates
        every = self.cfg.learning.snapshot_every
        if every <= 0 or n <= 0 or n % every != 0 or n == self._last_snapshot_n:
            return None
        self._last_snapshot_n = n
        return log_.save_scorer_version(
            self.scorer.snapshot(),
            self.calibrator.to_json(),
            self.n_labels,
            notes=None,
            keep=self.cfg.learning.snapshots_kept,
        )

    def rollback(self, log_: InteractionLog, version_id: Optional[int] = None) -> bool:
        version = log_.load_scorer_version(version_id)
        if version is None:
            return False
        self.scorer.restore(version.weights, self.scorer.n_updates)
        if version.calib is not None:
            self.calibrator = IsotonicCalibrator.from_json(version.calib)
        self.n_labels = version.n_labels
        return True

    # ---- equality of learned state

    def digest(self) -> dict:
        probe = np.full(self.embed_dim, _PROBE_SEED_VALUE / np.sqrt(self.embed_dim), dtype=np.float32)
        answer = self.knn.query(probe)
        # Unigram counts are read from the n-gram memory's own table.
        unigram = sorted(self.ngram._counts.get((1, ()), {}).items())  # noqa: SLF001
        return {
            "weights": self.scorer.weights.tobytes(),
            "n_updates": int(self.scorer.n_updates),
            "ngram_total_weight": float(self.ngram.total_weight),
            "unigram": [[int(t), float(c)] for t, c in unigram],
            "knn_rows": len(self.knn),
            "knn_probe_votes": sorted([int(t), float(v)] for t, v in answer.votes.items()),
            "knn_probe_max_sim": float(answer.max_sim),
            "calibrator": self.calibrator.to_json(),
            "n_labels": int(self.n_labels),
        }
