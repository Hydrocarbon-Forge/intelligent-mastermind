"""The gate: turns the scorer's p_you at each position into Auto(token) or Ask.

Pure Python apart from reading `scored.probs`; no I/O. Decisions never mutate their inputs; calibration and
rewind updates happen only through the explicit `observe` / `record_*` calls.
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Optional, Sequence

from humanapi.config import GateConfig
from humanapi.learn.types import Ask, Auto, Candidate, Decision, GateGroup, ScoredPosition
from humanapi.text.types import PosState, TokenClass

_EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _pava_blocks(xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> list[tuple[float, float, float, int]]:
    """Weighted PAVA. Returns blocks as (weight_sum, weighted_mean_x, pooled_y, n_points), nondecreasing in y."""
    # Each working block: [weight, sum_w*x, sum_w*y, count]
    blocks: list[list[float]] = []
    for x, y, w in zip(xs, ys, ws):
        blocks.append([float(w), float(w) * float(x), float(w) * float(y), 1])
        while len(blocks) >= 2:
            w2, wx2, wy2, n2 = blocks[-1]
            w1, wx1, wy1, n1 = blocks[-2]
            if wy1 / w1 > wy2 / w2:
                blocks[-2] = [w1 + w2, wx1 + wx2, wy1 + wy2, n1 + n2]
                blocks.pop()
            else:
                break
    return [(w, wx / w, wy / w, int(n)) for w, wx, wy, n in blocks]


def pava(xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> list[float]:
    """Weighted isotonic regression (pool adjacent violators). `xs` must be sorted ascending.

    Returns the fitted value for each input position, nondecreasing.
    """
    out: list[float] = []
    for _, _, fit, n in _pava_blocks(xs, ys, ws):
        out.extend([fit] * n)
    return out


class IsotonicCalibrator:
    """Per-group isotonic map from p_top to the measured hit rate."""

    def __init__(self, refit_every: int = 50):
        self.refit_every = int(refit_every)
        self._obs: dict[GateGroup, list[tuple[float, bool, float]]] = {}
        self._fits: dict[GateGroup, tuple[list[float], list[float]]] = {}

    def add(self, group: GateGroup, p_top: float, hit: bool, weight: float = 1.0) -> None:
        obs = self._obs.setdefault(group, [])
        obs.append((float(p_top), bool(hit), float(weight)))
        if self.refit_every > 0 and len(obs) % self.refit_every == 0:
            self.refit(group)

    def refit(self, group: Optional[GateGroup] = None) -> None:
        groups = [group] if group is not None else list(self._obs.keys())
        for g in groups:
            obs = self._obs.get(g)
            if not obs:
                continue
            ordered = sorted(obs, key=lambda o: o[0])
            xs = [o[0] for o in ordered]
            ys = [1.0 if o[1] else 0.0 for o in ordered]
            ws = [o[2] for o in ordered]
            blocks = _pava_blocks(xs, ys, ws)
            self._fits[g] = ([b[1] for b in blocks], [b[2] for b in blocks])

    def apply(self, p: float, group: GateGroup) -> float:
        fit = self._fits.get(group)
        if fit is None:
            return float(p)
        kx, ky = fit
        if p <= kx[0]:
            return _clamp01(ky[0])
        if p >= kx[-1]:
            return _clamp01(ky[-1])
        # Find the segment [kx[i], kx[i+1]] containing p.
        lo, hi = 0, len(kx) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if kx[mid] <= p:
                lo = mid
            else:
                hi = mid
        width = kx[hi] - kx[lo]
        if width <= 0.0:
            return _clamp01(ky[lo])
        t = (p - kx[lo]) / width
        return _clamp01(ky[lo] + t * (ky[hi] - ky[lo]))

    def n_labels(self, group: GateGroup) -> int:
        return len(self._obs.get(group, ()))

    def to_json(self) -> dict:
        return {
            "refit_every": self.refit_every,
            "obs": {g.value: [[p, h, w] for p, h, w in obs] for g, obs in self._obs.items()},
            "fits": {g.value: {"x": list(x), "y": list(y)} for g, (x, y) in self._fits.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "IsotonicCalibrator":
        cal = cls(refit_every=int(data.get("refit_every", 50)))
        for name, obs in data.get("obs", {}).items():
            cal._obs[GateGroup(name)] = [(float(p), bool(h), float(w)) for p, h, w in obs]
        for name, fit in data.get("fits", {}).items():
            cal._fits[GateGroup(name)] = ([float(v) for v in fit["x"]], [float(v) for v in fit["y"]])
        return cal


def _clamp01(v: float) -> float:
    return min(max(float(v), 0.0), 1.0)


class RewindController:
    """Raises a logit boost on every threshold while the rewind rate is above the ceiling."""

    def __init__(self, window: int, ceiling_per_100: float, step: float, max_boost: float):
        self.window = int(window)
        self.ceiling_per_100 = float(ceiling_per_100)
        self.step = float(step)
        self.max_boost = float(max_boost)
        self.boost: float = 0.0
        self._auto: deque[bool] = deque(maxlen=self.window)  # True = rewound
        self._since_check = 0

    def record_auto(self) -> None:
        self._auto.append(False)
        self._since_check += 1
        check_every = max(1, self.window // 10)
        if self._since_check >= check_every:
            self._since_check = 0
            if self.rate() <= self.ceiling_per_100 / 2.0:
                self.boost = max(0.0, self.boost - self.step)

    def record_rewind(self) -> None:
        # Mark the most recent auto token in the window that is not already rewound.
        for i in range(len(self._auto) - 1, -1, -1):
            if not self._auto[i]:
                self._auto[i] = True
                break
        else:
            return
        if self.rate() > self.ceiling_per_100:
            self.boost = min(self.boost + self.step, self.max_boost)

    def rate(self) -> float:
        if not self._auto:
            return 0.0
        rewound = sum(1 for r in self._auto if r)
        return 100.0 * rewound / len(self._auto)


class Thresholds:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg

    def _base(self, group: GateGroup) -> float:
        if group is GateGroup.CONTENT:
            return self.cfg.tau_content
        if group is GateGroup.FUNCTION:
            return self.cfg.tau_function
        return self.cfg.tau_sentence_start

    def tau(self, group: GateGroup, *, autonomy: float = 0.0, boost: float = 0.0, abstain: bool = False) -> float:
        base = self._base(group)
        if abstain and group is GateGroup.CONTENT:
            base = min(base + self.cfg.abstention_margin, 0.999)
        return sigmoid(logit(base) + autonomy + boost)


class Gate:
    def __init__(
        self,
        cfg: GateConfig,
        calibrator: Optional[IsotonicCalibrator] = None,
        rng: Optional[random.Random] = None,
    ):
        self.cfg = cfg
        self.calibrator = calibrator if calibrator is not None else IsotonicCalibrator(cfg.calibration_refit_every)
        self.controller = RewindController(
            window=cfg.rewind_window,
            ceiling_per_100=cfg.rewind_ceiling_per_100,
            step=cfg.rewind_step,
            max_boost=cfg.rewind_max_boost,
        )
        self.thresholds = Thresholds(cfg)
        self.rng = rng if rng is not None else random.Random()
        self.autonomy: float = float(cfg.autonomy)

    def group_of(self, cand: Candidate, state: PosState) -> GateGroup:
        if state is PosState.SENTENCE_START:
            return GateGroup.SENTENCE_START
        if cand.token_class in (TokenClass.CONTINUATION, TokenClass.PUNCT, TokenClass.LAYOUT):
            return GateGroup.FUNCTION
        return GateGroup.CONTENT

    def is_factual(self, cand: Candidate, state: PosState) -> bool:
        if cand.token_class is TokenClass.DIGIT:
            return True
        if cand.token_class not in (TokenClass.WORD_START, TokenClass.CONTINUATION):
            return False
        if state is PosState.SENTENCE_START:
            return False
        stripped = cand.text.strip()
        return bool(stripped) and stripped[0].isupper()

    def decide(self, scored: ScoredPosition, *, auto_run_len: int = 0, force_ask: bool = False) -> Decision:
        # Exactly one draw per call, first, so seeded runs stay reproducible even when the result is forced.
        draw = self.rng.random()
        state = scored.state

        top_idx: Optional[int] = None
        for i, cand in enumerate(scored.candidates):
            if cand.token_class is TokenClass.END:
                continue
            if top_idx is None or float(scored.probs[i]) > float(scored.probs[top_idx]):
                top_idx = i

        if top_idx is None:
            group = GateGroup.CONTENT
            tau = self.thresholds.tau(group, autonomy=self.autonomy, boost=self.controller.boost)
            top_id, p_top, conf, top_cand = None, 0.0, 0.0, None
        else:
            top_cand = scored.candidates[top_idx]
            top_id = top_cand.token_id
            p_top = float(scored.probs[top_idx])
            group = self.group_of(top_cand, state)
            abstain = (
                self.is_factual(top_cand, state)
                and not scored.retrieval_hit
                and group is GateGroup.CONTENT
            )
            conf = self.calibrator.apply(p_top, group)
            tau = self.thresholds.tau(
                group, autonomy=self.autonomy, boost=self.controller.boost, abstain=abstain
            )

        def result(action: Any) -> Decision:
            return Decision(action=action, top_id=top_id, p_top=p_top, conf=conf, tau=tau, group=group)

        if force_ask or auto_run_len >= self.cfg.max_auto_run:
            return result(Ask())
        if draw < self.cfg.explore_rate:
            return result(Ask(explore=True))

        if top_cand is None:
            return result(Ask())

        if self.calibrator.n_labels(group) < self.cfg.cold_start_min_labels:
            if (
                top_cand.token_class is TokenClass.CONTINUATION
                and top_cand.p_base is not None
                and top_cand.p_base >= self.cfg.cold_start_base_prob
            ):
                return result(Auto(top_id))
            return result(Ask())

        if conf >= tau:
            return result(Auto(top_id))
        return result(Ask())

    def observe(self, decision: Decision, hit: bool, *, explore: bool) -> None:
        if decision.top_id is None:
            return
        weight = 1.0 / self.cfg.explore_rate if explore else 1.0
        self.calibrator.add(decision.group, decision.p_top, hit, weight)
