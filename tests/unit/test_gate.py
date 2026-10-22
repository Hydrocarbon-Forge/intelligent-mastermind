"""Unit tests for humanapi.learn.gate (thresholds, cold start, calibration, rewind controller)."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from humanapi.config import GateConfig
from humanapi.learn.gate import Gate, IsotonicCalibrator, RewindController, Thresholds, logit, pava, sigmoid
from humanapi.learn.types import Ask, Auto, Candidate, Decision, GateGroup, ScoredPosition
from humanapi.text.types import PosState, TokenClass


def cand(token_id: int, text: str, cls: TokenClass, p_base: float | None = None) -> Candidate:
    return Candidate(
        token_id=token_id,
        text=text,
        p_base=p_base,
        base_unknown=p_base is None,
        base_rank=0,
        token_class=cls,
        sources=frozenset({"backend"}),
    )


def scored(
    cands: list[Candidate],
    probs: list[float],
    *,
    state: PosState = PosState.MID_SENTENCE,
    retrieval_hit: bool = False,
) -> ScoredPosition:
    assert len(probs) == len(cands) + 1, "probs must include the OTHER entry last"
    p = np.asarray(probs, dtype=float)
    return ScoredPosition(
        candidates=cands,
        features=np.zeros((len(cands) + 1, 1)),
        probs=p,
        state=state,
        p_other=float(p[-1]),
        base_k=len(cands),
        tail_mass=0.0,
        retrieval_hit=retrieval_hit,
        retrieval_max_sim=0.0,
    )


def gate_cfg(**overrides) -> GateConfig:
    # cold start off and exploration off unless a test turns them on
    base = dict(cold_start_min_labels=0, explore_rate=0.0)
    base.update(overrides)
    return GateConfig(**base)


def test_thresholds_by_group():
    cfg = GateConfig()
    th = Thresholds(cfg)
    assert th.tau(GateGroup.CONTENT) == pytest.approx(0.80)
    assert th.tau(GateGroup.FUNCTION) == pytest.approx(0.60)
    assert th.tau(GateGroup.SENTENCE_START) == pytest.approx(0.85)
    # abstention raises only the content threshold, by the margin
    assert th.tau(GateGroup.CONTENT, abstain=True) == pytest.approx(0.90)
    assert th.tau(GateGroup.FUNCTION, abstain=True) == pytest.approx(0.60)
    assert th.tau(GateGroup.SENTENCE_START, abstain=True) == pytest.approx(0.85)


def test_group_of():
    gate = Gate(gate_cfg())
    word = cand(1, " the", TokenClass.WORD_START)
    assert gate.group_of(word, PosState.MID_SENTENCE) is GateGroup.CONTENT
    assert gate.group_of(word, PosState.SENTENCE_START) is GateGroup.SENTENCE_START
    assert gate.group_of(cand(2, "ing", TokenClass.CONTINUATION), PosState.MID_WORD) is GateGroup.FUNCTION
    assert gate.group_of(cand(3, ",", TokenClass.PUNCT), PosState.AFTER_PUNCT) is GateGroup.FUNCTION
    assert gate.group_of(cand(4, "\n", TokenClass.LAYOUT), PosState.MID_SENTENCE) is GateGroup.FUNCTION
    assert gate.group_of(cand(5, "2", TokenClass.DIGIT), PosState.MID_SENTENCE) is GateGroup.CONTENT


def test_end_never_auto():
    gate = Gate(gate_cfg())
    end = cand(9, "", TokenClass.END)
    word = cand(1, " yes", TokenClass.WORD_START)
    # END has the highest probability, yet the top non-END candidate is chosen
    pos = scored([end, word], [0.99, 0.9, 0.01])
    d = gate.decide(pos)
    assert d.top_id == 1
    assert d.action == Auto(1)
    # END alone is never auto-selected and there is no top candidate
    only_end = scored([end], [0.99, 0.01])
    d2 = gate.decide(only_end)
    assert d2.top_id is None
    assert d2.action == Ask()
    assert d2.p_top == 0.0 and d2.conf == 0.0
    assert d2.group is GateGroup.CONTENT


def test_other_never_auto():
    gate = Gate(gate_cfg())
    # OTHER carries most of the mass; the top real candidate still has conf below tau
    pos = scored([cand(1, " yes", TokenClass.WORD_START)], [0.2, 0.8])
    assert gate.decide(pos).action == Ask()


def test_cold_start():
    cfg = GateConfig(cold_start_min_labels=200, explore_rate=0.0)
    gate = Gate(cfg)
    cont_hi = scored([cand(1, "ing", TokenClass.CONTINUATION, p_base=0.97)], [0.99, 0.01])
    cont_lo = scored([cand(1, "ing", TokenClass.CONTINUATION, p_base=0.90)], [0.99, 0.01])
    word_hi = scored([cand(2, " the", TokenClass.WORD_START, p_base=0.99)], [0.99, 0.01])
    # below cold_start_base_prob or wrong class: ask, whatever the probability
    assert gate.decide(cont_hi).action == Auto(1)
    assert gate.decide(cont_lo).action == Ask()
    assert gate.decide(word_hi).action == Ask()
    # once the group has enough labels the threshold rule takes over
    for _ in range(200):
        gate.calibrator.add(GateGroup.CONTENT, 0.99, True)
    assert gate.decide(word_hi).action == Auto(2)


def test_slider_logit_shift():
    cfg = gate_cfg()
    th = Thresholds(cfg)
    for delta in (-2.0, -0.5, 0.0, 0.5, 2.0):
        expected = sigmoid(logit(0.80) + delta)
        assert th.tau(GateGroup.CONTENT, autonomy=delta) == pytest.approx(expected)
        assert 0.0 < th.tau(GateGroup.CONTENT, autonomy=delta) < 1.0
    gate = Gate(cfg)
    gate.autonomy = 2.0
    pos = scored([cand(1, " yes", TokenClass.WORD_START)], [0.5, 0.5])
    d = gate.decide(pos)
    assert d.tau == pytest.approx(sigmoid(logit(0.80) + 2.0))
    assert 0.0 < d.tau < 1.0


def test_explore_rate():
    rate = 0.2
    n = 5000
    gate = Gate(gate_cfg(explore_rate=rate), rng=random.Random(12345))
    pos = scored([cand(1, " yes", TokenClass.WORD_START)], [0.99, 0.01])
    explores = 0
    for _ in range(n):
        d = gate.decide(pos)
        if d.action == Ask(explore=True):
            explores += 1
    expected = n * rate
    sd = math.sqrt(n * rate * (1 - rate))
    assert abs(explores - expected) <= 4 * sd


def test_isotonic_monotonic():
    # exact small example: alternating hits with equal weights pool to 0.5 everywhere
    assert pava([1, 2, 3, 4], [1, 0, 1, 0], [1, 1, 1, 1]) == pytest.approx([0.5, 0.5, 0.5, 0.5])
    # weighted: [0, 1, 0] with weights [1, 1, 3] -> [0, 0.25, 0.25]
    assert pava([1, 2, 3], [0, 1, 0], [1, 1, 3]) == pytest.approx([0.0, 0.25, 0.25])

    rng = np.random.default_rng(0)
    cal = IsotonicCalibrator(refit_every=0)
    for _ in range(400):
        p = float(rng.random())
        hit = bool(rng.random() < p)
        cal.add(GateGroup.CONTENT, p, hit)
    cal.refit()
    grid = np.linspace(0.0, 1.0, 201)
    vals = [cal.apply(float(g), GateGroup.CONTENT) for g in grid]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_isotonic_degenerate_inputs():
    cal = IsotonicCalibrator(refit_every=0)
    assert cal.apply(0.3, GateGroup.FUNCTION) == pytest.approx(0.3)  # no fit: identity
    cal.add(GateGroup.FUNCTION, 0.7, True)
    cal.refit()
    assert cal.apply(0.1, GateGroup.FUNCTION) == pytest.approx(1.0)
    assert cal.apply(0.9, GateGroup.FUNCTION) == pytest.approx(1.0)
    cal2 = IsotonicCalibrator(refit_every=0)
    for _ in range(3):
        cal2.add(GateGroup.CONTENT, 0.5, False)
    cal2.refit()
    assert cal2.apply(0.5, GateGroup.CONTENT) == pytest.approx(0.0)


def test_importance_weights():
    cal = IsotonicCalibrator(refit_every=0)
    cal.add(GateGroup.CONTENT, 0.5, True, weight=3.0)
    cal.add(GateGroup.CONTENT, 0.5, False, weight=1.0)
    cal.refit()
    assert cal.n_labels(GateGroup.CONTENT) == 2  # unweighted count
    assert cal.apply(0.5, GateGroup.CONTENT) == pytest.approx(0.75)

    # Gate.observe weights exploration labels by 1 / explore_rate
    gate = Gate(gate_cfg(explore_rate=0.05))
    top = Decision(action=Ask(explore=True), top_id=1, p_top=0.9, conf=0.9, tau=0.8, group=GateGroup.CONTENT)
    gate.observe(top, hit=True, explore=True)  # weight 20
    gate.observe(top, hit=False, explore=False)  # weight 1
    gate.calibrator.refit(GateGroup.CONTENT)
    assert gate.calibrator.apply(0.9, GateGroup.CONTENT) == pytest.approx(20.0 / 21.0)

    # decisions without a top candidate are ignored
    empty = Decision(action=Ask(), top_id=None, p_top=0.0, conf=0.0, tau=0.8, group=GateGroup.CONTENT)
    gate.observe(empty, hit=True, explore=True)
    assert gate.calibrator.n_labels(GateGroup.CONTENT) == 2


def test_calibrator_json_round_trip():
    cal = IsotonicCalibrator(refit_every=0)
    for p, h in [(0.2, False), (0.6, True), (0.8, True), (0.4, False)]:
        cal.add(GateGroup.FUNCTION, p, h, weight=2.0)
    cal.refit()
    restored = IsotonicCalibrator.from_json(cal.to_json())
    for p in (0.0, 0.3, 0.5, 0.9, 1.0):
        assert restored.apply(p, GateGroup.FUNCTION) == pytest.approx(cal.apply(p, GateGroup.FUNCTION))
    assert restored.n_labels(GateGroup.FUNCTION) == 4


def test_rewind_controller():
    ctl = RewindController(window=20, ceiling_per_100=5.0, step=0.25, max_boost=1.5)
    for _ in range(20):
        ctl.record_auto()
    assert ctl.rate() == 0.0
    ctl.record_rewind()  # 5 percent: not above the ceiling
    assert ctl.boost == 0.0
    boosts = []
    for _ in range(7):  # rate climbs above the ceiling and keeps rising
        ctl.record_rewind()
        boosts.append(ctl.boost)
    assert boosts == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.5])  # rises, then caps

    # the rewound tokens age out of the window; the rate recovers and the boost relaxes
    seen = [ctl.boost]
    for _ in range(40):
        ctl.record_auto()
        seen.append(ctl.boost)
    assert all(b2 <= b1 for b1, b2 in zip(seen, seen[1:]))
    assert ctl.rate() == 0.0
    assert ctl.boost == pytest.approx(0.0)


def test_rewind_empty_window_is_noop():
    ctl = RewindController(window=10, ceiling_per_100=5.0, step=0.25, max_boost=1.0)
    ctl.record_rewind()
    assert ctl.rate() == 0.0
    assert ctl.boost == 0.0


def test_max_auto_run():
    gate = Gate(GateConfig(cold_start_min_labels=0, explore_rate=0.0, max_auto_run=40))
    pos = scored([cand(1, " yes", TokenClass.WORD_START)], [0.999, 0.001])
    assert gate.decide(pos, auto_run_len=39).action == Auto(1)
    d = gate.decide(pos, auto_run_len=40)
    assert d.action == Ask()
    assert d.top_id == 1  # the decision still reports the computed top
    assert gate.decide(pos, force_ask=True).action == Ask()


def test_abstention_margin():
    gate = Gate(gate_cfg())
    # " Paris" is a factual-looking capitalised word; p_top 0.82 sits between 0.80 and 0.90
    pos_no_hit = scored(
        [cand(1, " Paris", TokenClass.WORD_START), cand(2, " the", TokenClass.WORD_START)],
        [0.82, 0.10, 0.08],
        retrieval_hit=False,
    )
    d = gate.decide(pos_no_hit)
    assert d.tau == pytest.approx(0.90)
    assert d.action == Ask()

    pos_hit = scored(
        [cand(1, " Paris", TokenClass.WORD_START), cand(2, " the", TokenClass.WORD_START)],
        [0.82, 0.10, 0.08],
        retrieval_hit=True,
    )
    d2 = gate.decide(pos_hit)
    assert d2.tau == pytest.approx(0.80)
    assert d2.action == Auto(1)


def test_decide_draws_exactly_once():
    class CountingRng(random.Random):
        def __init__(self):
            super().__init__(7)
            self.draws = 0

        def random(self):
            self.draws += 1
            return super().random()

    rng = CountingRng()
    gate = Gate(gate_cfg(), rng=rng)
    pos = scored([cand(1, " yes", TokenClass.WORD_START)], [0.99, 0.01])
    gate.decide(pos)
    gate.decide(pos, force_ask=True)
    gate.decide(pos, auto_run_len=100)
    assert rng.draws == 3
