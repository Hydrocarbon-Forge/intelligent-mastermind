import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from humanapi.backend.base import NextTokenResult, TokenLP
from humanapi.config import LearningConfig
from humanapi.learn.features import FEATURE_NAMES, IDX_LOG_P_BASE, N_FEATURES
from humanapi.learn.ngram import NgramMemory
from humanapi.learn.scorer import Scorer, score_position, with_chosen
from humanapi.learn.types import PositionContext
from humanapi.text.toy_tokenizer import ToyTokenizer
from humanapi.text.types import PosState


def _result(tok, pairs, tail_mass, k=20):
    top = [
        TokenLP(token_id=t, text=tok.token_text(t) or "", raw=tok.token_bytes(t), logprob=math.log(p))
        for t, p in sorted(pairs, key=lambda x: -x[1])
    ]
    return NextTokenResult(top=top, tail_mass=tail_mass, k=k)


def _id(tok, text):
    ids = tok.encode(text)
    assert len(ids) == 1, text
    return ids[0]


def _score(tok, ctx, scorer, cfg, ngram=None):
    return score_position(
        ctx, tok=tok, ngram=ngram or NgramMemory(), knn=None, scorer=scorer, cfg=cfg
    )


@settings(max_examples=100, deadline=None)
@given(
    raw=st.lists(st.floats(0.01, 1.0), min_size=1, max_size=7),
    total=st.floats(0.05, 1.0),
)
def test_initial_equals_base(raw, total):
    tok = ToyTokenizer()
    cfg = LearningConfig()
    end = tok.end_id
    letters = [97 + i for i in range(len(raw))]  # byte ids for 'a', 'b', ...
    ids = [end] + letters
    values = np.array([1.0] + raw)
    p = total * values / values.sum()
    tail = max(0.0, 1.0 - float(p.sum()))
    pairs = list(zip(ids, p.tolist()))
    ctx = PositionContext(
        response_ids=[],
        state=PosState.SENTENCE_START,
        result=_result(tok, pairs, tail_mass=tail),
    )
    scored = _score(tok, ctx, Scorer(cfg), cfg)
    listed = {c.token_id: float(pp) for c, pp in zip(scored.candidates, scored.probs[:-1])}
    assert set(listed) == set(ids)
    floor_tail = max(tail, cfg.tail_floor)
    denom = float(p.sum()) + floor_tail
    for token_id, prob in zip(ids, p):
        assert math.isclose(listed[token_id], prob / denom, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(scored.p_other, floor_tail / denom, rel_tol=1e-9)


def test_filtered_tail_floor():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    the, yes, end = _id(tok, " the"), _id(tok, " yes"), tok.end_id
    ctx = PositionContext(
        response_ids=[],
        state=PosState.SENTENCE_START,
        result=_result(tok, [(the, 0.6), (yes, 0.3), (end, 0.1)], tail_mass=0.0),
        filtered_probs=True,
    )
    scorer = Scorer(cfg)
    scored = _score(tok, ctx, scorer, cfg)
    other_row = len(scored.candidates)
    start = scored.p_other
    assert math.isclose(start, cfg.tail_floor / (1.0 + cfg.tail_floor), rel_tol=1e-6)
    for _ in range(30):
        scorer.update(scored.features, other_row, alpha=1.0)  # the person typed something unlisted
    after = float(scorer.probs(scored.features)[-1])
    assert after > start * 5
    assert after > 0.1


def test_gradient_numeric():
    rng = np.random.default_rng(0)
    cfg = LearningConfig()
    scorer = Scorer(cfg)
    scorer.weights = scorer.w0 + rng.normal(scale=0.3, size=N_FEATURES)
    feats = rng.normal(size=(5, N_FEATURES))
    chosen, alpha = 2, 0.7

    def loss(w):
        s = feats @ w
        s = s - s.max()
        log_p = s - np.log(np.exp(s).sum())
        return -alpha * log_p[chosen] + 0.5 * cfg.l2_to_init * float(np.sum((w - scorer.w0) ** 2))

    analytic = scorer._gradient(feats, chosen, alpha)
    eps = 1e-6
    numeric = np.zeros(N_FEATURES)
    for j in range(N_FEATURES):
        plus = scorer.weights.copy()
        minus = scorer.weights.copy()
        plus[j] += eps
        minus[j] -= eps
        numeric[j] = (loss(plus) - loss(minus)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_update_raises_chosen():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    the, yes, good = _id(tok, " the"), _id(tok, " yes"), _id(tok, " good")
    ctx = PositionContext(
        response_ids=[],
        state=PosState.SENTENCE_START,
        result=_result(tok, [(the, 0.5), (yes, 0.3), (good, 0.1)], tail_mass=0.1),
    )
    scorer = Scorer(cfg)
    scored = _score(tok, ctx, scorer, cfg)
    chosen = [c.token_id for c in scored.candidates].index(yes)
    before = scorer.probs(scored.features)
    pre = scorer.update(scored.features, chosen, alpha=1.0)
    after = scorer.probs(scored.features)
    assert math.isclose(pre, before[chosen])
    assert after[chosen] > before[chosen]
    # The others lose the mass the chosen token gained. Individual others can rise when they share
    # features with the chosen token (same token class and state), so only the group is checked.
    others = [j for j in range(len(before)) if j != chosen]
    assert after[others].sum() < before[others].sum()


def test_l2_pulls_to_init():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    scorer = Scorer(cfg)
    # The chosen row dominates so the likelihood gradient is essentially zero.
    feats = np.zeros((3, N_FEATURES))
    feats[0, IDX_LOG_P_BASE] = 50.0
    k = FEATURE_NAMES.index("recent")
    scorer.weights = scorer.w0.copy()
    scorer.weights[k] = 3.0
    start = abs(scorer.weights[k] - scorer.w0[k])
    for _ in range(40):
        scorer.update(feats, 0, alpha=1.0)
    assert abs(scorer.weights[k] - scorer.w0[k]) < 0.5 * start
    assert scorer.weights[k] < 3.0


def test_rewind_penalty():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    the, yes, good = _id(tok, " the"), _id(tok, " yes"), _id(tok, " good")
    ctx = PositionContext(
        response_ids=[],
        state=PosState.SENTENCE_START,
        result=_result(tok, [(the, 0.5), (yes, 0.3), (good, 0.1)], tail_mass=0.1),
    )
    scorer = Scorer(cfg)
    scored = _score(tok, ctx, scorer, cfg)
    ids = [c.token_id for c in scored.candidates]
    bad, replacement = ids.index(the), ids.index(good)
    p_bad_before = scorer.probs(scored.features)[bad]
    returned = scorer.penalize(scored.features, bad, alpha=1.0)
    assert math.isclose(returned, p_bad_before)
    p_bad_after = scorer.probs(scored.features)[bad]
    assert p_bad_after < p_bad_before
    # The replacement is a normal label.
    p_rep_before = scorer.probs(scored.features)[replacement]
    scorer.update(scored.features, replacement, alpha=1.0)
    assert scorer.probs(scored.features)[replacement] > p_rep_before


def test_replay_determinism():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    a, b, c = _id(tok, " the"), _id(tok, " yes"), _id(tok, " good")
    steps = [a, b, c, a, a, b] * 4

    def run():
        scorer = Scorer(cfg)
        ngram = NgramMemory()
        history = []
        response: list[int] = []
        for chosen_id in steps:
            ctx = PositionContext(
                response_ids=list(response),
                state=PosState.MID_SENTENCE,
                result=_result(tok, [(a, 0.4), (b, 0.3), (c, 0.2)], tail_mass=0.1),
            )
            scored = _score(tok, ctx, scorer, cfg, ngram=ngram)
            ids = [x.token_id for x in scored.candidates]
            if chosen_id in ids:
                row = ids.index(chosen_id)
                history.append(scorer.update(scored.features, row, alpha=1.0))
            ngram.update(response, chosen_id)
            response.append(chosen_id)
        return scorer.weights.copy(), history

    w1, h1 = run()
    w2, h2 = run()
    assert np.array_equal(w1, w2)
    assert h1 == h2


def test_recent_feature_learns_penalty():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    the, yes, good = _id(tok, " the"), _id(tok, " yes"), _id(tok, " good")
    recent_idx = FEATURE_NAMES.index("recent")

    def ctx_for(response):
        # Backend applies a presence penalty: " yes" is much less likely once it is in the context.
        p_yes = 0.05 if yes in response else 0.3
        return PositionContext(
            response_ids=list(response),
            state=PosState.MID_SENTENCE,
            result=_result(tok, [(the, 0.5), (yes, p_yes), (good, 0.1)], tail_mass=0.1),
        )

    scorer = Scorer(cfg)
    repeated = [yes]

    def p_yes(sc):
        scored = _score(tok, ctx_for(repeated), sc, cfg)
        ids = [c.token_id for c in scored.candidates]
        return float(sc.probs(scored.features)[ids.index(yes)])

    before = p_yes(scorer)
    response: list[int] = []
    for _ in range(60):
        ctx = ctx_for(response)
        scored = _score(tok, ctx, scorer, cfg)
        ids = [c.token_id for c in scored.candidates]
        scorer.update(scored.features, ids.index(yes), alpha=1.0)  # the person keeps choosing " yes"
        response.append(yes)
    after = p_yes(scorer)
    assert scorer.weights[recent_idx] > 0
    assert after > before
