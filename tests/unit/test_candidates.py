import math

import numpy as np

from humanapi.backend.base import NextTokenResult, TokenLP
from humanapi.config import LearningConfig
from humanapi.learn.candidates import assemble
from humanapi.learn.knn import KnnIndex
from humanapi.learn.ngram import NgramMemory
from humanapi.learn.scorer import Scorer, score_position, with_chosen
from humanapi.learn.types import SRC_BACKEND, SRC_END, SRC_KNN, SRC_NGRAM, SRC_PREFIX, PositionContext
from humanapi.text.toy_tokenizer import ToyTokenizer
from humanapi.text.types import PosState


def _result(tok, pairs, tail_mass, k=20):
    top = [TokenLP(token_id=t, text=tok.token_text(t) or "", raw=tok.token_bytes(t), logprob=math.log(p))
           for t, p in pairs]
    return NextTokenResult(top=top, tail_mass=tail_mass, k=k)


def _id(tok, text):
    ids = tok.encode(text)
    assert len(ids) == 1, text
    return ids[0]


def test_sources_union():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    the, yes, end = _id(tok, " the"), _id(tok, " yes"), tok.end_id
    thanks, this = _id(tok, " thanks"), _id(tok, " this")
    # Backend lists the top tokens in probability order, END included.
    result = _result(tok, [(the, 0.4), (yes, 0.2), (end, 0.1)], tail_mass=0.3)
    ngram = NgramMemory()
    ngram.update([], yes)      # n-gram top that the backend also lists
    ngram.update([], thanks)   # n-gram top that the backend does not list
    knn_index = KnnIndex(dim=2, k=4)
    knn_index.add(np.array([1.0, 0.0], dtype=np.float32), token_id=the)
    knn_index.add(np.array([0.0, 1.0], dtype=np.float32), token_id=this)
    knn = knn_index.query(np.array([1.0, 0.0], dtype=np.float32))
    ctx = PositionContext(
        response_ids=[],
        state=PosState.SENTENCE_START,
        typed_prefix="th",
        result=result,
    )
    cands = assemble(ctx, tok=tok, ngram=ngram, knn=knn, cfg=cfg)
    ids = [c.token_id for c in cands]
    assert len(ids) == len(set(ids))  # no duplicates
    assert ids[:3] == [the, yes, end]  # backend order first
    assert end in ids  # END present
    by_id = {c.token_id: c for c in cands}
    assert by_id[the].sources == frozenset({SRC_BACKEND, SRC_KNN, SRC_PREFIX})
    assert by_id[yes].sources == frozenset({SRC_BACKEND, SRC_NGRAM})
    assert by_id[end].sources == frozenset({SRC_BACKEND, SRC_END})
    assert by_id[thanks].sources == frozenset({SRC_NGRAM, SRC_PREFIX})
    assert by_id[thanks].base_unknown
    assert by_id[this].sources == frozenset({SRC_KNN, SRC_PREFIX})
    # END is always present, even when the backend and sources never mention it.
    empty_ctx = PositionContext(response_ids=[], state=PosState.MID_SENTENCE)
    only_end = assemble(empty_ctx, tok=tok, ngram=NgramMemory(), knn=None, cfg=cfg)
    assert [c.token_id for c in only_end] == [end]
    assert only_end[0].sources == frozenset({SRC_END})
    assert only_end[0].base_unknown and only_end[0].base_rank == 21


def test_unknown_base_bound():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    a, b, c = _id(tok, " the"), _id(tok, " yes"), _id(tok, " good")
    result = _result(tok, [(a, 0.5), (b, 0.2)], tail_mass=0.3, k=5)
    ngram = NgramMemory()
    ngram.update([], c)
    ctx = PositionContext(response_ids=[], state=PosState.SENTENCE_START, result=result)
    cands = {x.token_id: x for x in assemble(ctx, tok=tok, ngram=ngram, knn=None, cfg=cfg)}
    lowest = min(0.5, 0.2)
    assert cands[c].base_unknown
    assert math.isclose(cands[c].p_base, lowest, rel_tol=1e-9)
    assert cands[c].base_rank == 6  # k + 1
    assert not cands[a].base_unknown and math.isclose(cands[a].p_base, 0.5)
    assert cands[a].base_rank == 1

    # With no backend result the bound is tail_floor.
    none_ctx = PositionContext(response_ids=[], state=PosState.SENTENCE_START, result=None)
    none = {x.token_id: x for x in assemble(none_ctx, tok=tok, ngram=ngram, knn=None, cfg=cfg)}
    assert math.isclose(none[c].p_base, cfg.tail_floor)
    assert none[c].base_rank == 21


def test_missing_chosen_added():
    tok = ToyTokenizer()
    cfg = LearningConfig()
    a, b = _id(tok, " the"), _id(tok, " yes")
    z = _id(tok, "z")
    result = _result(tok, [(a, 0.6), (b, 0.3)], tail_mass=0.1)
    ctx = PositionContext(response_ids=[], state=PosState.SENTENCE_START, result=result)
    ngram = NgramMemory()
    scorer = Scorer(cfg)
    scored = score_position(ctx, tok=tok, ngram=ngram, knn=None, scorer=scorer, cfg=cfg)
    listed = [c.token_id for c in scored.candidates]
    assert z not in listed

    new, idx = with_chosen(scored, z, ctx, tok=tok, ngram=ngram, knn=None, scorer=scorer, cfg=cfg)
    assert new.candidates[idx].token_id == z
    assert new.candidates[idx].base_unknown
    assert new.features[idx, 1] == 1.0  # base_unknown slot
    assert math.isclose(new.candidates[idx].p_base, 0.3)  # lowest returned probability
    assert new.features.shape == (len(new.candidates) + 1, scored.features.shape[1])
    assert math.isclose(float(new.probs.sum()), 1.0)

    # A listed token is returned unchanged with its existing row.
    same, idx2 = with_chosen(scored, a, ctx, tok=tok, ngram=ngram, knn=None, scorer=scorer, cfg=cfg)
    assert same is scored and idx2 == listed.index(a)

    # The update raises the chosen (previously unlisted) token's probability.
    pre = scorer.update(new.features, idx, alpha=1.0)
    assert math.isclose(pre, new.probs[idx])
