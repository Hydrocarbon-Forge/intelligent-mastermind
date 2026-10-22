import math

import pytest

from humanapi.learn.ngram import START_ID, NgramMemory


def test_ngram_counts_and_backoff():
    m = NgramMemory(orders=(1, 2, 3), alpha=0.1, vocab_size=10, backoff_strength=1.0)
    m.update([], 5)
    m.update([5], 6)
    m.update([5], 6)
    assert m.total_weight == 3.0
    assert m.unigram_count(6) == 2.0
    # Context (START, 5) at order 3 was seen once by the second update.
    assert m.max_order_seen([5]) == 3
    # Context (9,) was never seen at order 2 or 3, so only the unigram applies.
    assert m.max_order_seen([9]) == 1
    # Seen bigram context (5,) -> 6 has count 2 out of 2.
    p2 = m.prob([5], 6, 2)
    assert p2 > m.prob([5], 7, 2)
    # Unseen bigram context backs off exactly to the unigram probability.
    unseen = [9]
    assert m.prob(unseen, 6, 2) == pytest.approx(m.prob(unseen, 6, 1))
    assert m.prob([9, 9], 6, 3) == pytest.approx(m.prob([9, 9], 6, 2))
    lp = m.log_probs([5], 6)
    assert len(lp) == 3
    assert lp[0] == pytest.approx(math.log(m.prob([5], 6, 1)))
    with pytest.raises(ValueError):
        m.prob([5], 6, 4)


def test_ngram_start_marker():
    m = NgramMemory(orders=(1, 2, 3))
    m.update([], 42)
    assert START_ID == -2
    # First response position: order-2 context (START,), order-3 context (START, START).
    assert m.prob([], 42, 2) > m.prob([], 7, 2)
    assert m.prob([], 42, 3) > m.prob([], 7, 3)
    assert m.top([], 1) == [(42, m.prob([], 42, m.max_order_seen([])))]
    assert m.max_order_seen([]) == 3


def test_ngram_top_and_draft_and_empty():
    empty = NgramMemory()
    assert empty.max_order_seen([1, 2]) == 0
    assert empty.top([1, 2], 5) == []
    assert empty.draft([1, 2], 4) == []

    m = NgramMemory(orders=(1, 2, 3))
    for _ in range(3):
        m.update([], 1)
        m.update([1], 2)
        m.update([1, 2], 3)
    before = m.total_weight
    draft = m.draft([], max_len=3, min_prob=0.1)
    assert draft == [1, 2, 3]
    assert m.total_weight == before  # draft never mutates
    assert m.draft([], max_len=0) == []
    m.update([], 4, weight=0.0)  # zero weight adds nothing and does not raise
    assert m.unigram_count(4) == 0.0
