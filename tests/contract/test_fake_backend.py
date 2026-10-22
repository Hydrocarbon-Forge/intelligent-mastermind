"""FakeBackend: the in-process backend every driver test uses."""
from __future__ import annotations

import asyncio
import math

import pytest

from humanapi.backend.base import BackendUnavailable, Capabilities
from humanapi.backend.fake import FakeBackend


def model_of(table: dict[int, float]):
    return lambda context: dict(table)


def test_sorted_truncated_and_tail():
    backend = FakeBackend(model_of({1: 0.5, 2: 0.3, 3: 0.1, 4: 0.05}), Capabilities(max_top_k=3))
    result = asyncio.run(backend.next_token([9], 10))
    assert result.k == 3
    assert [t.token_id for t in result.top] == [1, 2, 3]
    assert result.tail_mass == pytest.approx(1.0 - (0.5 + 0.3 + 0.1))
    assert result.top[0].raw == b"<1>" and result.top[0].text == "<1>"
    assert backend.calls == [("next_token", 1)]


def test_filtered_renormalises_and_zero_tail():
    backend = FakeBackend(model_of({1: 0.4, 2: 0.2}), Capabilities(filtered_probs=True))
    result = asyncio.run(backend.next_token([], 5))
    assert result.tail_mass == 0.0
    assert sum(math.exp(t.logprob) for t in result.top) == pytest.approx(1.0)


def test_presence_penalty_applies_to_context_before_sort():
    seen = []

    def model(context):
        seen.append(list(context))
        return {1: 0.5, 2: 0.4}

    backend = FakeBackend(model, presence_penalty=math.log(2.0))
    result = asyncio.run(backend.next_token([1], 5))
    # Token 1 is in the context, so its 0.5 becomes 0.25 and token 2 ranks first; no renormalising.
    assert [t.token_id for t in result.top] == [2, 1]
    assert result.top[1].logprob == pytest.approx(math.log(0.25))
    assert result.tail_mass == pytest.approx(1.0 - 0.65)


def test_missing_ids_are_omitted():
    backend = FakeBackend(model_of({1: 0.5, 2: 0.3, 3: 0.1}), missing_ids=[1])
    result = asyncio.run(backend.next_token([], 5))
    assert [t.token_id for t in result.top] == [2, 3]


def test_fail_with_raises_every_call_and_health_is_false():
    backend = FakeBackend(model_of({1: 0.5}))
    backend.fail_with = BackendUnavailable("down")
    assert asyncio.run(backend.health()) is False
    with pytest.raises(BackendUnavailable):
        asyncio.run(backend.next_token([], 1))
    with pytest.raises(BackendUnavailable):
        asyncio.run(backend.score_continuation([], [1, 2], 1))
    assert backend.calls[-1] == ("score_continuation", 2)


def test_score_fast_path_records_one_call_and_contexts():
    contexts = []

    def model(context):
        contexts.append(list(context))
        return {7: 0.6, 8: 0.3}

    backend = FakeBackend(model, Capabilities(prompt_logprobs=True))
    results = asyncio.run(backend.score_continuation([1, 2], [7, 8, 7], 5))
    assert len(results) == 3
    assert backend.calls == [("score_continuation", 3)]
    assert contexts == [[1, 2], [1, 2, 7], [1, 2, 7, 8]]


def test_score_slow_path_records_one_call_per_position():
    backend = FakeBackend(model_of({7: 0.6}), Capabilities(prompt_logprobs=False))
    results = asyncio.run(backend.score_continuation([1], [7, 7], 5))
    assert len(results) == 2
    assert backend.calls == [("next_token", 1), ("next_token", 1)]
