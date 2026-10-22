"""Contract tests: OpenAICompatBackend and the capability probe, run against the in-process stub server."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Callable, Sequence

import httpx
import pytest

from humanapi.backend.base import BackendUnavailable, Capabilities
from humanapi.backend.openai_compat import OpenAICompatBackend, load_capabilities, save_capabilities
from tests.contract.stub_server import VOCAB, default_model, default_token_bytes, make_stub_app
from tools.probe_backend import run_probe

FORBIDDEN_SAMPLER_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
}
BASE_URL = "http://stub/v1"


class FakeTokenizer:
    """Greedy longest-match byte tokenizer over a table of byte strings (same algorithm as the stub)."""

    def __init__(self, table: Sequence[bytes]) -> None:
        self.table = list(table)
        self._by_bytes: dict[bytes, int] = {}
        for token_id, raw in enumerate(self.table):
            self._by_bytes.setdefault(raw, token_id)
        self._longest = max(len(raw) for raw in self.table)

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        out: list[int] = []
        position = 0
        while position < len(data):
            for length in range(min(self._longest, len(data) - position), 0, -1):
                token_id = self._by_bytes.get(data[position : position + length])
                if token_id is not None:
                    out.append(token_id)
                    position += length
                    break
            else:
                raise ValueError(f"no token for byte {data[position]!r}")
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return b"".join(self.table[i] for i in ids).decode("utf-8", errors="replace")

    def bytes_to_ids(self, raw: bytes) -> list[int]:
        return [i for i, entry in enumerate(self.table) if entry == raw]


class FlakyTransport(httpx.AsyncBaseTransport):
    """Raises the queued errors on the first requests, then delegates to the inner transport."""

    def __init__(self, inner: httpx.AsyncBaseTransport, errors: list[Exception]) -> None:
        self.inner = inner
        self.errors = list(errors)
        self.attempts = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return await self.inner.handle_async_request(request)


def make_client(app, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=transport if transport is not None else httpx.ASGITransport(app=app),
        base_url="http://stub",
    )


def make_backend(app, caps: Capabilities | None = None, tokenizer=None, transport=None) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        BASE_URL,
        "stub-model",
        tokenizer if tokenizer is not None else FakeTokenizer(VOCAB),
        caps,
        timeout_s=5.0,
        client=make_client(app, transport),
    )


def ids_of(result) -> list[int]:
    return [token.token_id for token in result.top]


PROMPT = list(FakeTokenizer(VOCAB).encode("the cat sat on the mat"))


def test_next_token_sorted_and_tail():
    app = make_stub_app()

    async def run():
        backend = make_backend(app)
        result = await backend.next_token(PROMPT, 10)
        await backend.aclose()
        return result

    result = asyncio.run(run())
    logprobs = [token.logprob for token in result.top]
    assert logprobs == sorted(logprobs, reverse=True)
    assert len(result.top) == 10 and result.k == 10
    expected_tail = max(0.0, 1.0 - sum(math.exp(lp) for lp in logprobs))
    assert result.tail_mass == pytest.approx(expected_tail, abs=1e-9)
    assert result.tail_mass > 0.0


def test_k_clamped_to_cap():
    app = make_stub_app()

    async def run(caps: Capabilities, k: int):
        backend = make_backend(app, caps)
        result = await backend.next_token(PROMPT, k)
        await backend.aclose()
        return result

    limited = asyncio.run(run(Capabilities(max_top_k=5), 50))
    assert limited.k == 5 and len(limited.top) == 5
    hard = asyncio.run(run(Capabilities(max_top_k=50), 50))
    assert hard.k == 20 and len(hard.top) == 20
    sent = [body["logprobs"] for body in app.requests]
    assert sent[-1] == 20


def test_unresolvable_dropped():
    # id 999 has bytes that are no entry of the table; with returns_ids false it cannot be resolved.
    app = make_stub_app(
        return_ids=False,
        model=lambda ctx: {**default_model(ctx), 999: 0.9},
    )

    async def run():
        backend = make_backend(app, Capabilities(returns_ids=False))
        result = await backend.next_token(PROMPT, 20)
        await backend.aclose()
        return result

    result = asyncio.run(run())
    assert 999 not in ids_of(result)
    assert all(token.token_id is not None and token.token_id < len(VOCAB) for token in result.top)


def test_bytes_to_id_fallback(caplog):
    caplog.set_level(logging.WARNING, logger="humanapi.backend.openai_compat")
    # Id 64 duplicates the bytes of id 5, so those bytes are ambiguous and both ids are dropped.
    table = (*VOCAB, VOCAB[5])
    tokenizer = FakeTokenizer(table)

    def token_bytes(token_id: int) -> bytes:
        return table[token_id] if token_id < len(table) else default_token_bytes(token_id)

    app = make_stub_app(
        return_ids=False,
        token_bytes=token_bytes,
        model=lambda ctx: {64: 0.5, 5: 0.2, 7: 0.1},
    )

    async def run():
        backend = make_backend(app, Capabilities(returns_ids=False), tokenizer=tokenizer)
        result = await backend.next_token(PROMPT, 10)
        await backend.aclose()
        return result

    result = asyncio.run(run())
    assert ids_of(result) == [7]
    assert any("resolve to 2 ids" in record.getMessage() for record in caplog.records)


def test_text_prompt_fallback():
    app = make_stub_app()
    tokenizer = FakeTokenizer(VOCAB)

    async def run():
        ids_backend = make_backend(app, Capabilities(prompt_as_ids=True))
        text_backend = make_backend(app, Capabilities(prompt_as_ids=False))
        by_ids = await ids_backend.next_token(PROMPT, 5)
        by_text = await text_backend.next_token(PROMPT, 5)
        # A non-canonical id sequence (" " then "the") must not be sent as text at all.
        before = len(app.requests)
        noncanonical = [VOCAB.index(b" "), VOCAB.index(b"t"), VOCAB.index(b"h"), VOCAB.index(b"e")]
        with pytest.raises(BackendUnavailable):
            await text_backend.next_token(noncanonical, 5)
        assert len(app.requests) == before
        await ids_backend.aclose()
        await text_backend.aclose()
        return by_ids, by_text

    by_ids, by_text = asyncio.run(run())
    assert app.requests[-2]["prompt"] == PROMPT  # the id request
    assert app.requests[-1]["prompt"] == tokenizer.decode(PROMPT)  # the text request
    assert ids_of(by_ids) == ids_of(by_text)


def test_no_sampler_fields():
    app = make_stub_app()

    async def run():
        backend = make_backend(app)
        await backend.next_token(PROMPT, 5)
        await backend.score_continuation(PROMPT, PROMPT[:3], 5)
        plain = make_backend(app, Capabilities(prompt_logprobs=False))
        await plain.score_continuation(PROMPT, PROMPT[:2], 5)
        await run_probe(BASE_URL, "stub-model", FakeTokenizer(VOCAB), client=make_client(app), timer=_ticker())
        await backend.aclose()
        await plain.aclose()

    asyncio.run(run())
    assert app.requests
    for body in app.requests:
        assert not (FORBIDDEN_SAMPLER_KEYS & set(body)), body


def test_filtered_probs_detected():
    async def probe(app):
        caps, _ = await run_probe(BASE_URL, "stub-model", FakeTokenizer(VOCAB), client=make_client(app), timer=_ticker())
        return caps

    filtered = asyncio.run(probe(make_stub_app(filtered=True)))
    assert filtered.filtered_probs is True

    raw_app = make_stub_app(filtered=False)
    raw = asyncio.run(probe(raw_app))
    assert raw.filtered_probs is False

    async def tail_with(caps: Capabilities):
        backend = make_backend(make_stub_app(filtered=True), caps)
        result = await backend.next_token(PROMPT, 20)
        await backend.aclose()
        return result

    assert asyncio.run(tail_with(filtered)).tail_mass == 0.0


def test_prompt_logprobs_fallback():
    app = make_stub_app()

    async def run(caps: Capabilities):
        backend = make_backend(app, caps)
        results = await backend.score_continuation(PROMPT, PROMPT[:4], 5)
        await backend.aclose()
        return results

    fast_start = len(app.requests)
    fast = asyncio.run(run(Capabilities(prompt_logprobs=True)))
    fast_bodies = app.requests[fast_start:]
    slow_start = len(app.requests)
    slow = asyncio.run(run(Capabilities(prompt_logprobs=False)))
    slow_bodies = app.requests[slow_start:]

    assert len(fast) == len(slow) == 4
    assert len(fast_bodies) == 1 and fast_bodies[0].get("echo") is True
    assert fast_bodies[0]["max_tokens"] == 0
    assert len(slow_bodies) == 4 and all("echo" not in body for body in slow_bodies)
    assert all(body["max_tokens"] == 1 for body in slow_bodies)
    for fast_result, slow_result in zip(fast, slow):
        assert ids_of(fast_result) == ids_of(slow_result)
        for a, b in zip(fast_result.top, slow_result.top):
            assert a.logprob == pytest.approx(b.logprob)


def test_retry_once_on_connection_error():
    app = make_stub_app()
    transport = FlakyTransport(httpx.ASGITransport(app=app), [httpx.ConnectError("refused")])

    async def run():
        backend = make_backend(app, transport=transport)
        result = await backend.next_token(PROMPT, 3)
        await backend.aclose()
        return result

    result = asyncio.run(run())
    assert len(result.top) == 3
    assert transport.attempts == 2

    twice = FlakyTransport(
        httpx.ASGITransport(app=app),
        [httpx.ConnectError("refused"), httpx.ReadError("reset")],
    )

    async def run_twice():
        backend = make_backend(app, transport=twice)
        try:
            await backend.next_token(PROMPT, 3)
        finally:
            await backend.aclose()

    with pytest.raises(BackendUnavailable):
        asyncio.run(run_twice())
    assert twice.attempts == 2


def test_timeout_degrades():
    app = make_stub_app()
    transport = FlakyTransport(httpx.ASGITransport(app=app), [httpx.ReadTimeout("slow")])

    async def run():
        backend = make_backend(app, transport=transport)
        try:
            with pytest.raises(BackendUnavailable):
                await backend.next_token(PROMPT, 3)
            attempts_after_timeout = transport.attempts
            healthy = await backend.health()  # the queued timeout is spent, so the stub answers
            return attempts_after_timeout, healthy
        finally:
            await backend.aclose()

    attempts_after_timeout, healthy = asyncio.run(run())
    assert attempts_after_timeout == 1  # a timeout is not retried
    assert healthy is True


def test_health_false_when_unreachable():
    app = make_stub_app()
    transport = FlakyTransport(httpx.ASGITransport(app=app), [httpx.ConnectError("refused")] * 5)

    async def run():
        backend = make_backend(app, transport=transport)
        try:
            return await backend.health()
        finally:
            await backend.aclose()

    assert asyncio.run(run()) is False


def _ticker() -> Callable[[], float]:
    state = {"now": 0.0}

    def timer() -> float:
        state["now"] += 0.001
        return state["now"]

    return timer


PROBE_MODES = {
    "default": ({}, {"max_top_k": 20, "prompt_as_ids": True, "returns_ids": True,
                     "filtered_probs": False, "prompt_logprobs": True, "prefix_cache": False}),
    "legacy": ({"response_format": "legacy"}, {"returns_ids": False, "max_top_k": 20,
                                                "prompt_as_ids": True, "filtered_probs": False,
                                                "prompt_logprobs": True}),
    "filtered": ({"filtered": True}, {"filtered_probs": True, "max_top_k": 20}),
    "no_id_prompts": ({"accept_id_prompts": False}, {"prompt_as_ids": False, "max_top_k": 20}),
    "no_echo": ({"support_echo": False}, {"prompt_logprobs": False, "max_top_k": 20}),
    "cap10": ({"top_k_cap": 10}, {"max_top_k": 5, "prompt_logprobs": True}),
}


@pytest.mark.parametrize("mode", sorted(PROBE_MODES))
def test_probe_writes_capabilities(mode, tmp_path):
    kwargs, expected = PROBE_MODES[mode]
    app = make_stub_app(**kwargs)

    async def run():
        return await run_probe(BASE_URL, "stub-model", FakeTokenizer(VOCAB), client=make_client(app), timer=_ticker())

    caps, extra = asyncio.run(run())
    path = tmp_path / "capabilities.json"
    save_capabilities(str(path), caps, extra)
    written = load_capabilities(str(path))
    assert written == caps

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        assert data[key] == value, (mode, key, data)
    assert set(data) >= {"determinism_tolerance", "latency_ms", "max_top_k"}
    assert data["determinism_tolerance"] == 0.0
    assert set(data["latency_ms"]) == {"500", "4000"}
    assert data["prefix_cache"] is False
