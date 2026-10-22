"""Capability probe: measures what the inference server supports and writes the capabilities JSON.

Usage: python -m tools.probe_backend [--config config.yaml]

The probe sends raw requests but never a sampler field, and it reads responses through the adapter's
helpers in humanapi/backend/openai_compat.py, the only module that knows the server's JSON shape.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import httpx

from humanapi.backend.base import BackendUnavailable, Capabilities
from humanapi.backend.openai_compat import (
    RawEntry,
    echo_length,
    position_entries,
    post_completion,
    save_capabilities,
)

if TYPE_CHECKING:
    from humanapi.text.types import Tokenizer

PROBE_TIMEOUT_S = 60.0
_CAP_CHECK_KS = (5, 20, 50)
_SERVER_TOP_K = 20
_SAMPLE_TEXT = "i believe the cat sat on the mat, and yeah it is ok. "
_PREFIX_TOKENS = 1000
_LATENCY_TOKENS = (500, 4000)
_FILTER_TOLERANCE = 0.02
_CACHE_SPEEDUP = 0.30


def _repeat_to(ids: Sequence[int], length: int) -> list[int]:
    if not ids:
        raise ValueError("cannot build a probe prompt from an empty sample")
    repeats = length // len(ids) + 1
    return (list(ids) * repeats)[:length]


def _entries_or_none(data: Optional[dict[str, Any]]) -> Optional[list[RawEntry]]:
    if data is None:
        return None
    try:
        return position_entries(data, 0)
    except BackendUnavailable:
        return None


async def _probe(
    http: httpx.AsyncClient,
    base_url: str,
    model: str,
    tokenizer: "Tokenizer",
    timer: Callable[[], float],
) -> tuple[Capabilities, dict]:
    async def send(prompt: list[int] | str, max_tokens: int, k: int, **extra: Any) -> Optional[dict]:
        body: dict[str, Any] = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "logprobs": k}
        body.update(extra)
        try:
            return await post_completion(http, base_url, body, PROBE_TIMEOUT_S)
        except BackendUnavailable:
            return None

    base_ids = list(tokenizer.encode(_SAMPLE_TEXT))
    base_text = tokenizer.decode(base_ids)

    # Prompt as ids: the same prompt as ids and as text must give the same top token.
    ids_resp = await send(base_ids, 1, 5)
    text_resp = await send(base_text, 1, 5)
    ids_entries = _entries_or_none(ids_resp)
    text_entries = _entries_or_none(text_resp)
    prompt_as_ids = bool(
        ids_entries
        and text_entries
        and ids_entries[0].raw == text_entries[0].raw
    )

    def payload(ids: Sequence[int]) -> list[int] | str:
        return list(ids) if prompt_as_ids else tokenizer.decode(list(ids))

    # Top-logprob cap: the largest k that returned at least k entries, never above the server's top_k of 20.
    candidates: list[int] = []
    for k in _CAP_CHECK_KS:
        data = await send(payload(base_ids), 1, k)
        entries = _entries_or_none(data)
        if entries is not None and len(entries) >= k:
            candidates.append(k)
    max_top_k = min(_SERVER_TOP_K, max(candidates, default=1))

    # Ids in the response: every entry of a one-token response carries an id.
    sample_resp = ids_resp if prompt_as_ids else text_resp
    sample_entries = _entries_or_none(sample_resp) or []
    returns_ids = bool(sample_entries) and all(entry.token_id is not None for entry in sample_entries)

    # Filtered or raw: the top 20 entries sum to about 1 when the server already filtered them.
    wide = _entries_or_none(await send(payload(base_ids), 1, _SERVER_TOP_K)) or []
    total = sum(math.exp(entry.logprob) for entry in wide)
    filtered_probs = bool(wide) and len(wide) <= _SERVER_TOP_K and abs(total - 1.0) <= _FILTER_TOLERANCE

    # Prompt logprobs: an echo request with zero new tokens returns one entry per prompt token.
    echo_resp = await send(payload(base_ids), 0, 5, echo=True)
    prompt_logprobs = echo_resp is not None and echo_length(echo_resp) >= 2

    # Prefix cache: a second request sharing a long prefix is noticeably faster.
    prefix = _repeat_to(base_ids, _PREFIX_TOKENS)
    other_ids = [base_ids[-1], base_ids[0]]
    t0 = timer()
    await send(payload(prefix + other_ids[:1]), 1, 5)
    t1 = timer()
    await send(payload(prefix + other_ids[1:]), 1, 5)
    t2 = timer()
    first, second = t1 - t0, t2 - t1
    prefix_cache = first > 0 and second <= first * (1.0 - _CACHE_SPEEDUP)

    # Determinism: an identical request repeated gives the same logprobs.
    again = _entries_or_none(await send(payload(base_ids), 1, 5))
    determinism_tolerance: Optional[float] = None
    if sample_entries and again:
        first_by_raw = {entry.raw: entry.logprob for entry in sample_entries}
        diffs = [
            abs(first_by_raw[entry.raw] - entry.logprob)
            for entry in again
            if entry.raw in first_by_raw
        ]
        determinism_tolerance = max(diffs) if diffs else None

    # Latency: a one-token request at about 500 and about 4000 prompt tokens.
    latency_ms: dict[str, Optional[float]] = {}
    for target in _LATENCY_TOKENS:
        ids = _repeat_to(base_ids, target)
        start = timer()
        ok = await send(payload(ids), 1, 5)
        end = timer()
        latency_ms[str(target)] = (end - start) * 1000.0 if ok is not None else None

    caps = Capabilities(
        max_top_k=max_top_k,
        prompt_as_ids=prompt_as_ids,
        returns_ids=returns_ids,
        filtered_probs=filtered_probs,
        prompt_logprobs=prompt_logprobs,
        prefix_cache=prefix_cache,
    )
    extra = {"determinism_tolerance": determinism_tolerance, "latency_ms": latency_ms}
    return caps, extra


async def run_probe(
    base_url: str,
    model: str,
    tokenizer: "Tokenizer",
    *,
    client: httpx.AsyncClient | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[Capabilities, dict]:
    owns_client = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=PROBE_TIMEOUT_S)
    try:
        return await _probe(http, base_url, model, tokenizer, timer)
    finally:
        if owns_client:
            await http.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the inference server and write its capabilities.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    from humanapi.config import load_config
    from humanapi.text.tokenizer import HumanTokenizer

    config = load_config(args.config)
    tokenizer = HumanTokenizer(config.tokenizer.path, enable_thinking=config.tokenizer.enable_thinking)

    async def run() -> tuple[Capabilities, dict]:
        async with httpx.AsyncClient(timeout=config.backend.timeout_s) as client:
            return await run_probe(config.backend.base_url, config.backend.model, tokenizer, client=client)

    caps, extra = asyncio.run(run())
    save_capabilities(config.backend.capabilities_file, caps, extra)
    print(
        f"max_top_k={caps.max_top_k} prompt_as_ids={caps.prompt_as_ids} returns_ids={caps.returns_ids} "
        f"filtered_probs={caps.filtered_probs} prompt_logprobs={caps.prompt_logprobs} "
        f"prefix_cache={caps.prefix_cache} -> {config.backend.capabilities_file}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
