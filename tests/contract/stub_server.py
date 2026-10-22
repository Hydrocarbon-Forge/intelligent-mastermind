"""In-process stub of an OpenAI-compatible completions server with configurable quirks.

The contract tests run OpenAICompatBackend (and tools/probe_backend.py) against make_stub_app through
httpx.ASGITransport, so no socket is opened. The app's `requests` attribute records every request body.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping, Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Token id -> bytes for the stub's own vocabulary. Every entry is distinct and valid UTF-8.
VOCAB: tuple[bytes, ...] = (
    b" ", b".", b",",
    *(bytes([c]) for c in range(ord("a"), ord("z") + 1)),
    b" the", b" cat", b" sat", b" on", b" mat", b" ok", b" yeah", b"ing", b"ed", b"un",
    b"believ", b"able", b" I", b"I", b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9",
    b"!", b"?", b"'", b"\n", b" to", b" and", b" it", b" is", b" in", b"ly", b" do",
)
VOCAB_SIZE = len(VOCAB)
SERVER_FILTER_TOP_K = 20


def default_token_bytes(token_id: int) -> bytes:
    if 0 <= token_id < VOCAB_SIZE:
        return VOCAB[token_id]
    return f"<pad{token_id}>".encode("utf-8")


def default_model(context: Sequence[int]) -> dict[int, float]:
    """Deterministic distribution over the stub vocabulary; not normalised to a filtered set."""
    length = len(context)
    total_context = sum(context)
    weights = {i: 1.0 + float((i * 7 + length * 3 + total_context) % 11) for i in range(VOCAB_SIZE)}
    total = sum(weights.values())
    return {i: w / total for i, w in weights.items()}


def _greedy_ids(text: str, table: Mapping[bytes, int]) -> list[int]:
    data = text.encode("utf-8")
    longest = max(len(key) for key in table)
    out: list[int] = []
    position = 0
    while position < len(data):
        for length in range(min(longest, len(data) - position), 0, -1):
            token_id = table.get(data[position : position + length])
            if token_id is not None:
                out.append(token_id)
                position += length
                break
        else:
            position += 1
    return out


def _ranked(
    context: Sequence[int], model: Callable[[Sequence[int]], Mapping[int, float]], filtered: bool
) -> list[tuple[int, float]]:
    probs = {int(i): float(p) for i, p in model(list(context)).items()}
    ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    if filtered:
        ranked = ranked[:SERVER_FILTER_TOP_K]
        total = sum(p for _, p in ranked) or 1.0
        ranked = [(i, p / total) for i, p in ranked]
    return [(i, p) for i, p in ranked if p > 0]


def _content_item(token_id: int, probability: float, return_ids: bool, token_bytes: Callable[[int], bytes]) -> dict:
    raw = token_bytes(token_id)
    item: dict[str, Any] = {
        "token": raw.decode("utf-8", errors="replace"),
        "logprob": math.log(probability),
        "bytes": list(raw),
    }
    if return_ids:
        item["id"] = token_id
    return item


def _completion_body(
    ranked: list[tuple[int, float]], response_format: str, return_ids: bool, token_bytes: Callable[[int], bytes]
) -> dict:
    if response_format == "content":
        tops = [_content_item(i, p, return_ids, token_bytes) for i, p in ranked]
        first = tops[0] if tops else {"token": "", "logprob": 0.0, "bytes": []}
        logprobs: dict[str, Any] = {
            "content": [
                {
                    "token": first["token"],
                    "logprob": first["logprob"],
                    "bytes": first["bytes"],
                    "top_logprobs": tops,
                }
            ]
        }
    else:
        distribution = {
            token_bytes(i).decode("utf-8", errors="replace"): math.log(p) for i, p in ranked
        }
        logprobs = {"top_logprobs": [distribution]}
    return {
        "id": "cmpl-stub",
        "object": "text_completion",
        "choices": [{"index": 0, "text": "", "logprobs": logprobs, "finish_reason": "length"}],
    }


def _echo_body(
    context: Sequence[int],
    k: int,
    model: Callable[[Sequence[int]], Mapping[int, float]],
    filtered: bool,
    token_bytes: Callable[[int], bytes],
) -> dict:
    """Legacy shape: one entry per prompt token; the first has no context and is null."""
    tops: list[dict | None] = [None]
    for position in range(1, len(context)):
        ranked = _ranked(context[:position], model, filtered)[:k]
        tops.append({token_bytes(i).decode("utf-8", errors="replace"): math.log(p) for i, p in ranked})
    return {
        "id": "cmpl-stub",
        "object": "text_completion",
        "choices": [{"index": 0, "text": "", "logprobs": {"top_logprobs": tops}, "finish_reason": "length"}],
    }


def make_stub_app(
    *,
    response_format: str = "content",
    return_ids: bool = True,
    filtered: bool = False,
    top_k_cap: int = 20,
    support_echo: bool = True,
    accept_id_prompts: bool = True,
    model: Callable[[Sequence[int]], Mapping[int, float]] | None = None,
    token_bytes: Callable[[int], bytes] | None = None,
) -> FastAPI:
    if response_format not in ("content", "legacy"):
        raise ValueError(f"unknown response_format {response_format!r}")
    token_fn = token_bytes if token_bytes is not None else default_token_bytes
    model_fn = model if model is not None else default_model
    table: dict[bytes, int] = {}
    for token_id in range(VOCAB_SIZE):
        raw = token_fn(token_id)
        if raw:
            table.setdefault(raw, token_id)

    app = FastAPI()
    app.requests = []  # type: ignore[attr-defined]

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": "stub", "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        app.requests.append(copy.deepcopy(body))  # type: ignore[attr-defined]
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            if not accept_id_prompts:
                return JSONResponse({"error": "token id prompts are not accepted"}, status_code=400)
            context = [int(x) for x in prompt]
        elif isinstance(prompt, str):
            context = _greedy_ids(prompt, table)
        else:
            return JSONResponse({"error": "prompt must be a string or an id list"}, status_code=400)
        k = min(int(body.get("logprobs") or 0), top_k_cap)
        if body.get("echo"):
            if not support_echo:
                return JSONResponse({"error": "echo is not supported"}, status_code=400)
            return JSONResponse(_echo_body(context, k, model_fn, filtered, token_fn))
        ranked = _ranked(context, model_fn, filtered)[:k]
        return JSONResponse(_completion_body(ranked, response_format, return_ids, token_fn))

    return app
