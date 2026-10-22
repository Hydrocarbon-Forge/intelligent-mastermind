"""OpenAI-compatible completions adapter (llama.cpp "content" shape and OpenAI "legacy" shape).

This is the only module that knows the server's JSON shape. Requests never carry sampler fields, and the
adapter never asks the server to generate: max_tokens is 1 for next_token and 0 for prompt-logprob requests.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence

import httpx

from humanapi.backend.base import BackendUnavailable, Capabilities, NextTokenResult, TokenLP

if TYPE_CHECKING:
    from humanapi.text.types import Tokenizer

logger = logging.getLogger(__name__)

HARD_TOP_K_CAP = 20
_RETRY_ERRORS = (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)
_CAPABILITY_FIELDS = frozenset(field.name for field in dataclasses.fields(Capabilities))


class RawEntry(NamedTuple):
    """One alternative as the server reported it, before id resolution."""

    token_id: Optional[int]
    text: str
    raw: bytes
    logprob: float


def clamp_k(k: int, caps: Capabilities) -> int:
    return max(0, min(k, caps.max_top_k, HARD_TOP_K_CAP))


async def post_completion(
    client: httpx.AsyncClient, base_url: str, body: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    """POST one completion request. A connection error is retried once; anything else raises BackendUnavailable."""
    url = f"{base_url.rstrip('/')}/completions"
    failure: Optional[BaseException] = None
    for _attempt in range(2):
        try:
            response = await client.post(url, json=body, timeout=timeout_s)
        except _RETRY_ERRORS as exc:
            failure = exc
            continue
        except httpx.HTTPError as exc:  # timeouts and every other transport failure: no retry
            raise BackendUnavailable(f"request failed: {exc!r}") from exc
        if not 200 <= response.status_code < 300:
            raise BackendUnavailable(f"server answered HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise BackendUnavailable("unparsable response body") from exc
        if not isinstance(data, dict):
            raise BackendUnavailable("unparsable response body")
        return data
    raise BackendUnavailable(f"connection failed twice: {failure!r}") from failure


def _entry_from_content(item: dict[str, Any]) -> RawEntry:
    text = str(item["token"])
    raw_list = item.get("bytes")
    raw = bytes(raw_list) if raw_list is not None else text.encode("utf-8")
    token_id = item.get("id")
    return RawEntry(
        None if token_id is None else int(token_id),
        text,
        raw,
        float(item["logprob"]),
    )


def position_entries(data: dict[str, Any], position: int) -> list[RawEntry]:
    """Alternatives reported for one position, in either response shape.

    Content shape: choices[0].logprobs.content[position].top_logprobs (entries may carry id and bytes).
    Legacy shape: choices[0].logprobs.top_logprobs[position], a dict from token string to logprob.
    A null legacy entry (the first prompt token of an echo) is an empty distribution.
    """
    try:
        logprobs = data["choices"][0]["logprobs"]
        if logprobs.get("content") is not None:
            tops = logprobs["content"][position]["top_logprobs"]
            return [_entry_from_content(item) for item in tops]
        distribution = logprobs["top_logprobs"][position]
        if distribution is None:
            return []
        return [
            RawEntry(None, text, text.encode("utf-8"), float(logprob))
            for text, logprob in distribution.items()
        ]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise BackendUnavailable("unexpected completion shape") from exc


def echo_length(data: dict[str, Any]) -> int:
    """How many per-prompt-token entries an echo response carries (0 when absent)."""
    try:
        tops = data["choices"][0]["logprobs"]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return 0
    return len(tops) if isinstance(tops, list) else 0


def resolve_entries(
    entries: Sequence[RawEntry], caps: Capabilities, tokenizer: "Tokenizer"
) -> list[TokenLP]:
    """Ids come from the server when it returns them, else from the bytes-to-id table; unresolvable entries drop."""
    resolved: list[TokenLP] = []
    for entry in entries:
        if caps.returns_ids and entry.token_id is not None:
            token_id = entry.token_id
        else:
            matches = tokenizer.bytes_to_ids(entry.raw)
            if len(matches) != 1:
                logger.warning(
                    "dropping token %r: its bytes resolve to %d ids", entry.text, len(matches)
                )
                continue
            token_id = matches[0]
        resolved.append(TokenLP(token_id=token_id, text=entry.text, raw=entry.raw, logprob=entry.logprob))
    return resolved


def build_result(resolved: Sequence[TokenLP], k_eff: int, caps: Capabilities) -> NextTokenResult:
    top = sorted(resolved, key=lambda token: token.logprob, reverse=True)[:k_eff]
    if caps.filtered_probs:
        tail = 0.0
    else:
        tail = max(0.0, 1.0 - sum(math.exp(token.logprob) for token in top))
    return NextTokenResult(top=top, tail_mass=tail, k=k_eff)


def prompt_payload(tokenizer: "Tokenizer", ids: Sequence[int], prompt_as_ids: bool) -> list[int] | str:
    """The prompt as the server should receive it. Text is sent only after a round-trip check, before any request."""
    ids = list(ids)
    if prompt_as_ids:
        return ids
    text = tokenizer.decode(ids)
    if list(tokenizer.encode(text)) != ids:
        raise BackendUnavailable("prompt does not round-trip through the tokenizer; nothing was sent")
    return text


class OpenAICompatBackend:
    capabilities: Capabilities

    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer: "Tokenizer",
        capabilities: Capabilities | None = None,
        *,
        timeout_s: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer
        self.capabilities = capabilities if capabilities is not None else Capabilities()
        self.timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    def _body(self, prompt: list[int] | str, max_tokens: int, logprobs: int, **extra: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
        }
        body.update(extra)
        return body

    async def health(self) -> bool:
        try:
            response = await self._http().get(f"{self.base_url}/models", timeout=self.timeout_s)
        except Exception:
            return False
        return response.status_code < 500

    async def _distribution(self, prompt_ids: Sequence[int], k: int) -> NextTokenResult:
        k_eff = clamp_k(k, self.capabilities)
        prompt = prompt_payload(self.tokenizer, prompt_ids, self.capabilities.prompt_as_ids)
        data = await post_completion(
            self._http(), self.base_url, self._body(prompt, 1, k_eff), self.timeout_s
        )
        entries = position_entries(data, 0)
        resolved = resolve_entries(entries, self.capabilities, self.tokenizer)
        return build_result(resolved, k_eff, self.capabilities)

    async def next_token(self, prompt_ids: list[int], k: int) -> NextTokenResult:
        return await self._distribution(prompt_ids, k)

    async def score_continuation(
        self, prompt_ids: list[int], cont_ids: list[int], k: int
    ) -> list[NextTokenResult]:
        cont = list(cont_ids)
        if not cont:
            return []
        if not self.capabilities.prompt_logprobs:
            return [await self._distribution(list(prompt_ids) + cont[:j], k) for j in range(len(cont))]

        k_eff = clamp_k(k, self.capabilities)
        full = list(prompt_ids) + cont
        prompt = prompt_payload(self.tokenizer, full, self.capabilities.prompt_as_ids)
        data = await post_completion(
            self._http(),
            self.base_url,
            self._body(prompt, 0, k_eff, echo=True),
            self.timeout_s,
        )
        if echo_length(data) != len(full):
            raise BackendUnavailable("prompt logprobs did not return one entry per prompt token")
        results: list[NextTokenResult] = []
        # Entry i describes the distribution that produced token i of the echoed sequence.
        for j in range(len(cont)):
            entries = position_entries(data, len(prompt_ids) + j)
            resolved = resolve_entries(entries, self.capabilities, self.tokenizer)
            results.append(build_result(resolved, k_eff, self.capabilities))
        return results

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def load_capabilities(path: str) -> Capabilities:
    """Capabilities from the probe's JSON file; defaults when the file is missing. Extra keys are ignored."""
    if not os.path.exists(path):
        return Capabilities()
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return Capabilities(**{key: value for key, value in data.items() if key in _CAPABILITY_FIELDS})


def save_capabilities(path: str, caps: Capabilities, extra: dict | None = None) -> None:
    payload = dataclasses.asdict(caps)
    if extra:
        payload.update({key: value for key, value in extra.items() if key not in payload})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
