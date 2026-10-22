"""In-process FakeBackend: the InferenceBackend used by every driver test. No network, no server."""
from __future__ import annotations

import math
from typing import Callable, Collection, Mapping, Optional, Sequence

from humanapi.backend.base import Capabilities, NextTokenResult, TokenLP


class FakeBackend:
    capabilities: Capabilities
    calls: list[tuple[str, int]]
    fail_with: Optional[Exception]

    def __init__(
        self,
        model: Callable[[Sequence[int]], Mapping[int, float]],
        capabilities: Capabilities | None = None,
        *,
        presence_penalty: float = 0.0,
        token_bytes: Callable[[int], bytes] | None = None,
        missing_ids: Collection[int] = (),
    ) -> None:
        self.model = model
        self.capabilities = capabilities if capabilities is not None else Capabilities()
        self.presence_penalty = presence_penalty
        self._token_bytes = token_bytes if token_bytes is not None else (lambda i: f"<{i}>".encode("utf-8"))
        self.missing_ids = frozenset(missing_ids)
        self.calls = []
        self.fail_with = None

    def _check_failure(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    def _distribution(self, context: Sequence[int], k: int) -> NextTokenResult:
        k_eff = max(0, min(k, self.capabilities.max_top_k))
        probs = {int(token_id): float(p) for token_id, p in self.model(list(context)).items()}
        for token_id in self.missing_ids:
            probs.pop(token_id, None)
        if self.presence_penalty:
            factor = math.exp(-self.presence_penalty)
            seen = set(context)
            probs = {token_id: (p * factor if token_id in seen else p) for token_id, p in probs.items()}
        ranked = sorted(
            ((token_id, p) for token_id, p in probs.items() if p > 0),
            key=lambda item: (-item[1], item[0]),
        )[:k_eff]
        mass = sum(p for _, p in ranked)
        if self.capabilities.filtered_probs:
            tail = 0.0
            if mass > 0:
                ranked = [(token_id, p / mass) for token_id, p in ranked]
        else:
            tail = max(0.0, 1.0 - mass)
        top = []
        for token_id, p in ranked:
            raw = self._token_bytes(token_id)
            top.append(
                TokenLP(
                    token_id=token_id,
                    text=raw.decode("utf-8", errors="replace"),
                    raw=raw,
                    logprob=math.log(p),
                )
            )
        return NextTokenResult(top=top, tail_mass=tail, k=k_eff)

    async def health(self) -> bool:
        return self.fail_with is None

    async def next_token(self, prompt_ids: list[int], k: int) -> NextTokenResult:
        self.calls.append(("next_token", 1))
        self._check_failure()
        return self._distribution(prompt_ids, k)

    async def score_continuation(
        self, prompt_ids: list[int], cont_ids: list[int], k: int
    ) -> list[NextTokenResult]:
        count = len(cont_ids)
        if self.capabilities.prompt_logprobs:
            self.calls.append(("score_continuation", count))
        else:
            # The fallback costs one single-token call per position; record each so tests see it.
            self.calls.extend(("next_token", 1) for _ in range(count))
        self._check_failure()
        return [self._distribution(list(prompt_ids) + list(cont_ids[:j]), k) for j in range(count)]
