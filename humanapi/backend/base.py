"""Backend contract: the only interface through which the program talks to the inference server (port 8081).

The adapter works in token ids and returns probabilities, never text. Implementations: OpenAICompatBackend
(humanapi/backend/openai_compat.py) and FakeBackend (humanapi/backend/fake.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class BackendUnavailable(Exception):
    """The backend timed out, refused, or failed twice. The driver degrades the position to a plain ask."""


@dataclass
class TokenLP:
    token_id: Optional[int]  # None only transiently, before the adapter resolves an id
    text: str
    raw: bytes  # token bytes, used to recover ids locally
    logprob: float


@dataclass
class NextTokenResult:
    top: list[TokenLP]  # sorted by logprob descending, at most k entries, every entry has a resolved token_id
    tail_mass: float  # 1 - sum(exp(lp)) over the returned tokens, clamped to >= 0
    k: int  # the effective k used (after clamping to Capabilities.max_top_k)


@dataclass
class Capabilities:
    max_top_k: int = 20  # never above the server's own top_k of 20
    prompt_as_ids: bool = True
    returns_ids: bool = True
    filtered_probs: bool = False  # probabilities already filtered by the server's top_p / top_k
    prompt_logprobs: bool = True
    prefix_cache: bool = True


class InferenceBackend(Protocol):
    capabilities: Capabilities

    async def health(self) -> bool:
        """True when the server answers. Never raises."""
        ...

    async def next_token(self, prompt_ids: list[int], k: int) -> NextTokenResult:
        """Probabilities for the token that follows prompt_ids. Raises BackendUnavailable on failure."""
        ...

    async def score_continuation(
        self, prompt_ids: list[int], cont_ids: list[int], k: int
    ) -> list[NextTokenResult]:
        """One result per continuation position: result[j] is the distribution over the token at
        position j of cont_ids, given prompt_ids + cont_ids[:j]. Raises BackendUnavailable on failure."""
        ...
