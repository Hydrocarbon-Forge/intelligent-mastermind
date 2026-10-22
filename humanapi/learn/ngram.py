"""Interpolated n-gram memory over the person's own labeled token choices.

Order k means the context is the last k-1 response tokens (order 1 is the unigram).
Pure Python, deterministic, no I/O.
"""
from __future__ import annotations

import math
from typing import Sequence

START_ID: int = -2  # context padding token used before the first response token

Context = tuple[int, ...]


class NgramMemory:
    def __init__(
        self,
        orders: Sequence[int] = (1, 2, 3),
        alpha: float = 0.1,
        vocab_size: int = 150000,
        backoff_strength: float = 1.0,
    ) -> None:
        self.orders = list(orders)
        if not self.orders or any(k < 1 for k in self.orders):
            raise ValueError("orders must be a non-empty sequence of integers >= 1")
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.backoff_strength = backoff_strength
        self.total_weight: float = 0.0
        # Counts are kept for every order 1..max(orders) so that the backoff chain
        # P_n -> P_{n-1} -> ... -> P_1 is always available.
        self._max_order = max(self.orders)
        self._counts: dict[tuple[int, Context], dict[int, float]] = {}
        self._totals: dict[tuple[int, Context], float] = {}

    @staticmethod
    def _context(context_ids: Sequence[int], order: int) -> Context:
        m = order - 1
        if m == 0:
            return ()
        tail = tuple(context_ids[-m:])
        return (START_ID,) * (m - len(tail)) + tail

    def update(self, context_ids: Sequence[int], token_id: int, weight: float = 1.0) -> None:
        if weight == 0:
            return
        for k in range(1, self._max_order + 1):
            key = (k, self._context(context_ids, k))
            bucket = self._counts.setdefault(key, {})
            bucket[token_id] = bucket.get(token_id, 0.0) + weight
            self._totals[key] = self._totals.get(key, 0.0) + weight
        self.total_weight += weight

    def _prob(self, context_ids: Sequence[int], token_id: int, order: int) -> float:
        if order == 1:
            denom = self.total_weight + self.alpha * self.vocab_size
            if denom <= 0:
                return 0.0
            return (self.unigram_count(token_id) + self.alpha) / denom
        lower = self._prob(context_ids, token_id, order - 1)
        key = (order, self._context(context_ids, order))
        total = self._totals.get(key, 0.0)
        if total <= 0:
            return lower
        count = self._counts[key].get(token_id, 0.0)
        b = self.backoff_strength
        return (count + b * lower) / (total + b)

    def prob(self, context_ids: Sequence[int], token_id: int, order: int) -> float:
        if order not in self.orders:
            raise ValueError(f"order {order} is not configured (orders={self.orders})")
        return self._prob(context_ids, token_id, order)

    def log_probs(self, context_ids: Sequence[int], token_id: int) -> list[float]:
        return [math.log(self._prob(context_ids, token_id, k)) for k in self.orders]

    def max_order_seen(self, context_ids: Sequence[int]) -> int:
        best = 0
        for k in self.orders:
            if self._totals.get((k, self._context(context_ids, k)), 0.0) > 0:
                best = max(best, k)
        return best

    def top(self, context_ids: Sequence[int], n: int) -> list[tuple[int, float]]:
        order = self.max_order_seen(context_ids)
        if order == 0 or n <= 0:
            return []
        bucket = self._counts[(order, self._context(context_ids, order))]
        scored = [(tok, self._prob(context_ids, tok, order)) for tok, c in bucket.items() if c > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:n]

    def draft(self, context_ids: Sequence[int], max_len: int, min_prob: float = 0.5) -> list[int]:
        """Greedy continuation. Never mutates the memory."""
        ctx = list(context_ids)
        drafted: list[int] = []
        while len(drafted) < max_len:
            best = self.top(ctx, 1)
            if not best:
                break
            token_id, p = best[0]
            if p < min_prob:
                break
            drafted.append(token_id)
            ctx.append(token_id)
        return drafted

    def unigram_count(self, token_id: int) -> float:
        return self._counts.get((1, ()), {}).get(token_id, 0.0)
