"""Candidate assembly: the tokens that compete for a position, merged from several sources.

Order is stable: backend top-k first (eos ids folded into END), then n-gram tops, kNN choices,
prefix matches, and END last. The first source that lists an id fixes its slot.
"""
from __future__ import annotations

import math
from typing import Optional

from humanapi.config import LearningConfig
from humanapi.learn.knn import KnnResult
from humanapi.learn.ngram import NgramMemory
from humanapi.learn.types import (
    SRC_BACKEND,
    SRC_END,
    SRC_KNN,
    SRC_NGRAM,
    SRC_PREFIX,
    Candidate,
    PositionContext,
)
from humanapi.text.types import TokenClass, Tokenizer

DEFAULT_K = 20


def _token_class(tok: Tokenizer, token_id: int) -> Optional[TokenClass]:
    try:
        return tok.token_class(token_id)
    except KeyError:  # padding id: no string at all
        return None


def backend_entries(ctx: PositionContext, tok: Tokenizer) -> dict[int, tuple[float, int]]:
    """Backend list as token id -> (probability, 1-based rank), in backend order.
    Every eos id is folded into one END entry (probabilities summed, rank of the first eos)."""
    out: dict[int, tuple[float, int]] = {}
    if ctx.result is None:
        return out
    for rank, entry in enumerate(ctx.result.top, start=1):
        if entry.token_id is None:
            continue
        p = math.exp(entry.logprob)
        if entry.token_id in tok.eos_ids:
            end = tok.end_id
            if end in out:
                out[end] = (out[end][0] + p, out[end][1])
            else:
                out[end] = (p, rank)
        elif entry.token_id not in out:
            out[entry.token_id] = (p, rank)
    return out


def base_bound(entries: dict[int, tuple[float, int]], cfg: LearningConfig) -> float:
    """Upper bound for unlisted tokens: the lowest returned probability, or tail_floor when nothing was returned."""
    if not entries:
        return cfg.tail_floor
    return min(p for p, _ in entries.values())


def base_k(ctx: PositionContext) -> int:
    return ctx.result.k if ctx.result is not None else DEFAULT_K


def unlisted_candidate(
    ctx: PositionContext,
    *,
    tok: Tokenizer,
    cfg: LearningConfig,
    token_id: int,
    sources: frozenset[str] = frozenset(),
) -> Candidate:
    """A candidate for one token id, using the backend's numbers when listed, the bound otherwise."""
    entries = backend_entries(ctx, tok)
    if token_id in entries:
        p, rank = entries[token_id]
        unknown = False
    else:
        p, rank = base_bound(entries, cfg), base_k(ctx) + 1
        unknown = True
    text = tok.token_text(token_id)
    return Candidate(
        token_id=token_id,
        text=text or "",
        p_base=p,
        base_unknown=unknown,
        base_rank=rank,
        token_class=tok.token_class(token_id),
        sources=frozenset(sources),
    )


def assemble(
    ctx: PositionContext,
    *,
    tok: Tokenizer,
    ngram: NgramMemory,
    knn: Optional[KnnResult],
    cfg: LearningConfig,
    max_ngram: int = 8,
) -> list[Candidate]:
    entries = backend_entries(ctx, tok)
    bound = base_bound(entries, cfg)
    k = base_k(ctx)

    # token id -> {"p", "rank", "src"}; dict insertion order is the candidate order.
    found: dict[int, dict] = {}
    for token_id, (p, rank) in entries.items():
        found[token_id] = {"p": p, "rank": rank, "src": {SRC_BACKEND}}

    def add(token_id: int, source: str) -> None:
        slot = found.get(token_id)
        if slot is None:
            found[token_id] = {"p": bound, "rank": k + 1, "src": {source}}
        else:
            slot["src"].add(source)

    for token_id, _ in ngram.top(ctx.response_ids, max_ngram):
        add(token_id, SRC_NGRAM)
    if knn is not None:
        for token_id in knn.votes:
            add(token_id, SRC_KNN)
    if ctx.typed_prefix:
        for token_id in tok.tokens_with_prefix(ctx.typed_prefix, True):
            add(token_id, SRC_PREFIX)
        for token_id in tok.tokens_with_prefix(ctx.typed_prefix, False):
            add(token_id, SRC_PREFIX)
    add(tok.end_id, SRC_END)

    out: list[Candidate] = []
    for token_id, slot in found.items():
        cls = _token_class(tok, token_id)
        if cls is None:
            continue
        text = tok.token_text(token_id)
        if text is None and cls not in (TokenClass.END, TokenClass.FRAGMENT):
            continue
        out.append(
            Candidate(
                token_id=token_id,
                text=text or "",
                p_base=float(slot["p"]),
                base_unknown=token_id not in entries,
                base_rank=int(slot["rank"]),
                token_class=cls,
                sources=frozenset(slot["src"]),
            )
        )
    return out
