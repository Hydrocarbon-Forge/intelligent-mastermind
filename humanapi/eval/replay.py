"""Replay harness: runs the real candidate assembly, scorer, gate and learners over a corpus with a simulated
person who always chooses the true next token.

Learning runs over the first `train_fraction` of the corpus; the rest is frozen and used for evaluation only.
The harness never reimplements scoring or learning: it calls score_position, with_chosen, Gate and Learners.apply.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from humanapi.backend.base import BackendUnavailable, InferenceBackend, NextTokenResult, TokenLP
from humanapi.config import Config, load_config
from humanapi.eval import metrics
from humanapi.learn.embed import Embedder
from humanapi.learn.gate import Gate
from humanapi.learn.knn import KnnResult
from humanapi.learn.learners import Learners
from humanapi.learn.scorer import score_position, with_chosen
from humanapi.learn.types import (
    OTHER_ID,
    Auto,
    Decision,
    GateGroup,
    Outcome,
    PositionContext,
    PositionRecord,
    ScoredPosition,
)
from humanapi.text.typed import position_state
from humanapi.text.types import PosState, TokenClass, Tokenizer

_EMBED_TAIL_TOKENS = 128
_FAKE_FANOUT = 6
_FAKE_DECAY = 0.6
_FAKE_FIRST_PROB = 0.3


@dataclass
class CorpusItem:
    messages: list[dict]
    response: str


@dataclass
class ReplayConfig:
    seed: int = 0
    train_fraction: float = 0.8
    top_k: int = 20
    curve_window: int = 10  # responses per learning-curve point
    sweep_taus: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    cache_dir: str | None = None


@dataclass
class ReplayResult:
    n_train: int
    n_eval: int
    learning_curve: list[tuple[int, float]]  # (responses seen so far, auto-select rate over the last window)
    eval_auto_rate: float
    eval_precision: float | None  # share of auto-selected eval tokens that matched the true token
    threshold_sweep: list[dict]  # one per tau: tau, auto_rate, precision
    calibration: dict[str, list[dict]]  # per gate group: reliability bins on the eval slice
    class_breakdown: dict[str, dict[str, int]]  # per token class: {"auto": n, "asked": n}


def load_corpus(path: str) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            if not (
                isinstance(obj, dict)
                and isinstance(obj.get("messages"), list)
                and isinstance(obj.get("response"), str)
            ):
                raise ValueError(f"{path}:{lineno}: expected an object with a messages list and a response string")
            items.append(CorpusItem(messages=obj["messages"], response=obj["response"]))
    return items


def make_fake_model(tok: Tokenizer, seed: int = 0) -> Callable[[Sequence[int]], Mapping[int, float]]:
    """A habit-agnostic model: about six token ids per context, decreasing probabilities, some mass unseen."""
    vocab = sorted(tok.valid_ids())

    def model(context: Sequence[int]) -> dict[int, float]:
        last = context[-1] if context else -1
        chosen: list[int] = []
        attempt = 0
        while len(chosen) < _FAKE_FANOUT and attempt < 4 * _FAKE_FANOUT:
            digest = hashlib.blake2b(f"{seed}:{last}:{attempt}".encode("utf-8"), digest_size=8).digest()
            token_id = vocab[int.from_bytes(digest, "little") % len(vocab)]
            if token_id not in chosen:
                chosen.append(token_id)
            attempt += 1
        return {token_id: _FAKE_FIRST_PROB * (_FAKE_DECAY ** rank) for rank, token_id in enumerate(chosen)}

    return model


def _result_to_json(result: NextTokenResult) -> dict[str, Any]:
    return {
        "k": result.k,
        "tail_mass": result.tail_mass,
        "top": [
            {"token_id": t.token_id, "text": t.text, "raw": list(t.raw), "logprob": t.logprob}
            for t in result.top
        ],
    }


def _result_from_json(data: dict[str, Any]) -> NextTokenResult:
    top = [
        TokenLP(
            token_id=int(t["token_id"]),
            text=str(t["text"]),
            raw=bytes(t["raw"]),
            logprob=float(t["logprob"]),
        )
        for t in data["top"]
    ]
    return NextTokenResult(top=top, tail_mass=float(data["tail_mass"]), k=int(data["k"]))


class _CachingBackend:
    """Replay-local cache keyed by (prompt ids, k). Results live in memory, and on disk when cache_dir is set."""

    def __init__(self, backend: InferenceBackend, cache_dir: str | None) -> None:
        self._backend = backend
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._memory: dict[tuple[tuple[int, ...], int], NextTokenResult] = {}

    @property
    def capabilities(self):
        return self._backend.capabilities

    def _path(self, prompt_ids: tuple[int, ...], k: int) -> Path:
        assert self._cache_dir is not None
        key = hashlib.sha256(json.dumps([list(prompt_ids), k]).encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.json"

    async def next_token(self, prompt_ids: Sequence[int], k: int) -> NextTokenResult:
        key = (tuple(int(i) for i in prompt_ids), int(k))
        if key in self._memory:
            return self._memory[key]
        if self._cache_dir is not None:
            path = self._path(key[0], key[1])
            if path.is_file():
                result = _result_from_json(json.loads(path.read_text(encoding="utf-8")))
                self._memory[key] = result
                return result
        result = await self._backend.next_token(list(prompt_ids), k)
        self._memory[key] = result
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._path(key[0], key[1]).write_text(json.dumps(_result_to_json(result)), encoding="utf-8")
        return result


@dataclass
class _EvalStats:
    positions: int = 0
    autos: int = 0
    hits: int = 0
    asks: int = 0
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    points: dict[str, list[tuple[float, bool, float]]] = field(
        default_factory=lambda: {g.value: [] for g in GateGroup}
    )

    def count_class(self, cls: TokenClass, kind: str) -> None:
        bucket = self.class_counts.setdefault(cls.value, {"auto": 0, "asked": 0})
        bucket[kind] += 1


class _Session:
    """Holds the learned state and the per-position plumbing shared by learning and evaluation."""

    def __init__(
        self,
        *,
        cfg: Config,
        tok: Tokenizer,
        backend: _CachingBackend,
        embedder: Embedder,
        rc: ReplayConfig,
        learners: Learners,
        gate: Gate,
    ) -> None:
        self.cfg = cfg
        self.tok = tok
        self.backend = backend
        self.embedder = embedder
        self.rc = rc
        self.learners = learners
        self.gate = gate
        self.last_embedding: Optional[np.ndarray] = None

    # ---- per-position plumbing

    async def _fetch(self, prompt: list[int], response: list[int]) -> Optional[NextTokenResult]:
        try:
            return await self.backend.next_token(prompt + response, self.rc.top_k)
        except BackendUnavailable:
            return None  # the position degrades to a plain ask with no backend candidates

    def _embed_tail(self, response: list[int]) -> np.ndarray:
        text = self.tok.decode(response[-_EMBED_TAIL_TOKENS:])
        return np.asarray(self.embedder.embed([text])[0], dtype=np.float32)

    async def _position(
        self, prompt: list[int], response: list[int]
    ) -> tuple[PositionContext, ScoredPosition, Optional[KnnResult], bool]:
        result = await self._fetch(prompt, response)
        state = position_state(self.tok, response)
        embedded = state is not PosState.MID_WORD or self.last_embedding is None
        if embedded:
            self.last_embedding = self._embed_tail(response)
        ctx = PositionContext(
            response_ids=list(response),
            state=state,
            result=result,
            filtered_probs=bool(self.backend.capabilities.filtered_probs),
            embedding=self.last_embedding,
        )
        knn = self.learners.knn.query(self.last_embedding) if len(self.learners.knn) > 0 else None
        scored = score_position(
            ctx,
            tok=self.tok,
            ngram=self.learners.ngram,
            knn=knn,
            scorer=self.learners.scorer,
            cfg=self.cfg.learning,
        )
        return ctx, scored, knn, embedded

    def _with_true(
        self, ctx: PositionContext, scored: ScoredPosition, knn: Optional[KnnResult], token_id: int
    ) -> ScoredPosition:
        new, _ = with_chosen(
            scored,
            token_id,
            ctx,
            tok=self.tok,
            ngram=self.learners.ngram,
            knn=knn,
            scorer=self.learners.scorer,
            cfg=self.cfg.learning,
        )
        return new

    # ---- records

    @staticmethod
    def _candidate_rows(scored: ScoredPosition) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, cand in enumerate(scored.candidates):
            rows.append(
                {
                    "id": cand.token_id,
                    "p_base": cand.p_base,
                    "p_you": float(scored.probs[i]),
                    "f": [float(x) for x in scored.features[i]],
                    "cls": cand.token_class.value,
                }
            )
        last = len(scored.candidates)
        rows.append(
            {
                "id": OTHER_ID,
                "p_base": None,
                "p_you": float(scored.probs[last]),
                "f": [float(x) for x in scored.features[last]],
                "cls": "other",
            }
        )
        return rows

    def _record(
        self,
        *,
        request_id: int,
        idx: int,
        outcome: Outcome,
        weight: float,
        scored: ScoredPosition,
        decision: Optional[Decision],
        chosen_id: Optional[int] = None,
        rejected_id: Optional[int] = None,
        typed_text: Optional[str] = None,
        feat_vec: Optional[np.ndarray] = None,
    ) -> PositionRecord:
        return PositionRecord(
            request_id=request_id,
            idx=idx,
            outcome=outcome,
            sample_weight=weight,
            candidates=self._candidate_rows(scored),
            chosen_id=chosen_id,
            typed_text=typed_text,
            tail_mass=scored.tail_mass,
            base_k=scored.base_k,
            top_conf=decision.conf if decision is not None else None,
            tau=decision.tau if decision is not None else None,
            group_name=decision.group.value if decision is not None else None,
            rejected_id=rejected_id,
            feat_vec=feat_vec,
        )

    def _apply(self, rec: PositionRecord, target: list[int]) -> None:
        self.learners.apply(rec, target)

    # ---- learning

    async def learn_item(self, request_id: int, item: CorpusItem) -> tuple[int, int]:
        """Run one training response through the gate and the learners. Returns (auto-selected tokens, positions)."""
        prompt = list(self.tok.render_chat(item.messages, None))
        ids = list(self.tok.encode(item.response))
        target = ids + [self.tok.end_id]
        response: list[int] = []
        self.last_embedding = None
        auto_run = 0
        autos = 0
        typed_run = False
        lp = self.cfg.learning

        for i, true_id in enumerate(target):
            is_end = i == len(ids)
            ctx, scored, knn, embedded = await self._position(prompt, response)
            feat_vec = (
                np.asarray(ctx.embedding, dtype=np.float16)
                if embedded and ctx.embedding is not None
                else None
            )
            true_cls = self.tok.token_class(true_id)

            # After an off-list typed first token, following continuation tokens are labeled without asking.
            if typed_run and true_cls is TokenClass.CONTINUATION:
                full = self._with_true(ctx, scored, knn, true_id)
                rec = self._record(
                    request_id=request_id,
                    idx=i,
                    outcome=Outcome.TYPED,
                    weight=lp.weight_typed_continuation,
                    scored=full,
                    decision=None,
                    chosen_id=true_id,
                    typed_text=self.tok.token_text(true_id) or "",
                )
                self._apply(rec, target)
                response.append(true_id)
                auto_run = 0
                continue
            typed_run = False

            decision = self.gate.decide(scored, auto_run_len=auto_run, force_ask=is_end)
            action = decision.action
            listed = any(c.token_id == true_id for c in scored.candidates)

            if isinstance(action, Auto):
                if action.token_id == true_id:
                    self.gate.controller.record_auto()
                    autos += 1
                    auto_run += 1
                    response.append(true_id)
                    continue
                # Rewind: the auto token was wrong. Record it, then label the true token as a normal row.
                rewound = self._record(
                    request_id=request_id,
                    idx=i,
                    outcome=Outcome.REWOUND,
                    weight=1.0,
                    scored=scored,
                    decision=decision,
                    rejected_id=action.token_id,
                    feat_vec=feat_vec,
                )
                self._apply(rewound, target)
                self.gate.controller.record_rewind()
                outcome = Outcome.PICK if listed else Outcome.TYPED
                weight = 1.0 if listed else lp.weight_typed_first
            elif action.explore:
                outcome = Outcome.EXPLORE
                weight = 1.0 / self.cfg.gate.explore_rate
            else:
                outcome = Outcome.PICK if listed else Outcome.TYPED
                weight = 1.0 if listed else lp.weight_typed_first

            full = self._with_true(ctx, scored, knn, true_id)
            rec = self._record(
                request_id=request_id,
                idx=i,
                outcome=outcome,
                weight=weight,
                scored=full,
                decision=decision,
                chosen_id=true_id,
                typed_text=(self.tok.token_text(true_id) or "") if outcome is Outcome.TYPED else None,
                feat_vec=feat_vec,
            )
            self._apply(rec, target)
            if outcome is Outcome.TYPED:
                typed_run = True
            response.append(true_id)
            auto_run = 0
        return autos, len(target)

    # ---- evaluation (read only)

    async def evaluate(self, items: Sequence[CorpusItem], gate: Gate) -> _EvalStats:
        stats = _EvalStats()
        for item in items:
            prompt = list(self.tok.render_chat(item.messages, None))
            ids = list(self.tok.encode(item.response))
            target = ids + [self.tok.end_id]
            response: list[int] = []
            self.last_embedding = None
            auto_run = 0
            for i, true_id in enumerate(target):
                is_end = i == len(ids)
                _, scored, _, _ = await self._position(prompt, response)
                decision = gate.decide(scored, auto_run_len=auto_run, force_ask=is_end)
                stats.positions += 1
                action = decision.action
                if isinstance(action, Auto):
                    stats.autos += 1
                    auto_run += 1
                    if action.token_id == true_id:
                        stats.hits += 1
                    stats.count_class(self.tok.token_class(action.token_id), "auto")
                else:
                    stats.asks += 1
                    auto_run = 0
                    stats.count_class(self.tok.token_class(true_id), "asked")
                if decision.top_id is not None:
                    stats.points[decision.group.value].append(
                        (decision.conf, decision.top_id == true_id, 1.0)
                    )
                response.append(true_id)
        return stats


async def run_replay(
    items: Sequence[CorpusItem],
    *,
    cfg: Config,
    tok: Tokenizer,
    backend: InferenceBackend,
    embedder: Embedder,
    rc: ReplayConfig | None = None,
) -> ReplayResult:
    rc = rc if rc is not None else ReplayConfig()
    n_train = int(len(items) * rc.train_fraction)
    train, frozen = list(items[:n_train]), list(items[n_train:])

    learners = Learners(cfg, embed_dim=embedder.dim, vocab_size=tok.vocab_size)
    gate = Gate(cfg.gate, calibrator=learners.calibrator, rng=random.Random(rc.seed))
    session = _Session(
        cfg=cfg,
        tok=tok,
        backend=_CachingBackend(backend, rc.cache_dir),
        embedder=embedder,
        rc=rc,
        learners=learners,
        gate=gate,
    )

    # Learning, with the learning curve taken over windows of curve_window responses.
    curve: list[tuple[int, float]] = []
    window_autos = 0
    window_positions = 0
    for index, item in enumerate(train):
        autos, positions = await session.learn_item(index, item)
        window_autos += autos
        window_positions += positions
        seen = index + 1
        if seen % rc.curve_window == 0:
            curve.append((seen, metrics.auto_select_rate(window_autos, window_positions)))
            window_autos = 0
            window_positions = 0

    # Frozen evaluation with the learned state. Exploration is off and nothing is updated.
    eval_gate = Gate(
        replace(cfg.gate, explore_rate=0.0),
        calibrator=learners.calibrator,
        rng=random.Random(rc.seed),
    )
    eval_gate.controller = gate.controller
    stats = await session.evaluate(frozen, eval_gate)

    sweep: list[dict] = []
    for tau in rc.sweep_taus:
        sweep_gate = Gate(
            replace(
                cfg.gate,
                tau_content=tau,
                tau_function=tau,
                tau_sentence_start=tau,
                autonomy=0.0,
                explore_rate=0.0,
            ),
            calibrator=learners.calibrator,
            rng=random.Random(rc.seed),
        )
        sweep_gate.controller = gate.controller
        sweep_stats = await session.evaluate(frozen, sweep_gate)
        sweep.append(
            {
                "tau": float(tau),
                "auto_rate": metrics.auto_select_rate(sweep_stats.autos, sweep_stats.positions),
                "precision": metrics.precision(sweep_stats.hits, sweep_stats.autos),
            }
        )

    return ReplayResult(
        n_train=len(train),
        n_eval=len(frozen),
        learning_curve=curve,
        eval_auto_rate=metrics.auto_select_rate(stats.autos, stats.positions),
        eval_precision=metrics.precision(stats.hits, stats.autos),
        threshold_sweep=sweep,
        calibration={
            name: metrics.reliability_bins(points) for name, points in stats.points.items()
        },
        class_breakdown={name: dict(counts) for name, counts in stats.class_counts.items()},
    )


def _summary(result: ReplayResult) -> str:
    lines = [
        f"train responses: {result.n_train}, eval responses: {result.n_eval}",
        f"eval auto-select rate: {result.eval_auto_rate:.3f}",
        "eval precision: " + ("n/a" if result.eval_precision is None else f"{result.eval_precision:.3f}"),
    ]
    if result.learning_curve:
        first = result.learning_curve[0][1]
        last = result.learning_curve[-1][1]
        lines.append(f"learning curve: {first:.3f} -> {last:.3f} over {len(result.learning_curve)} windows")
    for row in result.threshold_sweep:
        prec = "n/a" if row["precision"] is None else f"{row['precision']:.3f}"
        lines.append(f"  tau {row['tau']:.2f}: auto {row['auto_rate']:.3f} precision {prec}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a corpus through the real scorer, gate and learners.")
    parser.add_argument("--corpus", required=True, help="JSON Lines corpus file")
    parser.add_argument("--config", default=None, help="config.yaml path")
    parser.add_argument("--fake", action="store_true", help="CPU-only run: ToyTokenizer, FakeBackend, HashEmbedder")
    parser.add_argument("--out", default=None, help="write the ReplayResult as JSON to this path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    items = load_corpus(args.corpus)
    rc = ReplayConfig()

    if args.fake:
        from humanapi.backend.fake import FakeBackend
        from humanapi.learn.embed import HashEmbedder
        from humanapi.text.toy_tokenizer import ToyTokenizer

        tok: Tokenizer = ToyTokenizer()
        backend: InferenceBackend = FakeBackend(
            make_fake_model(tok, rc.seed),
            token_bytes=lambda i: tok.token_bytes(i) or b"",
        )
        embedder: Embedder = HashEmbedder()
    else:
        from humanapi.backend.openai_compat import OpenAICompatBackend, load_capabilities
        from humanapi.learn.embed import SentenceTransformerEmbedder
        from humanapi.text.tokenizer import HumanTokenizer

        tok = HumanTokenizer(cfg.tokenizer.path, enable_thinking=cfg.tokenizer.enable_thinking)
        backend = OpenAICompatBackend(
            cfg.backend.base_url,
            cfg.backend.model,
            tok,
            load_capabilities(cfg.backend.capabilities_file),
            timeout_s=cfg.backend.timeout_s,
        )
        embedder = SentenceTransformerEmbedder(cfg.embed.model, device=cfg.embed.device)

    async def run() -> ReplayResult:
        try:
            return await run_replay(items, cfg=cfg, tok=tok, backend=backend, embedder=embedder, rc=rc)
        finally:
            aclose = getattr(backend, "aclose", None)
            if aclose is not None:
                await aclose()

    result = asyncio.run(run())
    print(_summary(result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
