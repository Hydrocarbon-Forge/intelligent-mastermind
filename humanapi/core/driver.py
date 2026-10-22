"""Per-request driver: walks one request position by position, auto-selects or asks, and commits decisions.

Structure (ENGINE2 extends this without restructuring):
- one `inbox` queue carries every UI message for the request and the engine's cancel signal (`Cancel`);
- `_classify` is the single place that decides whether an inbox message is a decision, a stale message or a
  control message. Rewind is accepted and ignored here: ENGINE2 adds its handler in `_classify`;
- `force_ask_at` / `_force_ask(idx)` is the hook that forces an ask at an index (the `force_ask` argument of
  `gate.decide`); it is never set in this task;
- every log write goes through the engine's ordered write queue, so rows follow creation order.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

from humanapi.api.schemas import Ask as AskMessage
from humanapi.api.schemas import (
    AskCandidate,
    AutoRun,
    AutoToken,
    Decline,
    Pick,
    Rewind,
    Rewound,
    RequestActive,
    RequestDone,
    Send,
    Typed,
)
from humanapi.backend.base import BackendUnavailable, NextTokenResult
from humanapi.core.session import Completion, DeclinedError, Request
from humanapi.learn.knn import KnnResult
from humanapi.learn.scorer import score_position, with_chosen
from humanapi.learn.types import OTHER_ID, Ask as GateAsk, Auto as GateAuto, Decision, Outcome
from humanapi.learn.types import PositionContext, PositionRecord, ScoredPosition
from humanapi.text.typed import Utf8Stream, commit_typed, position_state, space_default
from humanapi.text.types import PosState

log = logging.getLogger(__name__)

_PROMPT_TAIL_TOKENS = 200


class _Cancelled(Exception):
    """The client went away (engine.cancel or handle.cancel)."""


class _Declined(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Cancel:
    """Engine-internal control message placed in a driver's inbox."""


@dataclass
class _Provisional:
    token_id: int
    idx: int
    confidence: float
    record: PositionRecord  # AUTO row, written when the buffer is flushed


@dataclass
class _Position:
    idx: int
    ids: list[int]  # full response ids (committed + provisional) when the position was scored
    started_at: float
    ctx: Optional[PositionContext]
    scored: Optional[ScoredPosition]  # None when the backend was unavailable (plain ask, no candidates)
    decision: Optional[Decision]
    knn: Optional[KnnResult]
    vec: Optional[np.ndarray]  # embedding computed at this position; None when reused or not computed
    forced: bool = False  # the ask was forced (rewind target or max_auto_run): not a calibration observation


@dataclass
class _OpenAsk:
    pos_id: int
    pos: _Position
    explore: bool
    shown_ids: frozenset[int]
    message: AskMessage
    opened_at: float


class Driver:
    def __init__(self, engine: Any, req: Request) -> None:
        self.engine = engine
        self.req = req
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.finished = asyncio.Event()
        self.active_msg: Optional[RequestActive] = None
        self.ask: Optional[_OpenAsk] = None
        self.committed: list[int] = []
        self.provisional: list[_Provisional] = []
        self.prompt_ids: list[int] = []
        self.retrieval: Any = None
        self.retrieval_ids: frozenset[int] = frozenset()
        self.pending_send = False
        self.max_tokens: Optional[int] = req.max_tokens
        self.stops: list[str] = [s for s in req.stop if s]
        self.force_ask_at: Optional[int] = None  # ENGINE2 hook: force an ask at this index
        self._user_ids: list[int] = []
        self._last_knn: Optional[KnnResult] = None
        self._utf8 = Utf8Stream(engine.tokenizer)
        self._decoded = ""  # complete-character text committed so far
        self._emitted = ""  # text already put on the handle's stream
        self._stop_cut: Optional[int] = None
        self._done = False
        # Speculative lookahead: NextTokenResult per response-id tuple, filled by score_continuation.
        self._lookahead: dict[tuple[int, ...], NextTokenResult] = {}

    # ------------------------------------------------------------ public

    def post(self, msg: Any) -> None:
        self.inbox.put_nowait(msg)

    def ask_message(self) -> Optional[AskMessage]:
        return self.ask.message if self.ask is not None else None

    async def run(self) -> None:
        try:
            await self._activate()
            while not self._done:
                self._drain_inbox()
                if self._done:
                    break
                if self._stop_cut is not None:
                    self._finish("stop")
                    break
                if self.max_tokens is not None and self._total_len() >= self.max_tokens:
                    self._flush_provisional()
                    self._emit()
                    self._finish("length")
                    break
                pos = await self._position()
                if self.pending_send:
                    self.pending_send = False
                    await self._do_send(pos)
                    break
                if pos.scored is not None and isinstance(pos.decision.action, GateAuto):
                    await self._auto(pos)
                    continue
                await self._run_ask(pos)
        except _Cancelled:
            self._cancel()
        except _Declined as exc:
            self._decline(exc.reason)
        except asyncio.CancelledError:
            self._fail(RuntimeError("engine stopped"))
            raise
        except Exception as exc:  # an unexpected error ends this request only
            log.exception("driver failed for request %s", self.req.id)
            self._fail(exc)
        finally:
            self.ask = None
            self.finished.set()

    # ------------------------------------------------------------ activation

    async def _activate(self) -> None:
        engine = self.engine
        tok = engine.tokenizer
        messages = self.req.messages
        query = "\n".join(_message_text(m) for m in messages[-4:])
        retrieval = await asyncio.to_thread(engine.knowledge.retrieve, query)
        self.retrieval = retrieval
        self.retrieval_ids = engine.knowledge.token_ids(retrieval.chunks)
        knowledge_text = engine.knowledge.knowledge_text(retrieval.chunks)
        prompt_ids = list(tok.render_chat(messages, knowledge_text))
        self.prompt_ids = prompt_ids
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        self._user_ids = tok.encode(_message_text(last_user)) if last_user is not None else []
        request_id = self.req.id
        engine.enqueue_write(lambda: engine.log.activate_request(request_id, prompt_ids))
        self.active_msg = RequestActive(
            request_id=request_id,
            prompt_tail=tok.decode(prompt_ids[-_PROMPT_TAIL_TOKENS:]),
            max_tokens=self.max_tokens,
            knowledge=[c.text for c in retrieval.chunks],
        )
        engine.broadcast(self.active_msg)

    # ------------------------------------------------------------ inbox

    def _drain_inbox(self) -> None:
        while True:
            try:
                msg = self.inbox.get_nowait()
            except asyncio.QueueEmpty:
                return
            if self._classify(msg) == "rewind":
                self._apply_rewind(int(msg.to_idx))  # no ask is open between positions

    def _classify(self, msg: Any) -> Optional[str]:
        """Return 'pick' | 'typed' | 'send' for a decision that applies now, otherwise None.
        Raises _Cancelled / _Declined for control messages. Stale or invalid messages are ignored here."""
        if isinstance(msg, Cancel):
            raise _Cancelled()
        if isinstance(msg, Decline):
            raise _Declined(msg.reason)
        if isinstance(msg, Send):
            if self.ask is None:
                self.pending_send = True  # remembered; honoured at the next position
                return None
            return "send"
        if isinstance(msg, Pick):
            if self.ask is not None and msg.pos_id == self.ask.pos_id and msg.token_id in self.ask.shown_ids:
                return "pick"
            return None
        if isinstance(msg, Typed):
            if self.ask is not None and msg.pos_id == self.ask.pos_id and msg.final:
                return "typed"
            return None
        if isinstance(msg, Rewind):
            return "rewind"
        return None

    # ------------------------------------------------------------ positions

    def _total_len(self) -> int:
        return len(self.committed) + len(self.provisional)

    def _full_ids(self) -> list[int]:
        return self.committed + [p.token_id for p in self.provisional]

    def _force_ask(self, idx: int) -> bool:
        """Forces the ask at one index (the rewind target). Consumed when that position is scored."""
        if self.force_ask_at == idx:
            self.force_ask_at = None
            return True
        return False

    async def _position(self) -> _Position:
        engine = self.engine
        tok = engine.tokenizer
        started = engine.clock()
        ids = self._full_ids()
        idx = len(ids)
        state = position_state(tok, ids)
        cached = self._lookahead.pop(tuple(ids), None)
        result: Optional[NextTokenResult] = cached
        if cached is None:
            try:
                result = await engine.backend.next_token(self.prompt_ids + ids, engine.cfg.backend.top_k)
            except BackendUnavailable:
                result = None
        vec, knn = await self._knn_for(ids, state)
        forced = self._force_ask(idx) or len(self.provisional) >= engine.cfg.gate.max_auto_run
        if result is None:
            return _Position(idx, ids, started, None, None, None, knn, vec, forced)
        ctx = self._context(ids, state, result, vec)
        scored = score_position(
            ctx,
            tok=tok,
            ngram=engine.learners.ngram,
            knn=knn,
            scorer=engine.learners.scorer,
            cfg=engine.cfg.learning,
        )
        decision = engine.gate.decide(scored, auto_run_len=len(self.provisional), force_ask=forced)
        return _Position(idx, ids, started, ctx, scored, decision, knn, vec, forced)

    def _context(
        self, ids: list[int], state: PosState, result: Optional[NextTokenResult], vec: Optional[np.ndarray]
    ) -> PositionContext:
        return PositionContext(
            response_ids=list(ids),
            state=state,
            typed_prefix="",
            result=result,
            filtered_probs=bool(self.engine.backend.capabilities.filtered_probs),
            embedding=vec,
            retrieval_hit=bool(self.retrieval.hit),
            retrieval_max_sim=float(self.retrieval.max_sim),
            retrieval_ids=self.retrieval_ids,
        )

    async def _knn_for(self, ids: list[int], state: PosState) -> tuple[Optional[np.ndarray], Optional[KnnResult]]:
        """kNN features are computed at word starts only; mid-word positions reuse the previous result."""
        if state is PosState.MID_WORD:
            return None, self._last_knn
        vec = await asyncio.to_thread(self._embed_tail, ids)
        index = self.engine.learners.knn
        self._last_knn = index.query(vec) if len(index) > 0 else None
        return vec, self._last_knn

    def _embed_tail(self, ids: list[int]) -> np.ndarray:
        engine = self.engine
        window = engine.cfg.embed.context_tokens
        tail = (self._user_ids + list(ids))[-window:] if window > 0 else []
        text = engine.tokenizer.decode(tail)
        return np.asarray(engine.embedder.embed([text]), dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------ auto-selection

    async def _auto(self, pos: _Position) -> None:
        engine = self.engine
        decision = pos.decision
        token_id = int(decision.action.token_id)
        rec = self._record(
            request_id=self.req.id,
            idx=pos.idx,
            outcome=Outcome.AUTO,
            chosen=token_id,
            weight=0.0,
            typed_text=None,
            scored=pos.scored,
            decision=decision,
            feat_vec=pos.vec,
            started_at=pos.started_at,
        )
        self.provisional.append(_Provisional(token_id, pos.idx, float(decision.conf), rec))
        engine.gate.controller.record_auto()
        engine.stats["auto"] += 1
        engine.broadcast(
            AutoRun(
                request_id=self.req.id,
                tokens=[
                    AutoToken(
                        idx=pos.idx,
                        token_id=token_id,
                        display=engine.tokenizer.display(token_id),
                        confidence=float(decision.conf),
                    )
                ],
            )
        )
        await self._lookahead_after_auto(pos.ids, token_id)

    # ------------------------------------------------------------ speculative lookahead

    def _lookahead_enabled(self) -> bool:
        return bool(self.engine.backend.capabilities.prompt_logprobs and self.engine.cfg.gate.lookahead_max > 0)

    async def _lookahead_after_auto(self, ids_before: list[int], token_id: int) -> None:
        """Draft ahead from the n-gram memory and cache the server's distributions along the draft.

        One score_continuation call over [token_id] + draft. Result j (j >= 1) is the distribution after
        [token_id] + draft[:j-1], cached under that exact response-id tuple. Index 0 is the decided position.
        """
        if not self._lookahead_enabled():
            return
        engine = self.engine
        after = tuple(ids_before) + (token_id,)
        if after in self._lookahead:
            return  # the cached draft already covers the next position
        draft = engine.learners.ngram.draft(list(after), engine.cfg.gate.lookahead_max, 0.5)
        eos = engine.tokenizer.eos_ids
        cut = next((i for i, t in enumerate(draft) if t in eos), len(draft))
        draft = draft[:cut]  # END is never drafted
        if not draft:
            return
        cont = [token_id] + draft
        try:
            results = await engine.backend.score_continuation(
                self.prompt_ids + list(ids_before), cont, engine.cfg.backend.top_k
            )
        except BackendUnavailable:
            return  # lookahead is optional: the run continues with one call per position
        for j in range(1, min(len(results), len(cont))):
            self._lookahead[tuple(ids_before) + tuple(cont[:j])] = results[j]

    # ------------------------------------------------------------ rewind

    def _apply_rewind(self, to_idx: int) -> bool:
        """Truncate the provisional buffer at to_idx, write the rejected token's rewound row (a penalty, never a
        label), and force an ask at to_idx. Returns False (message ignored) when nothing can be rewound."""
        if not self.provisional:
            return False
        flushed = len(self.committed)
        target = max(int(to_idx), flushed)  # flushed tokens are never retracted
        if target >= self._total_len():
            return False
        pos_in_buffer = target - flushed
        item = self.provisional[pos_in_buffer]
        self.provisional = self.provisional[:pos_in_buffer]
        self.ask = None  # an open ask is replaced by the forced ask below
        self._lookahead.clear()
        engine = self.engine
        engine.gate.controller.record_rewind()
        rewound = replace(
            item.record,
            outcome=Outcome.REWOUND,
            sample_weight=1.0,
            chosen_id=None,
            rejected_id=item.token_id,
            group_name=None,
            top_conf=None,
            tau=None,
            feat_vec=None,
            pre_update_p=None,
            id=None,
            created_at=None,
        )
        self._enqueue_label(rewound)  # Learners.apply applies only the rewind penalty to it
        self.force_ask_at = target
        engine.broadcast(Rewound(request_id=self.req.id, to_idx=target))
        return True

    def _flush_provisional(self) -> None:
        """A decision accepts the provisional buffer: commit its tokens and write their AUTO rows in order."""
        self._lookahead.clear()  # drafted positions past a decision are discarded
        for item in self.provisional:
            self.committed.append(item.token_id)
            rec = item.record
            self.engine.enqueue_write(lambda rec=rec: self.engine.log.record_position(rec))
        self.provisional = []

    # ------------------------------------------------------------ asking

    async def _run_ask(self, pos: _Position) -> None:
        self._open_ask(pos)
        while True:
            msg = await self.inbox.get()
            kind = self._classify(msg)
            if kind is None:
                continue
            if kind == "rewind":
                if self._apply_rewind(int(msg.to_idx)):
                    return
                continue
            if kind == "pick":
                await self._do_pick(int(msg.token_id), pos)
                return
            if kind == "typed":
                if await self._do_typed(msg, pos):
                    return
                continue  # a word that changes nothing leaves the ask open
            await self._do_send(pos)
            return

    def _open_ask(self, pos: _Position) -> None:
        engine = self.engine
        tok = engine.tokenizer
        self._lookahead.clear()  # a position that asks discards the drafted continuation
        explore = bool(pos.decision is not None and isinstance(pos.decision.action, GateAsk)
                       and pos.decision.action.explore)
        full = self._full_ids()
        candidates: list[AskCandidate] = []
        other_mass = 1.0
        if pos.scored is not None:
            scored = pos.scored
            order = sorted(range(len(scored.candidates)), key=lambda i: -float(scored.probs[i]))
            for i in order[: engine.cfg.ui.max_rows]:
                cand = scored.candidates[i]
                candidates.append(
                    AskCandidate(
                        id=int(cand.token_id),
                        display=tok.display(cand.token_id),
                        p_you=float(scored.probs[i]),
                        p_base=None if cand.base_unknown else float(cand.p_base),
                        cls=cand.token_class.value,
                    )
                )
            other_mass = float(scored.p_other)
        pos_id = engine.next_pos_id()
        message = AskMessage(
            request_id=self.req.id,
            pos_id=pos_id,
            idx=pos.idx,
            committed_text=tok.decode(self.committed),
            provisional=[
                AutoToken(
                    idx=p.idx,
                    token_id=p.token_id,
                    display=tok.display(p.token_id),
                    confidence=float(p.confidence),
                )
                for p in self.provisional
            ],
            space_default=space_default(tok.decode(full), ""),
            candidates=candidates,
            other_mass=other_mass,
            explore=explore,
            tokens_remaining=(self.max_tokens - len(full)) if self.max_tokens is not None else None,
        )
        self.ask = _OpenAsk(
            pos_id=pos_id,
            pos=pos,
            explore=explore,
            shown_ids=frozenset(c.id for c in candidates),
            message=message,
            opened_at=engine.clock(),
        )
        engine.stats["asks"] += 1
        engine.broadcast(message)

    async def _do_pick(self, token_id: int, pos: _Position) -> None:
        engine = self.engine
        if token_id == engine.tokenizer.end_id:
            await self._do_send(pos)
            return
        explore = self.ask.explore
        self.ask = None
        self._flush_provisional()
        self.committed.append(token_id)
        rec = self._record(
            request_id=self.req.id,
            idx=pos.idx,
            outcome=Outcome.EXPLORE if explore else Outcome.PICK,
            chosen=token_id,
            weight=engine.cfg.learning.weight_typed_first,
            typed_text=None,
            scored=pos.scored,
            decision=pos.decision,
            feat_vec=pos.vec,
            started_at=pos.started_at,
            forced=pos.forced,
        )
        self._enqueue_label(rec)
        self._emit()

    async def _do_send(self, pos: _Position) -> None:
        engine = self.engine
        end_id = engine.tokenizer.end_id
        explore = self.ask.explore if self.ask is not None else False
        self.ask = None
        self._flush_provisional()
        scored = pos.scored
        if scored is not None:
            scored, _ = with_chosen(
                scored,
                end_id,
                pos.ctx,
                tok=engine.tokenizer,
                ngram=engine.learners.ngram,
                knn=pos.knn,
                scorer=engine.learners.scorer,
                cfg=engine.cfg.learning,
            )
        rec = self._record(
            request_id=self.req.id,
            idx=len(self.committed),
            outcome=Outcome.EXPLORE if explore else Outcome.PICK,
            chosen=end_id,
            weight=engine.cfg.learning.weight_typed_first,
            typed_text=None,
            scored=scored,
            decision=pos.decision,
            feat_vec=pos.vec,
            started_at=pos.started_at,
            forced=pos.forced,
        )
        self._enqueue_label(rec)
        self._emit()
        self._finish("stop")

    async def _do_typed(self, msg: Typed, pos: _Position) -> bool:
        """Commit a typed word. Returns False when the word changes nothing (the ask stays open)."""
        engine = self.engine
        tok = engine.tokenizer
        cfg = engine.cfg.learning
        full = self._full_ids()
        k, new_ids = commit_typed(tok, tok.decode(full), full, msg.text, msg.leading_space)
        if not new_ids and k == len(full):
            return False
        if self.max_tokens is not None:
            new_ids = new_ids[: max(0, self.max_tokens - k)]
        explore = self.ask.explore
        self.ask = None
        self._flush_provisional()
        final_ids = full[:k] + list(new_ids)
        asked_here = k == pos.idx  # the first new token sits at the asked position

        # One teacher-forced pass over the new tokens; the asked position reuses the scoring the person saw.
        results: Optional[list[NextTokenResult]] = None
        if new_ids:
            try:
                results = await engine.backend.score_continuation(
                    self.prompt_ids + full[:k], list(new_ids), engine.cfg.backend.top_k
                )
            except BackendUnavailable:
                results = None

        for j, token_id in enumerate(new_ids):
            p = k + j
            is_asked = asked_here and j == 0
            if is_asked:
                scored, decision, knn, vec, ctx = pos.scored, pos.decision, pos.knn, pos.vec, pos.ctx
            elif results is None:
                scored = decision = knn = vec = ctx = None
            else:
                ctx_ids = final_ids[:p]
                state = position_state(tok, ctx_ids)
                vec, knn = await self._knn_for(ctx_ids, state)
                ctx = self._context(ctx_ids, state, results[j], vec)
                scored = score_position(
                    ctx,
                    tok=tok,
                    ngram=engine.learners.ngram,
                    knn=knn,
                    scorer=engine.learners.scorer,
                    cfg=cfg,
                )
                decision = None  # no gate decision for continuation tokens: no group, no calibration
                vec = None  # continuation embeddings are used for scoring but not stored
            if scored is not None:
                scored, _ = with_chosen(
                    scored, token_id, ctx, tok=tok, ngram=engine.learners.ngram, knn=knn,
                    scorer=engine.learners.scorer, cfg=cfg,
                )
            rec = self._record(
                request_id=self.req.id,
                idx=p,
                outcome=Outcome.EXPLORE if (is_asked and explore) else Outcome.TYPED,
                chosen=token_id,
                weight=cfg.weight_typed_first if is_asked else cfg.weight_typed_continuation,
                typed_text=msg.text,
                scored=scored,
                decision=decision,
                feat_vec=vec if is_asked else None,
                started_at=pos.started_at,
                forced=pos.forced and is_asked,
            )
            self._enqueue_label(rec, ids=final_ids)
        self.committed = final_ids
        self._emit()
        return True

    # ------------------------------------------------------------ records and labels

    def _cand_dicts(self, scored: ScoredPosition) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, cand in enumerate(scored.candidates):
            out.append(
                {
                    "id": int(cand.token_id),
                    "p_base": None if cand.base_unknown else float(cand.p_base),
                    "p_you": float(scored.probs[i]),
                    "f": [float(x) for x in scored.features[i]],
                    "cls": cand.token_class.value,
                }
            )
        out.append(
            {
                "id": OTHER_ID,
                "p_base": None,
                "p_you": float(scored.probs[-1]),
                "f": [float(x) for x in scored.features[-1]],
                "cls": "other",
            }
        )
        return out

    def _record(
        self,
        *,
        request_id: int,
        idx: int,
        outcome: Outcome,
        chosen: int,
        weight: float,
        typed_text: Optional[str],
        scored: Optional[ScoredPosition],
        decision: Optional[Decision],
        feat_vec: Optional[np.ndarray],
        started_at: float,
        forced: bool = False,
    ) -> PositionRecord:
        """Build one position row. The group, confidence and threshold are stored only when the position had a
        gate decision (the asked or auto-selected position), so only those rows can calibrate."""
        engine = self.engine
        has_group = decision is not None and scored is not None and not forced
        return PositionRecord(
            request_id=request_id,
            idx=idx,
            outcome=outcome,
            sample_weight=float(weight),
            candidates=self._cand_dicts(scored) if scored is not None else [],
            chosen_id=chosen,
            typed_text=typed_text,
            tail_mass=float(scored.tail_mass) if scored is not None else None,
            base_k=int(scored.base_k) if scored is not None else None,
            top_conf=float(decision.conf) if has_group else None,
            tau=float(decision.tau) if has_group else None,
            group_name=decision.group.value if has_group else None,
            rejected_id=None,
            feat_vec=None if feat_vec is None else np.asarray(feat_vec, dtype=np.float16),
            latency_ms=int(1000 * (engine.clock() - started_at)),
            scorer_ver=int(engine.learners.scorer.n_updates),
        )

    def _enqueue_label(self, rec: PositionRecord, ids: Optional[list[int]] = None) -> None:
        """Label rows: Learners.apply runs inside the ordered write queue, then the row is written with its
        prequential probability. The response ids are snapshotted now, so later changes cannot leak in."""
        engine = self.engine
        snapshot = list(ids) if ids is not None else list(self.committed)

        def job() -> None:
            rec.pre_update_p = engine.learners.apply(rec, snapshot)
            engine.log.record_position(rec)
            engine.learners.maybe_snapshot(engine.log)

        engine.enqueue_write(job)

    # ------------------------------------------------------------ streaming and finishing

    def _emit(self, final: bool = False) -> None:
        """Stream newly committed text. Held back: an incomplete character, and, while a stop sequence could
        still complete, the last max(len(stop)) - 1 characters. A stop sequence cuts the text at its start."""
        chunk = self._utf8.finish(self.committed) if final else self._utf8.sync(self.committed)
        self._decoded += chunk
        text = self._decoded
        if self.stops:
            cuts = [text.find(s) for s in self.stops if s in text]
            if cuts:
                if self._stop_cut is None:
                    self._stop_cut = min(cuts)
                self._write_stream(text[: self._stop_cut])
                return
            hold = max(len(s) for s in self.stops) - 1
            self._write_stream(text if final else text[: max(0, len(text) - hold)])
            return
        self._write_stream(text)

    def _write_stream(self, safe: str) -> None:
        if len(safe) > len(self._emitted) and safe.startswith(self._emitted):
            self.req.handle.stream.put_nowait(safe[len(self._emitted):])
            self._emitted = safe

    def _final_text(self) -> str:
        return self._decoded if self._stop_cut is None else self._decoded[: self._stop_cut]

    def _finish(self, reason: str, status: str = "done") -> None:
        """Finish the request with the committed text. Callers flush or discard provisional tokens first."""
        engine = self.engine
        self._lookahead.clear()
        self._emit(final=True)
        text = self._final_text()
        ids = list(self.committed)
        self._done = True
        request_id = self.req.id
        engine.enqueue_write(
            lambda: engine.log.finish_request(
                request_id, status=status, finish_reason=reason, final_text=text, final_ids=ids
            )
        )
        self.req.handle.close_stream()
        self.req.handle.resolve(
            Completion(
                text=text,
                finish_reason=reason,
                prompt_tokens=len(self.prompt_ids),
                completion_tokens=len(ids),
            )
        )
        engine.broadcast(RequestDone(request_id=request_id, finish_reason=reason))

    def _cancel(self) -> None:
        self.provisional = []  # discarded, never written
        self._finish("cancelled", status="cancelled")

    def _decline(self, reason: str) -> None:
        engine = self.engine
        self.provisional = []  # discarded, never written
        self._lookahead.clear()
        self._done = True
        request_id = self.req.id
        text = self._final_text()
        ids = list(self.committed)
        engine.enqueue_write(
            lambda: engine.log.finish_request(
                request_id, status="cancelled", finish_reason="declined", final_text=text, final_ids=ids
            )
        )
        self.req.handle.close_stream()
        self.req.handle.fail(DeclinedError(reason))
        engine.broadcast(RequestDone(request_id=request_id, finish_reason="declined"))

    def _fail(self, exc: BaseException) -> None:
        if self._done:
            return
        engine = self.engine
        self.provisional = []
        self._lookahead.clear()
        self._done = True
        request_id = self.req.id
        ids = list(self.committed)
        engine.enqueue_write(
            lambda: engine.log.finish_request(
                request_id, status="error", finish_reason="error", final_text=None, final_ids=ids
            )
        )
        self.req.handle.close_stream()
        self.req.handle.fail(RuntimeError(f"request failed: {exc}"))
        engine.broadcast(RequestDone(request_id=request_id, finish_reason="error"))


def _message_text(message: Optional[dict]) -> str:
    if not message:
        return ""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            (part.get("text") or "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)
