import asyncio
import random

import pytest

from humanapi.api.schemas import (
    ChatCompletionRequest,
    Decline,
    Pick,
    Send,
    Sync,
    Typed,
)
from humanapi.backend.base import BackendUnavailable
from humanapi.backend.fake import FakeBackend
from humanapi.config import Config
from humanapi.core.engine import Engine
from humanapi.core.queue import RequestQueue
from humanapi.core.session import DeclinedError, QueueFull, Request, RequestHandle, RequestStatus
from humanapi.learn.embed import HashEmbedder
from humanapi.text.toy_tokenizer import ToyTokenizer
from tests.support.scripted_person import ScriptedPerson

TOK = ToyTokenizer()
END = TOK.end_id
NL = TOK.encode("\n")[0]
HELLO = TOK.encode(" Hello")[0]
THE = TOK.encode(" the")[0]
YES = TOK.encode(" yes")[0]
BANG = TOK.encode("!")[0]
USER = [{"role": "user", "content": "hi"}]
FIRST = {HELLO: 0.4, THE: 0.3, YES: 0.2, END: 0.1}
FLAT = {HELLO: 0.3, THE: 0.3, YES: 0.3, END: 0.1}


def _model(first, default, table=None):
    """Distribution by the last context token: `first` right after the prompt, `table[last]` or `default` after."""

    def model(ctx):
        last = ctx[-1]
        if last == NL:
            return first
        if table and last in table:
            return table[last]
        return default

    return model


def make_engine(model, *, cfg=None, backend_cls=FakeBackend, embedder=None, status_interval_s=1000.0):
    cfg = cfg if cfg is not None else Config()
    cfg.gate.cold_start_min_labels = 0
    backend = backend_cls(model, token_bytes=lambda i: TOK.token_bytes(i) or b"")
    engine = Engine.create(
        cfg,
        backend=backend,
        tokenizer=TOK,
        embedder=embedder or HashEmbedder(dim=32),
        rng=random.Random(0),
        status_interval_s=status_interval_s,
    )
    return engine, backend


def request(messages=USER, **extra):
    return ChatCompletionRequest(messages=messages, **extra)


async def collect(handle):
    parts = []
    while True:
        item = await handle.stream.get()
        if item is None:
            return "".join(parts)
        parts.append(item)


def labels_log(engine):
    rows = engine.conn.execute(
        "SELECT outcome, sample_weight, chosen_id, typed_text, idx FROM positions ORDER BY id"
    ).fetchall()
    return [tuple(r) for r in rows]


async def test_typed_word_rows():
    engine, _ = make_engine(_model(FIRST, FLAT))
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        assert ask.idx == 0 and ask.candidates
        await person.pick(ask, " Hello")
        ask = await person.wait_ask()
        assert ask.committed_text == " Hello"
        await person.type_word(ask, "there")
        ask = await person.wait_ask()
        assert ask.committed_text == " Hello there"
        await person.send(ask)
        completion = await handle.wait()
        streamed = await collect(handle)
        assert completion.text == " Hello there" == streamed
        assert completion.finish_reason == "stop"
        await engine.drain()
        rows = labels_log(engine)
        outcomes = [r[0] for r in rows]
        assert outcomes == ["pick", "typed", "typed", "typed", "pick"]
        assert [r[1] for r in rows[1:4]] == [1.0, 0.3, 0.3]  # typed word: first token then continuations
        assert all(r[3] == "there" for r in rows[1:4])
        assert rows[-1][2] == END
        assert engine.learners.n_labels == 5
        assert engine.learners.scorer.n_updates == 5
    finally:
        await engine.stop()


async def test_provisional_not_streamed():
    cfg = Config()
    cfg.gate.tau_sentence_start = 0.5
    cfg.gate.tau_content = 0.5
    cfg.gate.tau_function = 0.5
    engine, _ = make_engine(_model({HELLO: 0.9, THE: 0.05, END: 0.05}, FLAT), cfg=cfg)
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        auto = await person.wait_type("auto_run")
        assert [t.token_id for t in auto.tokens] == [HELLO]
        ask = await person.wait_ask()
        assert handle.stream.empty()  # the auto-selected token is still provisional
        assert [t.token_id for t in ask.provisional] == [HELLO]
        assert ask.committed_text == ""
        await person.pick(ask, " the")
        ask = await person.wait_ask()
        assert ask.committed_text == " Hello the"
        await person.send(ask)
        completion = await handle.wait()
        assert completion.text == " Hello the" == await collect(handle)
    finally:
        await engine.stop()


async def test_timeout_degrades():
    engine, backend = make_engine(_model(FIRST, FLAT))
    backend.fail_with = BackendUnavailable("down")
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        assert ask.candidates == [] and ask.other_mass == 1.0
        await person.type_word(ask, "hi", leading_space=False)
        ask = await person.wait_ask()
        assert ask.committed_text == "hi"
        await person.send(ask)
        completion = await handle.wait()
        assert completion.text == "hi" and completion.finish_reason == "stop"
        await engine.drain()
        assert [r[0] for r in labels_log(engine)] == ["typed", "typed", "pick"]
    finally:
        await engine.stop()


async def test_outcome_recording():
    cfg = Config()
    cfg.gate.tau_sentence_start = cfg.gate.tau_content = cfg.gate.tau_function = 0.5
    engine, _ = make_engine(_model({HELLO: 0.9, THE: 0.05, END: 0.05}, FLAT), cfg=cfg)
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        await person.wait_type("auto_run")
        ask = await person.wait_ask()
        await person.pick(ask, " the")
        ask = await person.wait_ask()
        await person.type_word(ask, "x", leading_space=True)
        ask = await person.wait_ask()
        await person.send(ask)
        await handle.wait()
        await engine.drain()
        outcomes = [r[0] for r in labels_log(engine)]
        assert outcomes[0] == "auto" and outcomes[1] == "pick"
        assert "typed" in outcomes and outcomes[-1] == "pick"
        assert engine.learners.n_labels == 4  # pick, two typed tokens, send; the auto row is not a label
        assert engine.learners.ngram.total_weight == pytest.approx(1.0 + 1.0 + 0.3 + 1.0 + 0.0)
    finally:
        await engine.stop()


async def test_explore_outcome():
    cfg = Config()
    cfg.gate.explore_rate = 1.0
    engine, _ = make_engine(_model(FLAT, FLAT), cfg=cfg)
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        assert ask.explore
        await person.pick(ask, " Hello")
        ask = await person.wait_ask()
        await person.type_word(ask, "there")
        ask = await person.wait_ask()
        await person.send(ask)
        await handle.wait()
        await engine.drain()
        outcomes = [r[0] for r in labels_log(engine)]
        assert outcomes[0] == "explore" and outcomes[-1] == "explore"
        # The first typed token sits at the explored position, so it is an explore row too.
        assert outcomes[1:4] == ["explore", "typed", "typed"]
    finally:
        await engine.stop()


async def test_cancelled_not_labeled():
    engine, _ = make_engine(_model(FIRST, FLAT))
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        await person.pick(ask, " Hello")
        ask = await person.wait_ask()
        await engine.cancel(handle.request_id)
        completion = await handle.wait()
        assert completion.finish_reason == "cancelled"
        await engine.drain()
        rows = engine.conn.execute(
            "SELECT COUNT(*) FROM positions WHERE request_id = ?", (handle.request_id,)
        ).fetchone()[0]
        assert rows == 1  # the pick row is kept
        labeled = [r for r in engine.log.iter_labeled() if r.request_id == handle.request_id]
        assert labeled == []
        assert engine.queue.active is None
    finally:
        await engine.stop()


async def test_max_tokens_length():
    engine, _ = make_engine(_model(FIRST, FLAT))
    await engine.start()
    try:
        handle = await engine.submit(request(max_tokens=2))
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        await person.pick(ask, " Hello")
        ask = await person.wait_ask()
        await person.pick(ask, " the")
        completion = await handle.wait()
        assert completion.finish_reason == "length"
        assert completion.text == " Hello the" == await collect(handle)
        assert completion.completion_tokens == 2
    finally:
        await engine.stop()


async def test_max_tokens_zero_finishes_empty():
    engine, _ = make_engine(_model(FIRST, FLAT))
    await engine.start()
    try:
        handle = await engine.submit(request(max_tokens=0))
        completion = await handle.wait()
        assert completion.finish_reason == "length" and completion.text == ""
    finally:
        await engine.stop()


async def test_stop_sequence_trim():
    engine, _ = make_engine(_model(FIRST, {BANG: 0.5, THE: 0.25, END: 0.25}))
    await engine.start()
    try:
        handle = await engine.submit(request(stop="!"))
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        await person.pick(ask, " Hello")
        ask = await person.wait_ask()
        await person.pick(ask, "!")
        completion = await handle.wait()
        assert completion.text == " Hello"
        assert completion.finish_reason == "stop"
        assert await collect(handle) == " Hello"
    finally:
        await engine.stop()


async def test_stale_pos_ignored():
    engine, _ = make_engine(_model(FIRST, FLAT))
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        await engine.handle_ui(Pick(request_id=ask.request_id, pos_id=ask.pos_id + 100, token_id=HELLO))
        await engine.handle_ui(Typed(request_id=ask.request_id, pos_id=ask.pos_id, text="x", leading_space=False, final=False))
        await asyncio.sleep(0)
        assert engine._driver.ask.pos_id == ask.pos_id  # still asking
        await person.pick(ask, " Hello")
        ask2 = await person.wait_ask()
        await engine.handle_ui(Pick(request_id=ask2.request_id, pos_id=ask.pos_id, token_id=THE))
        await engine.handle_ui(Pick(request_id=ask2.request_id, pos_id=ask2.pos_id, token_id=YES + 1000))
        await asyncio.sleep(0)
        assert engine._driver.ask.pos_id == ask2.pos_id
        await person.send(ask2)
        completion = await handle.wait()
        assert completion.text == " Hello"
    finally:
        await engine.stop()


async def test_decline_raises():
    engine, _ = make_engine(_model(FIRST, FLAT))
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        await person.wait_ask()
        await engine.handle_ui(Decline(request_id=handle.request_id, reason="busy"))
        with pytest.raises(DeclinedError) as info:
            await handle.wait()
        assert info.value.reason == "busy"
        await engine.drain()
        row = engine.conn.execute("SELECT status, finish_reason FROM requests WHERE id = ?", (handle.request_id,)).fetchone()
        assert tuple(row) == ("cancelled", "declined")
    finally:
        await engine.stop()


async def test_queue_full_and_pending_cancel():
    cfg = Config()
    cfg.server.max_pending = 1
    engine, _ = make_engine(_model(FIRST, FLAT), cfg=cfg)
    first = await engine.submit(request())
    with pytest.raises(QueueFull):
        await engine.submit(request())
    await engine.cancel(first.request_id)
    completion = await first.wait()
    assert completion.finish_reason == "cancelled"


def test_request_queue_limits_and_order():
    loop = asyncio.new_event_loop()
    try:
        async def scenario():
            queue = RequestQueue(max_pending=2, clock=lambda: 10.0)

            def make(i):
                return Request(
                    id=i, messages=[], params={}, tools=None, max_tokens=None, stop=[],
                    created_at=float(i), status=RequestStatus.PENDING,
                    handle=RequestHandle(i, on_cancel=lambda _: None),
                )

            queue.submit(make(1))
            queue.submit(make(2))
            with pytest.raises(QueueFull):
                queue.submit(make(3))
            first = await queue.next()
            assert first.id == 1 and first.status is RequestStatus.ACTIVE
            assert [r.id for r in queue.pending()] == [2]
            assert queue.ages() == {1: 9.0, 2: 8.0}
            queue.finish(1)
            assert queue.active is None

        loop.run_until_complete(scenario())
    finally:
        loop.close()


async def test_retrieval_once_per_request():
    engine, backend = make_engine(_model(FLAT, FLAT))
    recorded = []
    original = backend.next_token

    async def spy(prompt_ids, k):
        recorded.append(list(prompt_ids))
        return await original(prompt_ids, k)

    backend.next_token = spy
    engine.knowledge.add("The capital of France is Paris.", "note")
    messages = [{"role": "user", "content": "The capital of France is Paris."}]
    await engine.start()
    try:
        handle = await engine.submit(request(messages=messages))
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        await person.pick(ask, " Hello")
        ask = await person.wait_ask()
        await person.pick(ask, " the")
        ask = await person.wait_ask()
        await person.send(ask)
        await handle.wait()
        retrieval = engine.knowledge.retrieve("The capital of France is Paris.")
        assert retrieval.chunks
        expected = TOK.render_chat(messages, engine.knowledge.knowledge_text(retrieval.chunks))
        assert len(recorded) == 3
        for prompt in recorded:
            assert prompt[: len(expected)] == expected
    finally:
        await engine.stop()


async def test_knn_word_start_only():
    class Counting(HashEmbedder):
        def __init__(self):
            super().__init__(dim=32)
            self.calls = 0

        def embed(self, texts):
            self.calls += len(texts)
            return super().embed(texts)

    embedder = Counting()
    engine, _ = make_engine(_model(FIRST, FLAT), embedder=embedder)
    await engine.start()
    try:
        handle = await engine.submit(request())
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        await person.pick(ask, " Hello")  # next position is mid-word: no new embedding
        ask = await person.wait_ask()
        await person.pick(ask, " the")
        ask = await person.wait_ask()
        await person.send(ask)
        await handle.wait()
        # Only the first position (a sentence start) is embedded; retrieval is skipped with an empty store.
        assert embedder.calls == 1
    finally:
        await engine.stop()


async def test_sync_replays_state():
    engine, _ = make_engine(_model(FIRST, FLAT))
    await engine.start()
    try:
        await engine.submit(request())
        queued = await engine.submit(request())
        person = ScriptedPerson(engine)
        ask = await person.wait_ask()
        msgs = engine.sync_messages()
        kinds = [m.type for m in msgs]
        assert kinds[0] == "request_new" and kinds[-1] == "status"
        assert "request_active" in kinds and "ask" in kinds
        assert kinds.index("request_active") < kinds.index("ask") < kinds.index("status")
        assert msgs[kinds.index("ask")].pos_id == ask.pos_id
        assert queued.request_id != ask.request_id
        await engine.handle_ui(Sync())
    finally:
        await engine.stop()


async def test_send_first_empty_response():
    engine, _ = make_engine(_model(FIRST, FLAT))
    await engine.start()
    try:
        handle = await engine.submit(request())
        while engine._driver is None or engine._driver.active_msg is None:
            await asyncio.sleep(0)
        await engine.handle_ui(Send(request_id=handle.request_id))  # remembered, honoured at position 0
        completion = await handle.wait()
        assert completion.text == "" and completion.finish_reason == "stop"
        await engine.drain()
        assert labels_log(engine)[0][0] == "pick"
        assert labels_log(engine)[0][2] == END
    finally:
        await engine.stop()
