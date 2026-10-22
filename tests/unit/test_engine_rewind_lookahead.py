import random

import numpy as np
import pytest

from humanapi.api.schemas import ChatCompletionRequest, Rewind
from humanapi.backend.base import Capabilities
from humanapi.backend.fake import FakeBackend
from humanapi.config import Config
from humanapi.core.engine import Engine
from humanapi.learn.embed import HashEmbedder
from humanapi.learn.types import GateGroup
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
CHAIN = {HELLO: THE, THE: YES, YES: HELLO}
TEACH = [HELLO, THE, YES, HELLO, THE, YES]


def chain_model(ctx):
    """Near-deterministic chain: after each token the next chain token has probability 0.97."""
    last = ctx[-1]
    nxt = HELLO if last == NL else CHAIN.get(last, HELLO)
    return {nxt: 0.97, END: 0.01, BANG: 0.02}


def make_engine(*, max_auto_run=40, prompt_logprobs=True, lookahead_max=8, seed=0):
    cfg = Config()
    cfg.gate.cold_start_min_labels = 0
    cfg.gate.explore_rate = 0.0
    cfg.gate.max_auto_run = max_auto_run
    cfg.gate.lookahead_max = lookahead_max
    # Teaching asks: thresholds above the chain's confidence, so every position is asked.
    cfg.gate.tau_content = cfg.gate.tau_function = cfg.gate.tau_sentence_start = 0.999
    backend = FakeBackend(
        chain_model,
        capabilities=Capabilities(prompt_logprobs=prompt_logprobs),
        token_bytes=lambda i: TOK.token_bytes(i) or b"",
    )
    engine = Engine.create(
        cfg,
        backend=backend,
        tokenizer=TOK,
        embedder=HashEmbedder(dim=32),
        rng=random.Random(seed),
        status_interval_s=1000.0,
    )
    return engine, backend


def lower_thresholds(engine):
    """Auto-selection on: the gate reads these values at each position."""
    engine.gate.cfg.tau_content = 0.5
    engine.gate.cfg.tau_function = 0.5
    engine.gate.cfg.tau_sentence_start = 0.5


async def teach(engine, person, n=6):
    """Pick the first n chain tokens so the n-gram memory learns the chain. Returns the open ask.

    max_auto_run is 0 while teaching, so every position is asked: the scorer learns the chain quickly and
    would otherwise auto-select it, which is not what the teaching step needs."""
    engine.gate.cfg.max_auto_run = 0
    for token in TEACH[:n]:
        ask = await person.wait_ask()
        await person.pick(ask, TOK.token_text(token))
    return await person.wait_ask()


async def autorun(engine, person, max_auto_run=3):
    """Teach, then auto-select after the next pick until the forced ask. Returns that ask."""
    ask = await teach(engine, person)
    engine.gate.cfg.max_auto_run = max_auto_run
    lower_thresholds(engine)
    await person.pick(ask, TOK.token_text(HELLO))
    return await person.wait_ask()


def labels_rows(engine):
    return [
        tuple(r)
        for r in engine.conn.execute(
            "SELECT outcome, idx, chosen_id, rejected_id, sample_weight, group_name, top_conf "
            "FROM positions ORDER BY id"
        ).fetchall()
    ]


def count_calls(backend, name):
    return sum(n for kind, n in backend.calls if kind == name)


async def collect_stream(handle):
    parts = []
    while not handle.stream.empty():
        item = handle.stream.get_nowait()
        if item is None:
            break
        parts.append(item)
    return "".join(parts)


async def test_rewind_in_buffer():
    engine, _ = make_engine(max_auto_run=3)
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        forced = await autorun(engine, person, max_auto_run=3)
        assert forced.idx == 10 and len(forced.provisional) == 3
        rejected = forced.provisional[1]
        assert rejected.idx == 8
        await collect_stream(handle)  # nothing pending from earlier flushes
        await person.rewind(handle.request_id, 8)
        rewound = await person.wait_type("rewound")
        assert rewound.to_idx == 8
        ask = await person.wait_ask()
        assert ask.idx == 8 and ask.pos_id != forced.pos_id and not ask.explore
        assert [t.idx for t in ask.provisional] == [7]
        assert handle.stream.empty()  # the rewind streamed nothing
        await person.send(ask)
        completion = await handle.wait()
        assert completion.finish_reason == "stop"
        await engine.drain()
        rows = labels_rows(engine)
        rewound_rows = [r for r in rows if r[0] == "rewound"]
        assert len(rewound_rows) == 1
        assert rewound_rows[0][3] == rejected.token_id and rewound_rows[0][2] is None
        assert rewound_rows[0][1] == 8
    finally:
        await engine.stop()


async def test_rewind_controller():
    engine, _ = make_engine(max_auto_run=3)
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        await autorun(engine, person, max_auto_run=3)
        base_tau = engine.gate.thresholds.tau(GateGroup.CONTENT)
        assert engine.gate.controller.boost == 0.0
        await person.rewind(handle.request_id, 8)
        ask = await person.wait_ask()
        await person.rewind(handle.request_id, 7)  # a second rewind, while the new ask is open
        ask = await person.wait_ask()
        assert engine.gate.controller.boost > 0
        boosted = engine.gate.thresholds.tau(GateGroup.CONTENT, boost=engine.gate.controller.boost)
        assert boosted > base_tau
        await person.send(ask)
        await handle.wait()
    finally:
        await engine.stop()


async def test_rewind_clamped_to_flushed_and_ignored_after_finish():
    engine, _ = make_engine(max_auto_run=3)
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        await autorun(engine, person, max_auto_run=3)
        await person.rewind(handle.request_id, 2)  # below the flushed length 7: clamped to 7
        ask = await person.wait_ask()
        assert ask.idx == 7 and ask.provisional == []
        await person.send(ask)
        completion = await handle.wait()
        assert completion.text.endswith(" Hello")
        await engine.handle_ui(Rewind(request_id=handle.request_id, to_idx=0))  # finished: ignored
    finally:
        await engine.stop()


async def test_max_auto_run():
    engine, _ = make_engine(max_auto_run=3)
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        forced = await autorun(engine, person, max_auto_run=3)
        assert forced.idx == 10 and len(forced.provisional) == 3
        assert not forced.explore
        await person.send(forced)
        await handle.wait()
        await engine.drain()
        last = engine.conn.execute(
            "SELECT outcome, group_name, top_conf FROM positions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        # The forced ask is a normal label but not a calibration observation.
        assert last[0] == "pick" and last[1] is None and last[2] is None
    finally:
        await engine.stop()


async def test_rewound_row_and_replacement_label():
    engine, _ = make_engine(max_auto_run=3)
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        forced = await autorun(engine, person, max_auto_run=3)
        await engine.drain()
        before = engine.learners.scorer.snapshot()
        updates_before = engine.learners.scorer.n_updates
        rejected = forced.provisional[0]
        await person.rewind(handle.request_id, rejected.idx)
        ask = await person.wait_ask()
        replacement = next(c for c in ask.candidates if c.id == THE)
        await person.pick(ask, TOK.token_text(replacement.id))
        await person.send(await person.wait_ask())
        await handle.wait()
        await engine.drain()
        rows = engine.conn.execute(
            "SELECT outcome, chosen_id, rejected_id, sample_weight FROM positions WHERE outcome = 'rewound'"
        ).fetchall()
        assert len(rows) == 1 and rows[0][1] is None and rows[0][2] == rejected.token_id
        assert rows[0][3] == 1.0
        # The rewound row is a penalty, not a label; the replacement pick is a normal label.
        labeled = [r.outcome.value for r in engine.log.iter_labeled() if r.request_id == handle.request_id]
        assert "rewound" not in labeled
        assert labeled[-2] == "pick"
        assert engine.learners.scorer.n_updates == updates_before + 3  # penalty, replacement, send
        assert not np.array_equal(engine.learners.scorer.snapshot(), before)
    finally:
        await engine.stop()


async def test_lookahead_draft_from_ngram():
    engine, _ = make_engine()
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        await teach(engine, person)
        await engine.drain()
        ngram = engine.learners.ngram
        draft = ngram.draft([HELLO], 8, 0.5)
        assert draft == [THE, YES, HELLO, THE, YES, HELLO, THE, YES]
        # An unseen context has no confident answer, so nothing is drafted.
        assert ngram.draft([BANG], 8, 0.5) == []
        await engine.cancel(handle.request_id)
        await handle.wait()
    finally:
        await engine.stop()


async def test_lookahead_stops_and_discards_cache():
    engine, backend = make_engine(max_auto_run=2)
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        forced = await autorun(engine, person, max_auto_run=2)
        assert forced.idx == 9
        # The forced ask clears the drafted continuation.
        assert engine._driver._lookahead == {}
        assert count_calls(backend, "score_continuation") >= 1
        # The person answers off the chain: the next position is scored from the backend, not the cache.
        before = count_calls(backend, "next_token")
        await person.pick(forced, TOK.token_text(BANG))
        await person.send(await person.wait_ask())
        await handle.wait()
        assert count_calls(backend, "next_token") > before
    finally:
        await engine.stop()


async def test_lookahead_disabled():
    engine, backend = make_engine(max_auto_run=3, prompt_logprobs=False)
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        forced = await autorun(engine, person, max_auto_run=3)
        assert forced.idx == 10
        assert count_calls(backend, "score_continuation") == 0
        # Every position costs one backend call: 6 teach asks, the pick, 3 auto tokens and the forced ask.
        assert count_calls(backend, "next_token") == 11
        await person.send(forced)
        await handle.wait()
    finally:
        await engine.stop()


async def _scripted_session(*, lookahead_max):
    engine, backend = make_engine(max_auto_run=4, lookahead_max=lookahead_max, seed=7)
    await engine.start()
    try:
        handle = await engine.submit(ChatCompletionRequest(messages=USER))
        person = ScriptedPerson(engine)
        forced = await autorun(engine, person, max_auto_run=4)
        top = max((c for c in forced.candidates if c.id != END), key=lambda c: c.p_you)
        await person.pick(forced, TOK.token_text(top.id))  # the chain's next token
        forced = await person.wait_ask()
        await person.send(forced)
        completion = await handle.wait()
        await engine.drain()
        rows = engine.conn.execute(
            "SELECT outcome, idx, chosen_id, rejected_id, sample_weight, group_name, top_conf, tau, candidates "
            "FROM positions ORDER BY id"
        ).fetchall()
        return [tuple(r) for r in rows], completion.text, count_calls(backend, "next_token")
    finally:
        await engine.stop()


async def test_lookahead_rows_identical_and_fewer_calls():
    rows_on, text_on, next_on = await _scripted_session(lookahead_max=8)
    rows_off, text_off, next_off = await _scripted_session(lookahead_max=0)
    assert rows_on == rows_off
    assert text_on == text_off
    assert next_on < next_off
