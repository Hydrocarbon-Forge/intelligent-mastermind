"""UI WebSocket endpoint: protocol tests over the ASGI websocket interface, on the same event loop as the engine.

Starlette's TestClient would run the app in another thread and loop, so the socket is driven directly: frames go
into one queue as websocket.receive messages and the endpoint's websocket.send messages come back on another.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Optional

import pytest
from fastapi import FastAPI

from humanapi.api.schemas import ChatCompletionRequest
from humanapi.api.ui_ws import build_ui_router
from humanapi.backend.fake import FakeBackend
from humanapi.config import Config
from humanapi.core.engine import Engine
from humanapi.core.session import DeclinedError
from humanapi.learn.embed import HashEmbedder
from humanapi.text.toy_tokenizer import ToyTokenizer

TIMEOUT = 3.0
TOK = ToyTokenizer()
END = TOK.end_id
HELLO = TOK.encode(" Hello")[0]
THE = TOK.encode(" the")[0]
YES = TOK.encode(" yes")[0]
USER = [{"role": "user", "content": "hi"}]
FLAT = {HELLO: 0.3, THE: 0.3, YES: 0.2, END: 0.1}  # never confident enough to auto-select


def flat_model(ctx):
    return FLAT


class Socket:
    """One UI connection: `open` performs the websocket.connect handshake, `send` and `recv` move JSON frames."""

    def __init__(self, app: FastAPI) -> None:
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._outbox: asyncio.Queue = asyncio.Queue()
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "scheme": "ws",
            "path": "/ws",
            "raw_path": b"/ws",
            "root_path": "",
            "query_string": b"",
            "headers": [],
            "client": ("test", 0),
            "server": ("test", 80),
            "subprotocols": [],
        }
        self.task = asyncio.ensure_future(app(scope, self._receive, self._send))

    async def _receive(self) -> dict:
        return await self._inbox.get()

    async def _send(self, message: dict) -> None:
        await self._outbox.put(message)

    async def open(self) -> None:
        await self._inbox.put({"type": "websocket.connect"})
        first = await self._raw()
        assert first["type"] == "websocket.accept", first

    async def _raw(self) -> dict:
        return await asyncio.wait_for(self._outbox.get(), TIMEOUT)

    async def recv(self) -> dict:
        msg = await self._raw()
        assert msg["type"] == "websocket.send", msg
        return json.loads(msg["text"])

    async def recv_type(self, type_: str) -> dict:
        """Receive frames until one has the wanted type; earlier frames are skipped."""
        while True:
            msg = await self.recv()
            if msg["type"] == type_:
                return msg

    async def collect_until(self, type_: str) -> list[dict]:
        out: list[dict] = []
        while True:
            msg = await self.recv()
            out.append(msg)
            if msg["type"] == type_:
                return out

    async def send(self, payload: Any) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        await self._inbox.put({"type": "websocket.receive", "text": text})

    async def send_bytes(self, data: bytes) -> None:
        await self._inbox.put({"type": "websocket.receive", "bytes": data})

    async def quiet(self, seconds: float = 0.1) -> bool:
        """True when the server sent nothing for `seconds`."""
        try:
            msg = await asyncio.wait_for(self._outbox.get(), seconds)
        except asyncio.TimeoutError:
            return True
        self._outbox.put_nowait(msg)
        return False

    async def close(self) -> None:
        await self._inbox.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(self.task, TIMEOUT)


def make_engine() -> tuple[Engine, FakeBackend]:
    cfg = Config()
    cfg.gate.cold_start_min_labels = 0
    backend = FakeBackend(flat_model, token_bytes=lambda i: TOK.token_bytes(i) or b"")
    engine = Engine.create(
        cfg,
        backend=backend,
        tokenizer=TOK,
        embedder=HashEmbedder(dim=32),
        rng=random.Random(0),
        status_interval_s=1000.0,  # no periodic status during a test
    )
    return engine, backend


def make_app(engine: Engine) -> FastAPI:
    app = FastAPI()
    app.include_router(build_ui_router(engine, engine.cfg))
    return app


async def start_engine(engine: Engine) -> None:
    await engine.start()
    await asyncio.sleep(0)  # let the health loop send its first status before any UI subscribes


def request(content: str = "hi") -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[{"role": "user", "content": content}])


def token_id_for(ask: dict, display_text: str) -> int:
    for cand in ask["candidates"]:
        if cand["display"] == display_text:
            return cand["id"]
    raise AssertionError(f"no candidate {display_text!r} in {ask['candidates']}")


async def stop(engine: Engine, *sockets: Socket) -> None:
    for sock in sockets:
        if not sock.task.done():
            await sock.close()
    await engine.stop()


async def test_ws_happy_path():
    engine, _ = make_engine()
    await start_engine(engine)
    sock = Socket(make_app(engine))
    try:
        await sock.open()
        assert (await sock.recv())["type"] == "status"
        handle = await engine.submit(request())
        rid = handle.request_id

        first = await sock.recv_type("ask")
        assert first["request_id"] == rid
        assert {"pos_id", "idx", "committed_text", "candidates", "space_default", "other_mass"} <= set(first)
        await sock.send({"type": "pick", "request_id": rid, "pos_id": first["pos_id"],
                         "token_id": token_id_for(first, "␣Hello")})

        second = await sock.recv_type("ask")
        assert second["pos_id"] > first["pos_id"]
        await sock.send({"type": "typed", "request_id": rid, "pos_id": second["pos_id"],
                         "text": "there", "leading_space": second["space_default"], "final": True})

        third = await sock.recv_type("ask")
        await sock.send({"type": "send", "request_id": rid})
        done = await sock.recv_type("request_done")
        assert done["finish_reason"] == "stop"
        assert done["request_id"] == rid
        completion = await asyncio.wait_for(handle.wait(), TIMEOUT)
        assert completion.text == " Hello there"
        assert third["request_id"] == rid
    finally:
        await stop(engine, sock)


async def test_ws_stale_pos_ignored():
    engine, _ = make_engine()
    await start_engine(engine)
    sock = Socket(make_app(engine))
    try:
        await sock.open()
        await sock.recv()  # status
        handle = await engine.submit(request())
        rid = handle.request_id

        first = await sock.recv_type("ask")
        await sock.send({"type": "pick", "request_id": rid, "pos_id": first["pos_id"],
                         "token_id": token_id_for(first, "␣Hello")})
        second = await sock.recv_type("ask")
        assert second["pos_id"] != first["pos_id"]

        # A late duplicate of the answer to the first ask: must not change the committed text or the open ask.
        await sock.send({"type": "pick", "request_id": rid, "pos_id": first["pos_id"],
                         "token_id": token_id_for(first, "␣Hello")})
        await sock.send({"type": "pick", "request_id": rid, "pos_id": first["pos_id"],
                         "token_id": THE})
        assert await sock.quiet(0.1)

        await sock.send({"type": "send", "request_id": rid})
        await sock.recv_type("request_done")
        completion = await asyncio.wait_for(handle.wait(), TIMEOUT)
        assert completion.text == " Hello"
    finally:
        await stop(engine, sock)


async def test_ws_decline_errors_client():
    engine, _ = make_engine()
    await start_engine(engine)
    sock = Socket(make_app(engine))
    try:
        await sock.open()
        await sock.recv()  # status
        active = await engine.submit(request("first"))
        pending = await engine.submit(request("second"))
        await sock.recv_type("ask")

        # The pending request is declined from the queue; the active one through its driver.
        await sock.send({"type": "decline", "request_id": pending.request_id, "reason": "no thanks"})
        await sock.send({"type": "decline", "request_id": active.request_id, "reason": "no thanks"})

        with pytest.raises(DeclinedError):
            await asyncio.wait_for(pending.wait(), TIMEOUT)
        with pytest.raises(DeclinedError):
            await asyncio.wait_for(active.wait(), TIMEOUT)
    finally:
        await stop(engine, sock)


async def test_ws_sync_replays_state():
    engine, _ = make_engine()
    await start_engine(engine)
    app = make_app(engine)
    sock_a = Socket(app)
    sock_b: Optional[Socket] = None
    try:
        await sock_a.open()
        await sock_a.recv()  # status
        active = await engine.submit(request("first"))
        queued = await engine.submit(request("second"))
        ask_a = await sock_a.recv_type("ask")
        assert ask_a["request_id"] == active.request_id

        # A UI that reconnects mid-request recovers its view from sync alone.
        sock_b = Socket(app)
        await sock_b.open()
        assert (await sock_b.recv())["type"] == "status"
        await sock_b.send({"type": "sync"})
        replay = await sock_b.collect_until("status")

        types = [m["type"] for m in replay]
        assert types == ["request_new", "request_new", "request_active", "ask", "status"], types
        assert [m["request_id"] for m in replay[:2]] == [active.request_id, queued.request_id]
        ask_b = replay[3]
        assert ask_b["pos_id"] == ask_a["pos_id"]
        assert ask_b["request_id"] == active.request_id
    finally:
        if sock_b is not None:
            await stop(engine, sock_b)
        await stop(engine, sock_a)


async def test_ws_sync_with_no_requests_returns_status_only():
    engine, _ = make_engine()
    await start_engine(engine)
    sock = Socket(make_app(engine))
    try:
        await sock.open()
        await sock.recv()  # status on connect
        await sock.send({"type": "sync"})
        replay = await sock.collect_until("status")
        assert [m["type"] for m in replay] == ["status"]
    finally:
        await stop(engine, sock)


async def test_ws_garbage_frames_keep_socket_open():
    engine, _ = make_engine()
    await start_engine(engine)
    sock = Socket(make_app(engine))
    try:
        await sock.open()
        await sock.recv()  # status
        await sock.send("not json {")
        await sock.send("[1, 2]")
        await sock.send('{"type": "no_such_type"}')
        await sock.send('{"type": "pick"}')  # missing fields
        await sock.send('{"request_id": 1}')  # missing type
        await sock.send_bytes(b"\x00\xff\x10")
        await sock.send("x" * 1_100_000)  # oversized garbage
        assert await sock.quiet(0.1)
        assert not sock.task.done()

        await sock.send({"type": "sync"})
        replay = await sock.collect_until("status")
        assert replay[-1]["type"] == "status"
    finally:
        await stop(engine, sock)


async def test_ws_broadcasts_to_every_ui():
    engine, _ = make_engine()
    await start_engine(engine)
    app = make_app(engine)
    sock_a, sock_b = Socket(app), Socket(app)
    try:
        await sock_a.open()
        await sock_b.open()
        await sock_a.recv()
        await sock_b.recv()
        handle = await engine.submit(request())
        for sock in (sock_a, sock_b):
            msg = await sock.recv_type("request_new")
            assert msg["request_id"] == handle.request_id
    finally:
        await stop(engine, sock_a, sock_b)


async def test_ws_disconnect_keeps_request_running():
    engine, _ = make_engine()
    await start_engine(engine)
    app = make_app(engine)
    sock = Socket(app)
    await sock.open()
    await sock.recv()
    handle = await engine.submit(request())
    await sock.recv_type("ask")
    await sock.close()
    assert engine._subscribers == []  # noqa: SLF001 - the endpoint must unsubscribe on disconnect

    waiter = asyncio.ensure_future(handle.wait())
    done, _ = await asyncio.wait({waiter}, timeout=0.1)  # wait without cancelling: the request must still be running
    assert not done

    # A reconnecting UI finds the request still active and answers it.
    again = Socket(app)
    try:
        await again.open()
        await again.recv()
        await again.send({"type": "sync"})
        replay = await again.collect_until("status")
        ask = next(m for m in replay if m["type"] == "ask")
        assert ask["request_id"] == handle.request_id
        await again.send({"type": "pick", "request_id": handle.request_id, "pos_id": ask["pos_id"],
                          "token_id": token_id_for(ask, "␣Hello")})
        await again.send({"type": "send", "request_id": handle.request_id})
        completion = await asyncio.wait_for(waiter, TIMEOUT)
        assert completion.text == " Hello"
    finally:
        await stop(engine, again)
