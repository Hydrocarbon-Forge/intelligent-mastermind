import asyncio
import json
import random
import re
from pathlib import Path

import httpx
import openai
import pytest
from fastapi import FastAPI

import humanapi.api.openai_routes as routes
from humanapi.api.openai_routes import build_openai_router
from humanapi.api.schemas import Decline
from humanapi.backend.base import Capabilities
from humanapi.backend.fake import FakeBackend
from humanapi.config import Config
from humanapi.core.engine import Engine
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
FIRST = {HELLO: 0.4, THE: 0.3, YES: 0.2, END: 0.1}
FLAT = {HELLO: 0.3, THE: 0.3, YES: 0.3, END: 0.1}
GOLDEN = Path(__file__).resolve().parents[1] / "data" / "stream_golden.txt"
MESSAGES = [{"role": "user", "content": "hi"}]
CREATED = 1760000000.0


def model(ctx):
    """FIRST right after the prompt; after " Hello" a '!' is likely (for the stop-sequence test)."""
    last = ctx[-1]
    if last == NL:
        return FIRST
    if last == HELLO:
        return {BANG: 0.5, THE: 0.25, END: 0.25}
    return FLAT


def make_engine(cfg):
    cfg.gate.cold_start_min_labels = 0
    cfg.gate.explore_rate = 0.0
    cfg.gate.lookahead_max = 0
    backend = FakeBackend(model, capabilities=Capabilities(), token_bytes=lambda i: TOK.token_bytes(i) or b"")
    return Engine.create(
        cfg,
        backend=backend,
        tokenizer=TOK,
        embedder=HashEmbedder(dim=32),
        rng=random.Random(0),
        status_interval_s=1000.0,
    )


def build_app(engine, cfg, heartbeat=0.05):
    app = FastAPI()
    app.include_router(
        build_openai_router(
            engine,
            cfg,
            heartbeat_s=heartbeat,
            id_factory=lambda: "chatcmpl-golden",
            clock=lambda: CREATED,
        )
    )
    return app


async def start(cfg=None, heartbeat=0.05):
    cfg = cfg if cfg is not None else Config()
    engine = make_engine(cfg)
    await engine.start()
    app = build_app(engine, cfg, heartbeat)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return engine, app, client


async def stop(engine, client):
    await client.aclose()
    await engine.stop()


def answer_task(engine, person, steps, delay=0.0):
    """Answer each ask in turn: steps are ("pick", text), ("send",) or ("decline",)."""

    async def run():
        for step in steps:
            ask = await person.wait_ask()
            if delay:
                await asyncio.sleep(delay)
            if step[0] == "pick":
                await person.pick(ask, step[1])
            elif step[0] == "send":
                await person.send(ask)
            else:
                await engine.handle_ui(Decline(request_id=ask.request_id, reason="busy"))

    return asyncio.create_task(run())


def chat_body(**extra):
    body = {"model": "human", "messages": MESSAGES}
    body.update(extra)
    return body


async def test_models_endpoint():
    engine, _, client = await start()
    try:
        response = await client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "human"
    finally:
        await stop(engine, client)


async def test_stream_golden():
    engine, _, client = await start(heartbeat=0.05)
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("pick", " Hello"), ("pick", " the"), ("send",)], delay=0.2)
        response = await client.post("/v1/chat/completions", json=chat_body(stream=True))
        await script
        text = response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        normalized = re.sub(r"(: keepalive\n\n)+", ": keepalive\n\n", text)
        assert normalized == GOLDEN.read_text(encoding="utf-8")
    finally:
        await stop(engine, client)


async def test_sdk_streaming():
    engine, _, client = await start()
    sdk = openai.AsyncOpenAI(base_url="http://test/v1", api_key="sk-test", http_client=client)
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("pick", " Hello"), ("send",)])
        stream = await sdk.chat.completions.create(model="human", messages=MESSAGES, stream=True)
        parts, finish = [], None
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
        await script
        assert "".join(parts) == " Hello"
        assert finish == "stop"
    finally:
        await sdk.close()
        await stop(engine, client)


async def test_sdk_non_streaming():
    engine, _, client = await start()
    sdk = openai.AsyncOpenAI(base_url="http://test/v1", api_key="sk-test", http_client=client)
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("pick", " Hello"), ("send",)])
        completion = await sdk.chat.completions.create(model="human", messages=MESSAGES)
        await script
        choice = completion.choices[0]
        assert choice.message.content == " Hello"
        assert choice.finish_reason == "stop"
        assert completion.usage.completion_tokens == 1
        assert completion.usage.total_tokens == completion.usage.prompt_tokens + 1
    finally:
        await sdk.close()
        await stop(engine, client)


async def test_n_greater_than_one():
    engine, _, client = await start()
    try:
        response = await client.post("/v1/chat/completions", json=chat_body(n=2))
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error" and error["param"] == "n"
    finally:
        await stop(engine, client)


async def test_params_ignored_but_logged():
    engine, _, client = await start()
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("send",)])
        response = await client.post(
            "/v1/chat/completions", json=chat_body(temperature=0.9, top_p=0.5, seed=3, custom_field="x")
        )
        await script
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == ""
        await engine.drain()
        params = json.loads(engine.conn.execute("SELECT params_json FROM requests").fetchone()[0])
        assert params["temperature"] == 0.9 and params["top_p"] == 0.5 and params["seed"] == 3
        assert params["custom_field"] == "x"
    finally:
        await stop(engine, client)


async def test_stop_sequence_trim():
    engine, _, client = await start()
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("pick", " Hello"), ("pick", "!")])
        response = await client.post("/v1/chat/completions", json=chat_body(stop="!"))
        await script
        choice = response.json()["choices"][0]
        assert choice["message"]["content"] == " Hello"
        assert choice["finish_reason"] == "stop"
    finally:
        await stop(engine, client)


async def test_max_tokens_length():
    engine, _, client = await start()
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("pick", " Hello"), ("pick", " the")])
        response = await client.post("/v1/chat/completions", json=chat_body(max_tokens=2))
        await script
        choice = response.json()["choices"][0]
        assert choice["message"]["content"] == " Hello the"
        assert choice["finish_reason"] == "length"
    finally:
        await stop(engine, client)


async def test_disconnect_cancels(monkeypatch):
    monkeypatch.setattr(routes, "_DISCONNECT_POLL_S", 0.01)
    engine, app, client = await start()
    try:
        body = json.dumps(chat_body()).encode()
        messages = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            if messages:
                return messages.pop(0)
            return {"type": "http.disconnect"}

        async def send(_message):
            return None

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 5555),
            "server": ("test", 80),
        }
        await app(scope, receive, send)
        await asyncio.sleep(0.05)
        await engine.drain()
        status = engine.conn.execute("SELECT status FROM requests ORDER BY id DESC LIMIT 1").fetchone()[0]
        assert status == "cancelled"
    finally:
        await stop(engine, client)


async def test_max_pending_503():
    cfg = Config()
    cfg.server.max_pending = 1
    engine, _, client = await start(cfg)
    try:
        person = ScriptedPerson(engine)
        first = asyncio.create_task(client.post("/v1/chat/completions", json=chat_body()))
        for _ in range(100):
            if engine.queue.active is not None or engine.queue.pending():
                break
            await asyncio.sleep(0.01)
        second = await client.post("/v1/chat/completions", json=chat_body())
        assert second.status_code == 503
        error = second.json()["error"]
        assert error["type"] == "server_error" and error["code"] == "queue_full"
        script = answer_task(engine, person, [("send",)])
        assert (await first).status_code == 200
        await script
    finally:
        await stop(engine, client)


async def test_tools_shown_not_called():
    engine, _, client = await start()
    try:
        person = ScriptedPerson(engine)
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        script = answer_task(engine, person, [("pick", " Hello"), ("send",)])
        response = await client.post("/v1/chat/completions", json=chat_body(tools=tools))
        await script
        request_new = await person.wait_type("request_new")
        assert request_new.tools == tools
        choice = response.json()["choices"][0]
        assert choice["message"]["content"] == " Hello"
        assert "tool_calls" not in response.text
    finally:
        await stop(engine, client)


async def test_no_reasoning_fields():
    engine, _, client = await start()
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("pick", " Hello"), ("send",)])
        streamed = await client.post("/v1/chat/completions", json=chat_body(stream=True))
        await script
        regular_script = answer_task(engine, person, [("send",)])
        regular = await client.post("/v1/chat/completions", json=chat_body())
        await regular_script
        for text in (streamed.text, regular.text):
            assert "reasoning_content" not in text and "tool_calls" not in text
    finally:
        await stop(engine, client)


async def test_bearer_token():
    cfg = Config()
    cfg.server.bearer_token = "s3cret"
    engine, _, client = await start(cfg)
    try:
        assert (await client.get("/v1/models")).status_code == 401
        wrong = await client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        assert wrong.json()["error"]["code"] == "invalid_api_key"
        assert (await client.get("/v1/models", headers={"Authorization": "Bearer s3cret"})).status_code == 200
        assert (await client.get("/health")).status_code == 200
    finally:
        await stop(engine, client)


async def test_decline_returns_error():
    engine, _, client = await start()
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("decline",)])
        response = await client.post("/v1/chat/completions", json=chat_body())
        await script
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "request_declined"
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("decline",)])
        streamed = await client.post("/v1/chat/completions", json=chat_body(stream=True))
        await script
        assert '"code":"request_declined"' in streamed.text
        assert streamed.text.endswith("data: [DONE]\n\n")
    finally:
        await stop(engine, client)


async def test_zero_deltas():
    engine, _, client = await start()
    try:
        person = ScriptedPerson(engine)
        script = answer_task(engine, person, [("send",)])
        streamed = await client.post("/v1/chat/completions", json=chat_body(stream=True))
        await script
        assert '"finish_reason":"stop"' in streamed.text and streamed.text.endswith("data: [DONE]\n\n")
        assert streamed.text.count('"content":"') == 1  # only the empty role chunk
    finally:
        await stop(engine, client)
