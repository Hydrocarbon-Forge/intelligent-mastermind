"""End to end on one process: a real uvicorn server, the WebSocket UI driven by a scripted person, and the official
OpenAI client calling the HTTP API. Fake backend and tokenizer; every wait is bounded."""
import asyncio
import json
import random

import openai
import uvicorn
import websockets

from humanapi.backend.base import Capabilities
from humanapi.backend.fake import FakeBackend
from humanapi.config import Config
from humanapi.learn.embed import HashEmbedder
from humanapi.main import Deps, create_app
from humanapi.text.toy_tokenizer import ToyTokenizer

WAIT = 10.0
TOK = ToyTokenizer()
END = TOK.end_id
HELLO = TOK.encode(" Hello")[0]
THE = TOK.encode(" the")[0]
YES = TOK.encode(" yes")[0]
FLAT = {HELLO: 0.3, THE: 0.3, YES: 0.3, END: 0.1}


def _model(ctx):
    return FLAT


def _app(tmp_path):
    cfg = Config()
    cfg.gate.cold_start_min_labels = 0
    cfg.gate.explore_rate = 0.0
    cfg.gate.lookahead_max = 0
    # Every position is asked, so the scripted person's answers line up with the asks (no auto-selection).
    cfg.gate.max_auto_run = 0
    backend = FakeBackend(_model, capabilities=Capabilities(), token_bytes=lambda i: TOK.token_bytes(i) or b"")
    deps = Deps(backend=backend, tokenizer=TOK, embedder=HashEmbedder(dim=32))
    return create_app(cfg, deps=deps, static_dir=str(tmp_path))


async def _serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on"))
    task = asyncio.create_task(server.serve())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAIT
    while not server.started:
        if task.done():
            task.result()  # surfaces a startup failure
        if loop.time() > deadline:
            raise TimeoutError("uvicorn did not start")
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port


def _stripped(display):
    return display.replace("␣", " ").replace("↵", "\n").replace("⇥", "\t")


def _reply(ask, step):
    """The WebSocket message the scripted person sends for one ask."""
    kind = step[0]
    rid, pid = ask["request_id"], ask["pos_id"]
    if kind == "pick":
        cand = next(c for c in ask["candidates"] if _stripped(c["display"]) == step[1])
        return {"type": "pick", "request_id": rid, "pos_id": pid, "token_id": cand["id"]}
    if kind == "typed":
        return {"type": "typed", "request_id": rid, "pos_id": pid, "text": step[1],
                "leading_space": ask["space_default"], "final": True}
    return {"type": "send", "request_id": rid}


async def _person(ws, steps):
    """Answers each ask with the next scripted step, in order across requests."""
    steps = list(steps)
    async for raw in ws:
        msg = json.loads(raw)
        if msg.get("type") != "ask":
            continue
        await ws.send(json.dumps(_reply(msg, steps.pop(0))))
        if not steps:
            return


async def test_e2e_sdk_streaming_and_regular(tmp_path):
    app = _app(tmp_path)
    server, task, port = await _serve(app)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", open_timeout=WAIT) as ws:
            # Request 1 (regular): pick " Hello", type "there", send. Request 2 (stream): pick " the", send.
            steps = [("pick", " Hello"), ("typed", "there"), ("send",), ("pick", " the"), ("send",)]
            person = asyncio.create_task(_person(ws, steps))
            sdk = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="sk-test", max_retries=0)
            try:
                completion = await asyncio.wait_for(
                    sdk.chat.completions.create(model="human", messages=[{"role": "user", "content": "lunch?"}]),
                    WAIT,
                )
                assert completion.choices[0].message.content == " Hello there"
                assert completion.choices[0].finish_reason == "stop"
                assert completion.usage is not None and completion.usage.completion_tokens > 0

                stream = await asyncio.wait_for(
                    sdk.chat.completions.create(
                        model="human",
                        messages=[{"role": "user", "content": "and then?"}],
                        stream=True,
                        stream_options={"include_usage": True},
                    ),
                    WAIT,
                )

                async def collect():
                    parts, finish, usage = [], None, None
                    async for chunk in stream:
                        if chunk.usage is not None:
                            usage = chunk.usage
                        for choice in chunk.choices:
                            if choice.delta.content:
                                parts.append(choice.delta.content)
                            if choice.finish_reason:
                                finish = choice.finish_reason
                    return "".join(parts), finish, usage

                text, finish, usage = await asyncio.wait_for(collect(), WAIT)
                assert text == " the"
                assert finish == "stop"
                assert usage is not None and usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
                await asyncio.wait_for(person, WAIT)
            finally:
                await sdk.close()
                if not person.done():
                    person.cancel()
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, WAIT)
