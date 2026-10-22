"""Integration suite: the real components on the runtime machine (llama-server on port 8081, the GPU embedder,
the Qwen tokenizer files). Collected here but never run on the development VM: select with `pytest -m integration`.

Every test builds the real components from config.yaml, or from the file named by HUMANAPI_CONFIG.
Assertions follow docs/spec/eval.md ("Integration suite").
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path

import pytest

from humanapi.config import load_config

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
LATENCY_BUDGET_MS = 300.0


def _cfg():
    return load_config(os.environ.get("HUMANAPI_CONFIG") or None)


def _tokenizer(cfg):
    from humanapi.text.tokenizer import HumanTokenizer

    return HumanTokenizer(cfg.tokenizer.path, cfg.tokenizer.enable_thinking)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


async def test_probe_matches_server():
    from humanapi.backend.base import Capabilities
    from humanapi.backend.openai_compat import OpenAICompatBackend, load_capabilities

    cfg = _cfg()
    caps_path = _resolve(cfg.backend.capabilities_file)
    assert caps_path.exists(), "run `python -m tools.probe_backend` first"
    probed = load_capabilities(str(caps_path))
    assert isinstance(probed, Capabilities)
    backend = OpenAICompatBackend(cfg.backend.base_url, cfg.backend.model, _tokenizer(cfg), probed,
                                  timeout_s=cfg.backend.timeout_s)
    try:
        assert await backend.health()
        result = await backend.next_token(_tokenizer(cfg).encode("Hello"), cfg.backend.top_k)
        assert result.k <= probed.max_top_k
        assert len(result.top) <= result.k
    finally:
        close = getattr(backend, "aclose", None)
        if close is not None:
            await close()


async def test_probs_match_probe():
    from humanapi.backend.openai_compat import OpenAICompatBackend, load_capabilities

    cfg = _cfg()
    probed = load_capabilities(str(_resolve(cfg.backend.capabilities_file)))
    tok = _tokenizer(cfg)
    backend = OpenAICompatBackend(cfg.backend.base_url, cfg.backend.model, tok, probed, timeout_s=cfg.backend.timeout_s)
    try:
        result = await backend.next_token(tok.encode("The quick brown fox"), cfg.backend.top_k)
        total = sum(math.exp(entry.logprob) for entry in result.top)
        if probed.filtered_probs:
            # Filtered by the server's top_p / top_k: the returned probabilities are renormalised.
            assert total == pytest.approx(1.0, abs=1e-3)
        else:
            assert total + result.tail_mass == pytest.approx(1.0, abs=1e-3)
    finally:
        close = getattr(backend, "aclose", None)
        if close is not None:
            await close()


async def test_ids_roundtrip_real_server():
    cfg = _cfg()
    tok = _tokenizer(cfg)
    text = "Sounds good, see you at noon!"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    # The prompt the server receives decodes to the text the local tokenizer produced.
    prompt_ids = tok.render_chat([{"role": "user", "content": text}], None)
    assert text in tok.decode(prompt_ids)


async def test_vocab_padding_ids_dropped():
    from humanapi.learn.ngram import NgramMemory
    from humanapi.learn.candidates import assemble
    from humanapi.learn.types import PositionContext
    from humanapi.text.types import PosState
    from humanapi.config import LearningConfig

    cfg = _cfg()
    tok = _tokenizer(cfg)
    real = set(tok.valid_ids())
    padding = [i for i in range(tok.vocab_size) if i not in real]
    assert padding, "the tokenizer should report padding ids above its string table"
    ngram = NgramMemory()
    for pad in padding[:5]:
        ngram.update([], pad)  # a padding id learned by mistake must still never be offered
    ctx = PositionContext(response_ids=[], state=PosState.SENTENCE_START)
    candidates = assemble(ctx, tok=tok, ngram=ngram, knn=None, cfg=LearningConfig())
    assert not {c.token_id for c in candidates} & set(padding[:5])


async def test_end_to_end_sdk(tmp_path):
    """The OpenAI client gets a reply produced by a scripted person through the WebSocket (real components)."""
    import httpx
    import openai
    import uvicorn
    import websockets

    from humanapi.main import create_app

    cfg = _cfg()
    cfg.storage.db_path = str(tmp_path / "e2e.db")
    app = create_app(cfg, static_dir=str(tmp_path))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(1000):
            if server.started:
                break
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:

            async def person():
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "ask":
                        await ws.send(json.dumps({"type": "send", "request_id": msg["request_id"]}))
                        return

            person_task = asyncio.create_task(person())
            sdk = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="sk-test")
            completion = await asyncio.wait_for(
                sdk.chat.completions.create(model=cfg.server.model_id, messages=[{"role": "user", "content": "hi"}]),
                120,
            )
            await asyncio.wait_for(person_task, 120)
            assert completion.choices[0].finish_reason in ("stop", "length")
            await sdk.close()
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 30)


async def test_latency_recorded(tmp_path):
    """Commit-to-list latency: from the person's pick to the next ask on the socket. Written next to the budget."""
    import uvicorn
    import websockets

    from humanapi.main import create_app

    cfg = _cfg()
    cfg.storage.db_path = str(tmp_path / "latency.db")
    app = create_app(cfg, static_dir=str(tmp_path))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(1000):
            if server.started:
                break
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        samples: list[float] = []
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            import httpx

            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=120) as http:
                request = asyncio.create_task(http.post(
                    "/v1/chat/completions",
                    json={"model": cfg.server.model_id, "messages": [{"role": "user", "content": "hi"}]},
                ))
                ask = None
                while ask is None:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 120))
                    if msg.get("type") == "ask":
                        ask = msg
                for _ in range(3):
                    top = ask["candidates"][0]
                    started = time.perf_counter()
                    await ws.send(json.dumps({"type": "pick", "request_id": ask["request_id"],
                                               "pos_id": ask["pos_id"], "token_id": top["id"]}))
                    while True:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), 120))
                        if msg.get("type") == "ask":
                            samples.append((time.perf_counter() - started) * 1000.0)
                            ask = msg
                            break
                await ws.send(json.dumps({"type": "send", "request_id": ask["request_id"]}))
                await asyncio.wait_for(request, 120)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 30)

    samples.sort()
    median = samples[len(samples) // 2]
    out = _resolve("data/latency.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"commit_to_list_ms": samples, "median_ms": median, "budget_ms": LATENCY_BUDGET_MS}, indent=2),
        encoding="utf-8",
    )
    assert samples and all(s >= 0 for s in samples)


async def test_embedder_dimension():
    from humanapi.learn.embed import SentenceTransformerEmbedder

    cfg = _cfg()
    embedder = SentenceTransformerEmbedder(cfg.embed.model, cfg.embed.device)
    vectors = embedder.embed(["a quick check of the embedder"])
    assert vectors.shape == (1, 1024)
    assert embedder.dim == 1024
