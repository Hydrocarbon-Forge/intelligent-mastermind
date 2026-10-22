"""OpenAI-compatible routes: models, chat completions (regular and SSE), and the health check.

The shim owns parsing, framing, keepalives, disconnects and OpenAI-format errors. It makes no decisions about
tokens: each request goes to `Engine.submit`, and the engine's text deltas and completion are framed here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from humanapi.api.schemas import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    ErrorBody,
    ErrorResponse,
    ModelCard,
    ModelList,
    Usage,
)
from humanapi.config import Config
from humanapi.core.engine import Engine
from humanapi.core.session import Completion, DeclinedError, QueueFull

log = logging.getLogger(__name__)

NON_STREAM_WARN_S: float = 60.0
_DISCONNECT_POLL_S: float = 0.5  # how often a non-streaming wait checks for a client disconnect
_NO_CONTENT_STATUS = 499  # nginx's "client closed request"; the body is never read by a client that left


def openai_error(
    status: int,
    message: str,
    type_: str,
    *,
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(message=message, type=type_, param=param, code=code))
    return JSONResponse(status_code=status, content=body.model_dump())


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n\n"


def _bearer_ok(request: Request, token: Optional[str]) -> bool:
    if not token:
        return True
    expected = f"Bearer {token}"
    supplied = request.headers.get("authorization")
    return supplied is not None and supplied == expected


def build_openai_router(
    engine: Engine,
    cfg: Config,
    *,
    heartbeat_s: Optional[float] = None,
    id_factory: Optional[Callable[[], str]] = None,
    clock: Optional[Callable[[], float]] = None,
) -> APIRouter:
    router = APIRouter()
    heartbeat = float(cfg.ui.heartbeat_seconds if heartbeat_s is None else heartbeat_s)
    new_id = id_factory if id_factory is not None else (lambda: "chatcmpl-" + uuid.uuid4().hex)
    now = clock if clock is not None else time.time
    model_id = cfg.server.model_id
    token = cfg.server.bearer_token

    def unauthorized(request: Request) -> Optional[JSONResponse]:
        if _bearer_ok(request, token):
            return None
        return openai_error(
            401, "Incorrect API key provided.", "invalid_request_error", code="invalid_api_key"
        )

    @router.get("/v1/models")
    async def list_models(request: Request):
        denied = unauthorized(request)
        if denied is not None:
            return denied
        return ModelList(data=[ModelCard(id=model_id)])

    @router.get("/health")
    async def health():
        status = engine.status_message()
        try:
            engine.log.conn.execute("SELECT 1").fetchone()
            database = "ok"
        except Exception:
            database = "error"
        queue = engine.queue
        return {
            "process": "ok",
            "database": database,
            "backend": bool(status.backend_ok),
            "capabilities": status.capabilities,
            "queue": {
                "pending": len(queue.pending()),
                "active": queue.active.id if queue.active is not None else None,
            },
        }

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        denied = unauthorized(request)
        if denied is not None:
            return denied
        try:
            raw = await request.json()
        except Exception:
            return openai_error(400, "Invalid JSON in request body.", "invalid_request_error")
        try:
            req = ChatCompletionRequest.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            where = ".".join(str(part) for part in first.get("loc", ())) or "body"
            return openai_error(
                400, f"Invalid request: {where}: {first.get('msg', 'invalid value')}", "invalid_request_error"
            )
        if not req.messages:
            return openai_error(
                400, "'messages' must contain at least one message.", "invalid_request_error", param="messages"
            )
        if req.n > 1:
            return openai_error(
                400, "Only n=1 is supported: one person gives one answer.", "invalid_request_error", param="n"
            )

        host = request.client.host if request.client is not None else "unknown"
        agent = request.headers.get("user-agent", "")
        try:
            handle = await engine.submit(req, client_info=f"{host} {agent}".strip())
        except QueueFull:
            return openai_error(
                503, "The server is busy: too many requests are waiting.", "server_error", code="queue_full"
            )

        created_id = new_id()
        created = int(now())
        if req.stream:
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return StreamingResponse(
                _event_stream(handle, created_id, created, model_id, include_usage, heartbeat),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return await _regular(request, handle, created_id, created, model_id)

    async def _regular(request: Request, handle, created_id: str, created: int, model: str):
        waiter = asyncio.ensure_future(handle.wait())
        started = time.monotonic()
        warned = False
        try:
            while True:
                done, _ = await asyncio.wait({waiter}, timeout=_DISCONNECT_POLL_S)
                if waiter in done:
                    break
                if not warned and time.monotonic() - started > NON_STREAM_WARN_S:
                    warned = True
                    log.warning(
                        "non-streaming request %s has waited %.0f s; the client may time out",
                        handle.request_id,
                        time.monotonic() - started,
                    )
                if await request.is_disconnected():
                    handle.cancel()
                    waiter.cancel()
                    return Response(status_code=_NO_CONTENT_STATUS)
            completion: Completion = waiter.result()
        except DeclinedError:
            return openai_error(
                503, "The person declined this request.", "server_error", code="request_declined"
            )
        except RuntimeError:
            return openai_error(500, "The request failed inside the server.", "server_error")
        finally:
            if not waiter.done():
                waiter.cancel()
        body = ChatCompletion(
            id=created_id,
            created=created,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=AssistantMessage(content=completion.text),
                    finish_reason=completion.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                total_tokens=completion.prompt_tokens + completion.completion_tokens,
            ),
        )
        return JSONResponse(content=body.model_dump())

    return router


async def _event_stream(
    handle,
    created_id: str,
    created: int,
    model: str,
    include_usage: bool,
    heartbeat_s: float,
) -> AsyncIterator[str]:
    def chunk(delta: dict[str, Any], finish: Optional[str] = None) -> dict[str, Any]:
        return {
            "id": created_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    finished = False
    try:
        yield _sse(chunk({"role": "assistant", "content": ""}))
        while True:
            try:
                item = await asyncio.wait_for(handle.stream.get(), timeout=heartbeat_s)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            yield _sse(chunk({"content": item}))

        try:
            completion: Completion = await handle.wait()
        except DeclinedError:
            finished = True
            yield _sse(
                {
                    "error": {
                        "message": "The person declined this request.",
                        "type": "server_error",
                        "param": None,
                        "code": "request_declined",
                    }
                }
            )
            yield "data: [DONE]\n\n"
            return
        except RuntimeError:
            finished = True
            yield _sse({"error": {"message": "The request failed inside the server.", "type": "server_error",
                                  "param": None, "code": None}})
            yield "data: [DONE]\n\n"
            return

        yield _sse(chunk({}, completion.finish_reason))
        if include_usage:
            total = completion.prompt_tokens + completion.completion_tokens
            yield _sse(
                {
                    "id": created_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": completion.prompt_tokens,
                        "completion_tokens": completion.completion_tokens,
                        "total_tokens": total,
                    },
                }
            )
        finished = True
        yield "data: [DONE]\n\n"
    finally:
        if not finished:
            # The client went away (generator closed or cancelled) before the request finished.
            handle.cancel()
