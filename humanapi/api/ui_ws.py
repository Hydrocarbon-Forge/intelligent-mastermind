"""UI WebSocket endpoint at /ws.

The router only forwards. Engine broadcasts go to the socket as JSON; UI frames are parsed into the protocol
models and passed to Engine.handle_ui; a `sync` frame replays the current state to this socket only. The engine
decides everything (stale and duplicate messages are ignored there), so no request state lives here.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket
from pydantic import TypeAdapter, ValidationError

from humanapi.api.schemas import ClientMessage, Sync
from humanapi.config import Config
from humanapi.core.engine import Engine

log = logging.getLogger(__name__)

_CLIENT_MESSAGE = TypeAdapter(ClientMessage)
_MAX_FRAME_CHARS = 1_000_000  # larger frames are garbage for this protocol and are ignored


async def _sender(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        msg = await queue.get()
        await websocket.send_json(msg.model_dump(mode="json"))


async def _replay(websocket: WebSocket, engine: Engine) -> None:
    for msg in engine.sync_messages():
        await websocket.send_json(msg.model_dump(mode="json"))


async def _receiver(websocket: WebSocket, engine: Engine) -> None:
    while True:
        frame: dict[str, Any] = await websocket.receive()
        if frame["type"] == "websocket.disconnect":
            return
        text = frame.get("text")
        if text is None:
            log.warning("ignoring binary UI frame")
            continue
        if len(text) > _MAX_FRAME_CHARS:
            log.warning("ignoring oversized UI frame (%d chars)", len(text))
            continue
        try:
            data = json.loads(text)
        except ValueError:
            log.warning("ignoring UI frame that is not JSON")
            continue
        try:
            msg = _CLIENT_MESSAGE.validate_python(data)
        except ValidationError as exc:
            log.warning("ignoring invalid UI message: %s", exc.errors(include_url=False)[:1])
            continue
        if isinstance(msg, Sync):
            await _replay(websocket, engine)
            continue
        try:
            await engine.handle_ui(msg)
        except Exception:  # one bad message must not drop the connection
            log.exception("handling UI message failed")


def build_ui_router(engine: Engine, cfg: Config) -> APIRouter:
    """The router for /ws. `cfg` is accepted for the app wiring; the endpoint reads its limits from the engine."""
    router = APIRouter()

    @router.websocket("/ws")
    async def ui_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = engine.subscribe()
        try:
            await websocket.send_json(engine.status_message().model_dump(mode="json"))
            sender = asyncio.ensure_future(_sender(websocket, queue))
            receiver = asyncio.ensure_future(_receiver(websocket, engine))
            done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    log.info("UI connection ended: %r", exc)
        finally:
            engine.unsubscribe(queue)

    return router
