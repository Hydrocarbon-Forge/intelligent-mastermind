"""A scripted person for in-process engine tests: waits for asks and answers them through `Engine.handle_ui`.

Every server message the person has subscribed to is kept in a backlog until a `wait_*` call consumes it, so no
message is lost between calls.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from pydantic import BaseModel

from humanapi.api.schemas import Ask, Pick, Rewind, Send, Typed
from humanapi.core.engine import Engine

_MARKERS = {"␣": " ", "↵": "\n", "⇥": "\t"}  # ␣ ↵ ⇥


def _strip_markers(display: str) -> str:
    return "".join(_MARKERS.get(ch, ch) for ch in display)


class ScriptedPerson:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._queue = engine.subscribe()
        self._backlog: list[BaseModel] = []

    async def _next_matching(self, predicate, timeout: float) -> BaseModel:
        for index, msg in enumerate(self._backlog):
            if predicate(msg):
                return self._backlog.pop(index)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError("no matching server message")
            msg = await asyncio.wait_for(self._queue.get(), remaining)
            if predicate(msg):
                return msg
            self._backlog.append(msg)

    async def wait_ask(self, timeout: float = 2.0) -> Ask:
        return await self._next_matching(lambda m: isinstance(m, Ask), timeout)  # type: ignore[return-value]

    async def wait_type(self, type_: str, timeout: float = 2.0) -> BaseModel:
        return await self._next_matching(lambda m: getattr(m, "type", None) == type_, timeout)

    async def pick(self, ask: Ask, token_text: str) -> None:
        for cand in ask.candidates:
            if _strip_markers(cand.display) == token_text:
                await self.engine.handle_ui(Pick(request_id=ask.request_id, pos_id=ask.pos_id, token_id=cand.id))
                return
        raise AssertionError(f"no candidate with text {token_text!r} in ask {ask.pos_id}")

    async def type_word(self, ask: Ask, text: str, leading_space: Optional[bool] = None) -> None:
        ls = ask.space_default if leading_space is None else leading_space
        await self.engine.handle_ui(
            Typed(request_id=ask.request_id, pos_id=ask.pos_id, text=text, leading_space=ls, final=True)
        )

    async def send(self, ask: Ask) -> None:
        await self.engine.handle_ui(Send(request_id=ask.request_id))

    async def rewind(self, request_id: int, to_idx: int) -> None:
        await self.engine.handle_ui(Rewind(request_id=request_id, to_idx=to_idx))
