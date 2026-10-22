"""Request-level types: status, the handle the API client waits on, and the queued request record."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class RequestStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"


class QueueFull(Exception):
    """More requests are waiting or active than server.max_pending allows."""


class DeclinedError(Exception):
    def __init__(self, reason: str = "") -> None:
        super().__init__(reason or "declined")
        self.reason = reason


@dataclass
class Completion:
    text: str
    finish_reason: str  # "stop" | "length" | "cancelled"
    prompt_tokens: int
    completion_tokens: int


class RequestHandle:
    """What the API client holds: committed text deltas on `stream` and the final Completion via `wait()`."""

    def __init__(self, request_id: int, on_cancel: Callable[[int], None]) -> None:
        self.request_id = request_id
        self.stream: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._future: asyncio.Future[Completion] = asyncio.get_running_loop().create_future()
        self._on_cancel = on_cancel

    async def wait(self) -> Completion:
        return await self._future

    def cancel(self) -> None:
        self._on_cancel(self.request_id)

    # ---- resolution (used by the engine; each call is a no-op once resolved)

    def resolve(self, completion: Completion) -> None:
        if not self._future.done():
            self._future.set_result(completion)

    def fail(self, error: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(error)

    def close_stream(self) -> None:
        self.stream.put_nowait(None)


@dataclass
class Request:
    id: int
    messages: list[dict]
    params: dict
    tools: Optional[list[dict[str, Any]]]
    max_tokens: Optional[int]
    stop: list[str]
    created_at: float  # engine clock (monotonic seconds)
    status: RequestStatus
    handle: RequestHandle = field(repr=False)
