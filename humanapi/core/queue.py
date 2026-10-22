"""FIFO of requests: one active at a time, a bounded number waiting or active, ages for the UI."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Callable, Optional

from humanapi.core.session import QueueFull, Request, RequestStatus


class RequestQueue:
    def __init__(self, max_pending: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_pending = int(max_pending)
        self._clock = clock
        self._pending: deque[Request] = deque()
        self._wakeup = asyncio.Event()
        self.active: Optional[Request] = None

    def submit(self, req: Request) -> None:
        occupied = len(self._pending) + (1 if self.active is not None else 0)
        if occupied + 1 > self.max_pending:
            raise QueueFull(f"{occupied} requests waiting or active (max_pending={self.max_pending})")
        self._pending.append(req)
        self._wakeup.set()

    async def next(self) -> Request:
        while not self._pending:
            self._wakeup.clear()
            await self._wakeup.wait()
        req = self._pending.popleft()
        req.status = RequestStatus.ACTIVE
        self.active = req
        return req

    def remove(self, request_id: int) -> Optional[Request]:
        for pos, req in enumerate(self._pending):
            if req.id == request_id:
                del self._pending[pos]
                return req
        return None

    def finish(self, request_id: int) -> None:
        if self.active is not None and self.active.id == request_id:
            self.active = None

    def pending(self) -> list[Request]:
        return list(self._pending)

    def ages(self) -> dict[int, float]:
        now = self._clock()
        out = {req.id: now - req.created_at for req in self._pending}
        if self.active is not None:
            out[self.active.id] = now - self.active.created_at
        return out
