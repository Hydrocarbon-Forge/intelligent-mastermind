"""The engine: request submission and queue, the worker that runs one driver at a time, the ordered write queue,
the UI event hub and the status message.

Every learner update and every log write goes through `_writes`, a single FIFO drained by one task, so rows follow
creation order and `Learners.apply` runs exactly where the rebuild would run it.
"""
from __future__ import annotations

import asyncio
import logging
import random
import sqlite3
import time
from dataclasses import asdict
from typing import Any, Callable, Optional

from pydantic import BaseModel

from humanapi.api.schemas import (
    Ask as AskMessage,
    ChatCompletionRequest,
    Decline,
    Forget,
    NoteInfo,
    Pick,
    RequestDone,
    RequestNew,
    Rewind,
    Send,
    SetAutonomy,
    Status,
    Sync,
    Teach,
    Typed,
)
from humanapi.backend.base import InferenceBackend
from humanapi.config import Config
from humanapi.core.driver import Cancel, Driver
from humanapi.core.queue import RequestQueue
from humanapi.core.session import (
    Completion,
    DeclinedError,
    QueueFull,
    Request,
    RequestHandle,
    RequestStatus,
)
from humanapi.knowledge.store import KnowledgeStore
from humanapi.learn.embed import Embedder
from humanapi.learn.gate import Gate
from humanapi.learn.learners import Learners
from humanapi.store import db
from humanapi.store.log import InteractionLog
from humanapi.store.rebuild import rebuild
from humanapi.text.types import Tokenizer

_NOTE_PREVIEW_CHARS = 80
log = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        cfg: Config,
        *,
        backend: InferenceBackend,
        tokenizer: Tokenizer,
        embedder: Embedder,
        conn: sqlite3.Connection,
        log: InteractionLog,
        learners: Learners,
        gate: Gate,
        knowledge: KnowledgeStore,
        clock: Callable[[], float] = time.monotonic,
        status_interval_s: float = 5.0,
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.conn = conn
        self.log = log
        self.learners = learners
        self.gate = gate
        self.knowledge = knowledge
        self.clock = clock
        self.status_interval_s = float(status_interval_s)
        self.queue = RequestQueue(cfg.server.max_pending, clock=clock)
        self.stats: dict[str, int] = {"auto": 0, "asks": 0}
        self._driver: Optional[Driver] = None
        self._subscribers: list[asyncio.Queue] = []
        self._writes: asyncio.Queue[Callable[[], None]] = asyncio.Queue()
        self._pos_seq = 0
        self._backend_ok = False
        self._tasks: set[asyncio.Task] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._writer_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

    @classmethod
    def create(
        cls,
        cfg: Config,
        *,
        backend: InferenceBackend,
        tokenizer: Tokenizer,
        embedder: Embedder,
        conn: Optional[sqlite3.Connection] = None,
        rng: Optional[random.Random] = None,
        clock: Callable[[], float] = time.monotonic,
        status_interval_s: float = 5.0,
    ) -> "Engine":
        conn = conn if conn is not None else db.connect(":memory:")
        db.migrate(conn)
        log = InteractionLog(conn)
        learners = rebuild(log, cfg, embed_dim=embedder.dim, vocab_size=tokenizer.vocab_size)
        gate = Gate(cfg.gate, calibrator=learners.calibrator, rng=rng)
        knowledge = KnowledgeStore(conn, embedder, tokenizer, cfg.knowledge)
        return cls(
            cfg,
            backend=backend,
            tokenizer=tokenizer,
            embedder=embedder,
            conn=conn,
            log=log,
            learners=learners,
            gate=gate,
            knowledge=knowledge,
            clock=clock,
            status_interval_s=status_interval_s,
        )

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self._writer_task = asyncio.get_running_loop().create_task(self._writer())
        self._worker_task = asyncio.get_running_loop().create_task(self._worker())
        self._health_task = asyncio.get_running_loop().create_task(self._health())

    async def stop(self) -> None:
        await self.drain()
        tasks = [t for t in (self._health_task, self._worker_task, self._writer_task) if t is not None]
        tasks.extend(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._health_task = self._worker_task = self._writer_task = None

    async def drain(self) -> None:
        """Wait until every queued learner update and log write has run."""
        if self._writer_task is not None:
            await self._writes.join()
            return
        while not self._writes.empty():
            self._run_write(self._writes.get_nowait())

    # ------------------------------------------------------------ background loops

    async def _writer(self) -> None:
        while True:
            job = await self._writes.get()
            try:
                self._run_write(job)
            finally:
                self._writes.task_done()

    def _run_write(self, job: Callable[[], None]) -> None:
        try:
            job()
        except Exception:  # a failed write must not stop the other writes
            log.exception("write job failed")

    async def _worker(self) -> None:
        while True:
            req = await self.queue.next()
            driver = Driver(self, req)
            self._driver = driver
            try:
                await driver.run()
            finally:
                self._driver = None
                self.queue.finish(req.id)

    async def _health(self) -> None:
        while True:
            try:
                self._backend_ok = bool(await self.backend.health())
            except Exception:
                self._backend_ok = False
            self.broadcast(self.status_message())
            await asyncio.sleep(self.status_interval_s)

    # ------------------------------------------------------------ internals used by the driver

    def enqueue_write(self, job: Callable[[], None]) -> None:
        self._writes.put_nowait(job)

    def next_pos_id(self) -> int:
        self._pos_seq += 1
        return self._pos_seq

    def broadcast(self, msg: BaseModel) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(msg)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------ submission

    async def submit(self, req: ChatCompletionRequest, *, client_info: Optional[str] = None) -> RequestHandle:
        messages = [m.model_dump() for m in req.messages]
        params = req.model_dump(exclude={"messages"})
        request_id = self.log.create_request(client_info=client_info, messages=messages, params=params)
        max_tokens = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
        if req.stop is None:
            stop: list[str] = []
        elif isinstance(req.stop, str):
            stop = [req.stop]
        else:
            stop = list(req.stop)
        handle = RequestHandle(request_id, on_cancel=self._schedule_cancel)
        request = Request(
            id=request_id,
            messages=messages,
            params=params,
            tools=req.tools,
            max_tokens=max_tokens,
            stop=stop,
            created_at=self.clock(),
            status=RequestStatus.PENDING,
            handle=handle,
        )
        try:
            self.queue.submit(request)
        except QueueFull:
            self.log.finish_request(
                request_id, status="cancelled", finish_reason="queue_full", final_text=None, final_ids=None
            )
            raise
        self.broadcast(RequestNew(request_id=request_id, messages=messages, params=params, tools=req.tools))
        return handle

    def _schedule_cancel(self, request_id: int) -> None:
        self._spawn(self.cancel(request_id))

    async def cancel(self, request_id: int) -> None:
        """The client went away."""
        pending = self.queue.remove(request_id)
        if pending is not None:
            self._end_pending(pending, reason="cancelled", status="cancelled")
            return
        driver = self._driver
        if driver is not None and driver.req.id == request_id:
            driver.post(Cancel())
            await driver.finished.wait()

    def _end_pending(self, req: Request, *, reason: str, status: str) -> None:
        req.status = RequestStatus.CANCELLED
        request_id = req.id
        self.enqueue_write(
            lambda: self.log.finish_request(
                request_id, status=status, finish_reason=reason, final_text=None, final_ids=None
            )
        )
        req.handle.close_stream()
        if reason == "declined":
            req.handle.fail(DeclinedError(reason))
        else:
            req.handle.resolve(Completion(text="", finish_reason=reason, prompt_tokens=0, completion_tokens=0))
        self.broadcast(RequestDone(request_id=request_id, finish_reason=reason))

    # ------------------------------------------------------------ UI messages

    async def handle_ui(self, msg: BaseModel) -> None:
        if isinstance(msg, Sync):
            return
        if isinstance(msg, (Pick, Typed, Send, Rewind)):
            driver = self._driver
            if driver is not None and driver.req.id == msg.request_id:
                driver.post(msg)
            return
        if isinstance(msg, Decline):
            driver = self._driver
            if driver is not None and driver.req.id == msg.request_id:
                driver.post(msg)
                return
            pending = self._remove_pending(msg.request_id)
            if pending is not None:
                self._end_pending(pending, reason="declined", status="cancelled")
            return
        if isinstance(msg, SetAutonomy):
            self.gate.autonomy = float(msg.delta)
            self.broadcast(self.status_message())
            return
        if isinstance(msg, Teach):
            await asyncio.to_thread(self.knowledge.add, msg.text, "teach")
            self.broadcast(self.status_message())
            return
        if isinstance(msg, Forget):
            self.knowledge.forget(int(msg.chunk_id))
            self.broadcast(self.status_message())
            return

    def _remove_pending(self, request_id: int) -> Optional[Request]:
        return self.queue.remove(request_id)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    # ------------------------------------------------------------ status and sync

    def status_message(self) -> Status:
        autos = self.stats["auto"]
        asks = self.stats["asks"]
        total = autos + asks
        metrics = {
            "auto_select_rate": autos / total if total else 0.0,
            "asks_per_100": 100.0 * asks / total if total else 0.0,
            "rewind_rate": 0.0,
            "pick_accuracy": float(self.learners.scorer.metrics()["pick_accuracy"]),
        }
        notes = [
            NoteInfo(id=c.id, source=c.source, preview=c.text[:_NOTE_PREVIEW_CHARS])
            for c in self.knowledge.notes()
        ]
        return Status(
            backend_ok=self._backend_ok,
            capabilities=asdict(self.backend.capabilities),
            scorer_version=int(self.learners.scorer.n_updates),
            autonomy=float(self.gate.autonomy),
            metrics=metrics,
            knowledge_count=self.knowledge.count(),
            notes=notes,
        )

    def sync_messages(self) -> list[BaseModel]:
        """What a (re)connected UI needs: queued and active requests, the active request, the open ask, status."""
        ages = self.queue.ages()
        out: list[BaseModel] = []
        requests = ([self.queue.active] if self.queue.active is not None else []) + self.queue.pending()
        for req in requests:
            out.append(
                RequestNew(
                    request_id=req.id,
                    messages=req.messages,
                    params=req.params,
                    tools=req.tools,
                    age_s=float(ages.get(req.id, 0.0)),
                )
            )
        driver = self._driver
        if driver is not None:
            if driver.active_msg is not None:
                out.append(driver.active_msg)
            ask: Optional[AskMessage] = driver.ask_message()
            if ask is not None:
                out.append(ask)
        out.append(self.status_message())
        return out
