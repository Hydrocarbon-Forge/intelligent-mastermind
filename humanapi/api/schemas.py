"""Pydantic models: OpenAI payloads (shim side) and the UI WebSocket protocol.

The UI protocol is mirrored one-to-one by ui/src/protocol.ts. Every UI message is a JSON object with a
`type` field. Messages that belong to a request carry `request_id`.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------- OpenAI request


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    text: Optional[str] = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[str, list[ContentPart], None] = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """Unknown fields are accepted and logged. Sampling fields are recorded and ignored."""

    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    n: int = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Union[str, list[str], None] = None
    tools: Optional[list[dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------- OpenAI response


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str  # content only: never reasoning_content, never tool_calls


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Optional[Usage] = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "human"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI error format: {"error": {"message", "type", "param", "code"}}."""

    error: ErrorBody


# ---------------------------------------------------------------- UI protocol: shared parts


class AskCandidate(BaseModel):
    id: int
    display: str  # token text with visible whitespace markers
    p_you: float
    p_base: Optional[float] = None
    cls: str  # a TokenClass value


class AutoToken(BaseModel):
    idx: int  # token index within the response
    token_id: int
    display: str
    confidence: float  # calibrated confidence at the time of the auto-selection


class NoteInfo(BaseModel):
    id: int
    source: str
    preview: str  # first ~80 characters of the chunk


# ---------------------------------------------------------------- UI protocol: server -> UI


class RequestNew(BaseModel):
    type: Literal["request_new"] = "request_new"
    request_id: int
    messages: list[dict[str, Any]]  # the client's original messages
    params: dict[str, Any]  # temperature, top_p, ... as sent (logged, ignored)
    tools: Optional[list[dict[str, Any]]] = None  # shown for reference only
    age_s: float = 0.0


class RequestActive(BaseModel):
    type: Literal["request_active"] = "request_active"
    request_id: int
    prompt_tail: str  # the last part of the rendered prompt, for reference
    max_tokens: Optional[int] = None
    knowledge: list[str] = Field(default_factory=list)  # retrieved chunks, for reference


class Ask(BaseModel):
    type: Literal["ask"] = "ask"
    request_id: int
    pos_id: int  # unique per asked position; every answer must echo it
    idx: int  # token index being decided (flushed tokens + provisional tokens)
    committed_text: str  # text flushed so far (up to the last human decision)
    provisional: list[AutoToken] = Field(default_factory=list)  # auto-selected, not yet flushed
    space_default: bool  # proposed leading_space for a typed word
    candidates: list[AskCandidate]  # sorted by p_you descending, at most ui.max_rows
    other_mass: float
    explore: bool = False
    tokens_remaining: Optional[int] = None  # remaining max_tokens budget


class AutoRun(BaseModel):
    type: Literal["auto_run"] = "auto_run"
    request_id: int
    tokens: list[AutoToken]  # newly auto-selected provisional tokens


class Rewound(BaseModel):
    type: Literal["rewound"] = "rewound"
    request_id: int
    to_idx: int


class RequestDone(BaseModel):
    type: Literal["request_done"] = "request_done"
    request_id: int
    finish_reason: str  # stop | length | cancelled | declined | error


class Status(BaseModel):
    type: Literal["status"] = "status"
    request_id: Optional[int] = None
    backend_ok: bool
    capabilities: dict[str, Any]
    scorer_version: int
    autonomy: float  # slider delta in logit space
    metrics: dict[str, float]  # auto_select_rate, asks_per_100, rewind_rate, pick_accuracy, ...
    knowledge_count: int
    notes: list[NoteInfo] = Field(default_factory=list)


# ---------------------------------------------------------------- UI protocol: UI -> server


class Sync(BaseModel):
    type: Literal["sync"] = "sync"
    request_id: Optional[int] = None


class Pick(BaseModel):
    type: Literal["pick"] = "pick"
    request_id: int
    pos_id: int
    token_id: int


class Typed(BaseModel):
    type: Literal["typed"] = "typed"
    request_id: int
    pos_id: int
    text: str
    leading_space: bool
    final: bool  # False: a preview while typing; the server ignores previews in v1


class Rewind(BaseModel):
    type: Literal["rewind"] = "rewind"
    request_id: int
    to_idx: int


class Send(BaseModel):
    type: Literal["send"] = "send"
    request_id: int


class Decline(BaseModel):
    type: Literal["decline"] = "decline"
    request_id: int
    reason: str = ""


class SetAutonomy(BaseModel):
    type: Literal["set_autonomy"] = "set_autonomy"
    request_id: Optional[int] = None
    delta: float  # the absolute slider setting in logit space (not an increment)


class Teach(BaseModel):
    type: Literal["teach"] = "teach"
    request_id: Optional[int] = None
    text: str


class Forget(BaseModel):
    type: Literal["forget"] = "forget"
    request_id: Optional[int] = None
    chunk_id: int


ServerMessage = Annotated[
    Union[RequestNew, RequestActive, Ask, AutoRun, Rewound, RequestDone, Status],
    Field(discriminator="type"),
]
ClientMessage = Annotated[
    Union[Sync, Pick, Typed, Rewind, Send, Decline, SetAutonomy, Teach, Forget],
    Field(discriminator="type"),
]
