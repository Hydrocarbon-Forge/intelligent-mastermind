"""Shared learning-layer types: candidates, scored positions, gate decisions, and the log record shapes."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import numpy as np

from humanapi.backend.base import NextTokenResult
from humanapi.text.types import PosState, TokenClass

OTHER_ID = -1  # token_id of the OTHER pseudo-candidate ("you will type something that is not listed")

# Candidate sources, used in Candidate.sources.
SRC_BACKEND = "backend"
SRC_NGRAM = "ngram"
SRC_KNN = "knn"
SRC_PREFIX = "prefix"
SRC_END = "end"


class GateGroup(str, Enum):
    CONTENT = "content"
    FUNCTION = "function"  # punctuation and mid-word continuations
    SENTENCE_START = "sentence_start"


class Outcome(str, Enum):
    AUTO = "auto"
    PICK = "pick"
    TYPED = "typed"
    REWOUND = "rewound"
    EXPLORE = "explore"


@dataclass
class Candidate:
    token_id: int
    text: str  # decoded text of the token (no display markers)
    p_base: Optional[float]  # None when the backend did not return it
    base_unknown: bool  # True when p_base is None, or only a bound
    base_rank: int  # rank in the backend list, capped at k + 1
    token_class: TokenClass
    sources: frozenset[str]  # subset of the SRC_* constants


@dataclass
class PositionContext:
    """Everything the candidate, feature and gate code needs to know about one token position."""

    response_ids: list[int]  # tokens of the response so far (flushed and provisional)
    state: PosState
    typed_prefix: str = ""  # partial word the person is typing (held out of the model context)
    result: Optional[NextTokenResult] = None  # None when the backend is unavailable
    filtered_probs: bool = False
    embedding: Optional[np.ndarray] = None  # normalized context embedding (word-start positions only)
    retrieval_hit: bool = False
    retrieval_max_sim: float = 0.0
    retrieval_ids: frozenset[int] = frozenset()  # token ids occurring in retrieved knowledge chunks


@dataclass
class ScoredPosition:
    candidates: list[Candidate]  # real tokens only (END always included); OTHER is not in this list
    features: np.ndarray  # shape (len(candidates) + 1, n_features); the last row is OTHER
    probs: np.ndarray  # shape (len(candidates) + 1,), softmax over candidates + OTHER; last is OTHER
    state: PosState
    p_other: float
    base_k: int
    tail_mass: float
    retrieval_hit: bool
    retrieval_max_sim: float


@dataclass(frozen=True)
class Ask:
    explore: bool = False


@dataclass(frozen=True)
class Auto:
    token_id: int


@dataclass
class Decision:
    action: Union[Ask, Auto]
    top_id: Optional[int]  # best non-OTHER, non-END candidate (None when there is none)
    p_top: float  # raw p_you of the top candidate
    conf: float  # calibrated confidence of the top candidate
    tau: float  # threshold that was applied (after slider, controller and abstention margin)
    group: GateGroup


@dataclass
class PositionRecord:
    """One row of the `positions` table (humanapi/store/schema.sql)."""

    request_id: int
    idx: int
    outcome: Outcome
    sample_weight: float
    candidates: list[dict[str, Any]] = field(default_factory=list)
    # candidates: [{"id": int, "p_base": float | None, "p_you": float, "f": list[float], "cls": str}, ...]
    # "cls" is the TokenClass value ("other" for the OTHER entry). The last entry is OTHER ("id": -1).
    # A chosen token that was not listed is added (with its features) before OTHER.
    chosen_id: Optional[int] = None  # token that ended up committed
    typed_text: Optional[str] = None  # raw typed text when outcome is typed
    tail_mass: Optional[float] = None
    base_k: Optional[int] = None
    pre_update_p: Optional[float] = None  # p_you of the chosen token before the update
    top_conf: Optional[float] = None
    tau: Optional[float] = None
    group_name: Optional[str] = None
    rejected_id: Optional[int] = None  # auto token removed by a rewind
    feat_vec: Optional[np.ndarray] = None  # float16 context embedding, None when not computed
    latency_ms: Optional[int] = None
    scorer_ver: Optional[int] = None
    id: Optional[int] = None  # set by the log on insert
    created_at: Optional[str] = None


@dataclass
class ScorerVersion:
    id: int
    created_at: str
    weights: np.ndarray  # float64
    calib: Optional[dict[str, Any]]
    n_labels: int
    notes: Optional[str]
