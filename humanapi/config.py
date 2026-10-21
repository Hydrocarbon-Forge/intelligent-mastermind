"""Typed settings. The dataclass defaults mirror config.yaml; `load_config` merges a YAML file over them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    model_id: str = "human"
    bearer_token: Optional[str] = None
    max_pending: int = 8


@dataclass
class BackendConfig:
    base_url: str = "http://127.0.0.1:8081/v1"
    model: str = "qwen3.8-27b"
    top_k: int = 20
    timeout_s: float = 30.0
    capabilities_file: str = "data/capabilities.json"


@dataclass
class TokenizerConfig:
    path: str = "data/tokenizer"
    enable_thinking: bool = False


@dataclass
class EmbedConfig:
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cuda"
    context_tokens: int = 128


@dataclass
class UiConfig:
    max_rows: int = 8
    heartbeat_seconds: float = 15.0
    send_key: str = "Ctrl+Enter"


@dataclass
class GateConfig:
    tau_content: float = 0.80
    tau_function: float = 0.60
    tau_sentence_start: float = 0.85
    explore_rate: float = 0.05
    max_auto_run: int = 40
    cold_start_min_labels: int = 200
    cold_start_base_prob: float = 0.95
    abstention_margin: float = 0.10
    rewind_window: int = 100
    rewind_ceiling_per_100: float = 5.0
    rewind_step: float = 0.25  # logit units added to every threshold per controller step
    rewind_max_boost: float = 1.5  # cap on the accumulated logit boost
    calibration_refit_every: int = 50
    lookahead_max: int = 8
    autonomy: float = 0.0  # slider delta in logit space, about -2 to +2


@dataclass
class LearningConfig:
    ngram_orders: list[int] = field(default_factory=lambda: [1, 2, 3])
    ngram_alpha: float = 0.1
    l2_to_init: float = 0.001
    learning_rate: float = 0.5
    tail_floor: float = 0.01
    weight_typed_first: float = 1.0
    weight_typed_continuation: float = 0.3
    recent_window: int = 20
    knn_k: int = 16
    knn_temperature: float = 0.05
    snapshot_every: int = 50
    snapshots_kept: int = 10


@dataclass
class KnowledgeConfig:
    min_similarity: float = 0.35
    top_chunks: int = 4
    chunk_tokens: int = 300
    chunk_overlap_tokens: int = 40


@dataclass
class StorageConfig:
    db_path: str = "data/human.db"
    index_dir: str = "data/index"


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def load_config(path: Optional[str] = None) -> Config:
    """Load settings from a YAML file (default: config.yaml next to the package root) over the defaults.

    Unknown sections or keys raise ValueError. A missing file at an explicit path raises FileNotFoundError.
    """
    raise NotImplementedError
