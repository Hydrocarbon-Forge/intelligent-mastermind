"""Shared pytest hooks. Skips tests marked tokenizer_files when no tokenizer files are present."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TOKENIZER_DIR = "data/tokenizer"


def tokenizer_dir() -> Path:
    raw = os.environ.get("HUMANAPI_TOKENIZER_PATH", _DEFAULT_TOKENIZER_DIR)
    path = Path(raw)
    return path if path.is_absolute() else _REPO_ROOT / path


def pytest_collection_modifyitems(config, items):
    directory = tokenizer_dir()
    if (directory / "tokenizer.json").exists() or (directory / "tokenizer_config.json").exists():
        return
    reason = (
        "tokenizer files not found in the tokenizer path; copy the Qwen/Qwen3.8-27B "
        "tokenizer files there or set HUMANAPI_TOKENIZER_PATH"
    )
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("tokenizer_files") is not None:
            item.add_marker(marker)
