"""Config loader: defaults, the checked-in config.yaml, partial overrides and rejected keys."""
from __future__ import annotations

from pathlib import Path

import pytest

from humanapi.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_defaults_equal_checked_in_yaml():
    assert load_config(str(REPO_ROOT / "config.yaml")) == Config()


def test_partial_override(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("gate:\n  tau_content: 0.7\nserver:\n  bearer_token: null\n  port: 9000\n", encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.gate.tau_content == 0.7
    assert cfg.gate.tau_function == Config().gate.tau_function
    assert cfg.server.port == 9000
    assert cfg.server.bearer_token is None
    assert cfg.server.host == Config().server.host
    assert cfg.backend == Config().backend


def test_unknown_key_rejected(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("gate:\n  tau_contnt: 0.7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tau_contnt"):
        load_config(str(path))


def test_unknown_section_rejected(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("nonsense:\n  a: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonsense"):
        load_config(str(path))


def test_missing_explicit_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))


def test_default_reads_cwd_or_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert load_config() == Config()
    (tmp_path / "config.yaml").write_text("ui:\n  max_rows: 3\n", encoding="utf-8")
    assert load_config().ui.max_rows == 3
