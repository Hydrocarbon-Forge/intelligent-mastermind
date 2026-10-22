"""Replay harness and metrics, run on the synthetic corpus with ToyTokenizer, FakeBackend and HashEmbedder."""
from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from humanapi.backend.fake import FakeBackend
from humanapi.config import Config
from humanapi.eval import metrics
from humanapi.eval.replay import (
    CorpusItem,
    ReplayConfig,
    load_corpus,
    make_fake_model,
    run_replay,
)
from humanapi.learn.embed import HashEmbedder
from humanapi.text.toy_tokenizer import ToyTokenizer

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus_small.jsonl"


def make_cfg() -> Config:
    cfg = Config()
    # Smaller cold-start minimum so the 90-line corpus can show learning; production defaults are untouched.
    cfg.gate.cold_start_min_labels = 20
    return cfg


def make_backend(tok: ToyTokenizer, seed: int = 0) -> FakeBackend:
    return FakeBackend(make_fake_model(tok, seed), token_bytes=lambda i: tok.token_bytes(i) or b"")


def run(items, *, rc: ReplayConfig | None = None, cfg: Config | None = None, backend=None):
    tok = ToyTokenizer()
    backend = backend if backend is not None else make_backend(tok, seed=0)
    return asyncio.run(
        run_replay(
            items,
            cfg=cfg if cfg is not None else make_cfg(),
            tok=tok,
            backend=backend,
            embedder=HashEmbedder(),
            rc=rc if rc is not None else ReplayConfig(),
        )
    )


@pytest.fixture(scope="module")
def corpus() -> list[CorpusItem]:
    return load_corpus(str(CORPUS))


@pytest.fixture(scope="module")
def replayed(corpus) -> object:
    return run(corpus)


def test_corpus_file_shape(corpus):
    assert 80 <= len(corpus) <= 100
    assert all(item.messages and item.response for item in corpus)


def test_load_corpus_reports_bad_line(tmp_path):
    good = json.dumps({"messages": [{"role": "user", "content": "hi"}], "response": "Yes!"})
    path = tmp_path / "bad.jsonl"
    path.write_text(f"{good}\n\n{{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":3:"):
        load_corpus(str(path))


def test_load_corpus_skips_blank_lines(tmp_path):
    good = json.dumps({"messages": [{"role": "user", "content": "hi"}], "response": "Yes!"})
    path = tmp_path / "ok.jsonl"
    path.write_text(f"\n{good}\n   \n{good}\n", encoding="utf-8")
    assert len(load_corpus(str(path))) == 2


def test_replay_learning_curve_rises(replayed):
    curve = replayed.learning_curve
    assert len(curve) >= 6
    third = max(1, len(curve) // 3)
    first = sum(rate for _, rate in curve[:third]) / third
    last = sum(rate for _, rate in curve[-third:]) / third
    assert last - first >= 0.1, curve


def test_replay_precision_holds(replayed):
    cfg = make_cfg()
    smallest = min(cfg.gate.tau_content, cfg.gate.tau_function, cfg.gate.tau_sentence_start)
    assert replayed.eval_precision is not None
    assert replayed.eval_precision >= smallest - 0.05


def test_replay_deterministic(corpus):
    first = run(corpus)
    second = run(corpus)
    assert dataclasses.asdict(first) == dataclasses.asdict(second)


def test_disk_cache_removes_backend_calls(corpus, tmp_path):
    cache = str(tmp_path / "cache")
    tok = ToyTokenizer()
    first_backend = make_backend(tok)
    first = run(corpus, rc=ReplayConfig(cache_dir=cache), backend=first_backend)
    assert len(first_backend.calls) > 0
    assert any(Path(cache).iterdir())

    second_backend = make_backend(tok)
    second = run(corpus, rc=ReplayConfig(cache_dir=cache), backend=second_backend)
    assert second_backend.calls == []
    assert dataclasses.asdict(first) == dataclasses.asdict(second)


def test_small_corpus_and_single_token_responses():
    tok = ToyTokenizer()
    items = [
        CorpusItem(messages=[{"role": "user", "content": "ok?"}], response="Sure"),  # one token plus END
        CorpusItem(messages=[{"role": "user", "content": "hi"}], response="Yes"),
    ]
    result = run(items, rc=ReplayConfig(curve_window=10))
    assert result.n_train == 1 and result.n_eval == 1
    assert result.learning_curve == []  # no full window of ten responses
    assert result.eval_precision is None or 0.0 <= result.eval_precision <= 1.0


def test_empty_eval_slice_gives_zero_rates(corpus):
    result = run(corpus[:12], rc=ReplayConfig(train_fraction=1.0))
    assert result.n_eval == 0
    assert result.eval_auto_rate == 0.0
    assert result.eval_precision is None


def test_reliability_bins_and_calibration_error():
    points = [(0.05, False, 1.0), (0.15, True, 1.0), (0.95, True, 1.0), (0.97, True, 2.0)]
    bins = metrics.reliability_bins(points, n_bins=10)
    assert [round(b["lo"], 2) for b in bins] == [0.0, 0.1, 0.9]
    assert bins[-1]["n"] == pytest.approx(3.0)
    assert bins[-1]["accuracy"] == pytest.approx(1.0)
    assert bins[0]["accuracy"] == pytest.approx(0.0)
    ece = metrics.expected_calibration_error(bins)
    expected = (1.0 * abs(0.05 - 0.0) + 1.0 * abs(0.15 - 1.0) + 3.0 * abs((0.95 * 1 + 0.97 * 2) / 3 - 1.0)) / 5.0
    assert ece == pytest.approx(expected)
    assert metrics.expected_calibration_error([]) == 0.0
    assert metrics.precision(3, 0) is None
    assert metrics.precision(3, 4) == pytest.approx(0.75)
    assert metrics.asks_per_100(5, 20) == pytest.approx(25.0)
    assert metrics.bits_contributed([0.5, 0.25]) == pytest.approx(3.0)
