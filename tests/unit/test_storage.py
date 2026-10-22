"""SQLite layer and interaction log: migrations, round trips, labels, retention and backup."""
from __future__ import annotations

import numpy as np
import pytest

from humanapi.learn.types import Outcome, PositionRecord
from humanapi.store.db import CURRENT_VERSION, backup, connect, migrate
from humanapi.store.log import InteractionLog


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "human.db")


@pytest.fixture
def conn(db_path):
    c = connect(db_path)
    migrate(c)
    yield c
    c.close()


@pytest.fixture
def log(conn):
    return InteractionLog(conn)


def make_request(log: InteractionLog, prompt=(1, 2, 3)) -> int:
    rid = log.create_request(client_info="test", messages=[{"role": "user", "content": "hi"}], params={"t": 0.7})
    log.activate_request(rid, list(prompt))
    return rid


def cands() -> list[dict]:
    return [
        {"id": 5, "p_base": 0.6, "p_you": 0.7, "f": [0.1, 0.2], "cls": "word_start"},
        {"id": -1, "p_base": None, "p_you": 0.3, "f": [0.0, 0.0], "cls": "other"},
    ]


def record(request_id: int, idx: int, outcome: Outcome, **kw) -> PositionRecord:
    base = dict(
        request_id=request_id,
        idx=idx,
        outcome=outcome,
        sample_weight=1.0,
        candidates=cands(),
        chosen_id=5,
        tail_mass=0.02,
        base_k=20,
        pre_update_p=0.7,
        top_conf=0.9,
        tau=0.8,
        group_name="content",
        rejected_id=None,
        feat_vec=None,
        latency_ms=12,
        scorer_ver=1,
    )
    base.update(kw)
    return PositionRecord(**base)


def comparable(rec: PositionRecord) -> dict:
    d = dict(rec.__dict__)
    d.pop("created_at")
    return d


def test_migrations_apply(db_path):
    c = connect(db_path)
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert migrate(c) == CURRENT_VERSION == 1
    assert migrate(c) == 1
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"requests", "positions", "scorer_versions", "events"} <= tables
    log = InteractionLog(c)
    rid = make_request(log)
    c.close()

    # reopening keeps the data and does not re-run migration 1
    c2 = connect(db_path)
    assert migrate(c2) == 1
    assert InteractionLog(c2).request_status(rid) == "active"
    c2.close()


def test_request_lifecycle_and_blobs(conn, log):
    rid = log.create_request(client_info=None, messages=[], params={})
    assert log.request_status(rid) == "pending"
    log.activate_request(rid, [10, -3, 70000])
    raw = conn.execute("SELECT prompt_ids, status FROM requests WHERE id = ?", (rid,)).fetchone()
    assert raw["status"] == "active"
    assert np.frombuffer(raw["prompt_ids"], dtype="<i4").tolist() == [10, -3, 70000]
    log.finish_request(rid, status="done", finish_reason="stop", final_text="ok", final_ids=[4, 5])
    assert log.request_status(rid) == "done"
    assert log.request_final_ids(rid) == [4, 5]


def test_outcome_recording(log):
    rid = make_request(log)
    feat = np.arange(8, dtype=np.float16)
    rows = [
        record(rid, 0, Outcome.AUTO, feat_vec=None),
        record(rid, 1, Outcome.PICK, feat_vec=feat, rejected_id=None),
        record(rid, 2, Outcome.TYPED, chosen_id=None, typed_text=" hel", pre_update_p=None, tail_mass=None),
        record(rid, 3, Outcome.REWOUND, chosen_id=None, rejected_id=5, sample_weight=0.0),
        record(rid, 4, Outcome.EXPLORE, chosen_id=5, sample_weight=20.0, feat_vec=feat, tau=None),
    ]
    for r in rows:
        assert log.record_position(r) == r.id
        assert r.created_at is not None

    labeled = list(log.iter_labeled())
    assert [r.outcome for r in labeled] == [Outcome.PICK, Outcome.TYPED, Outcome.EXPLORE]
    training = list(log.iter_training())
    assert [r.outcome for r in training] == [Outcome.PICK, Outcome.TYPED, Outcome.REWOUND, Outcome.EXPLORE]
    assert [r.idx for r in training] == [1, 2, 3, 4]

    for got, want in zip(training, [rows[1], rows[2], rows[3], rows[4]]):
        assert comparable(got).keys() == comparable(want).keys()
        g, w = comparable(got), comparable(want)
        f_got, f_want = g.pop("feat_vec"), w.pop("feat_vec")
        assert g == w
        if f_want is None:
            assert f_got is None
        else:
            assert f_got.dtype == np.float16
            np.testing.assert_array_equal(f_got, f_want)
    assert training[2].chosen_id is None and training[2].rejected_id == 5


def test_cancelled_not_labeled(log):
    kept = make_request(log)
    cancelled = make_request(log)
    log.record_position(record(kept, 0, Outcome.PICK))
    log.record_position(record(cancelled, 0, Outcome.PICK))
    log.record_position(record(cancelled, 1, Outcome.REWOUND, chosen_id=None))
    log.finish_request(cancelled, status="cancelled", finish_reason="cancelled", final_text="partial", final_ids=None)
    assert log.request_final_ids(cancelled) == []
    assert [r.request_id for r in log.iter_labeled()] == [kept]
    assert [r.request_id for r in log.iter_training()] == [kept]


def test_auto_rows_never_yielded(log):
    rid = make_request(log)
    log.record_position(record(rid, 0, Outcome.AUTO))
    assert list(log.iter_labeled()) == []
    assert list(log.iter_training()) == []


def test_forget_skips_request(log):
    a = make_request(log)
    b = make_request(log)
    log.record_position(record(a, 0, Outcome.PICK))
    log.record_position(record(b, 0, Outcome.PICK))
    log.record_position(record(b, 1, Outcome.REWOUND, chosen_id=None))
    log.forget_request(b)
    assert [r.request_id for r in log.iter_labeled()] == [a]
    assert [r.request_id for r in log.iter_training()] == [a]


def test_scorer_version_retention(log):
    assert log.load_scorer_version() is None
    ids = []
    for i in range(12):
        w = np.full(4, float(i))
        ids.append(log.save_scorer_version(w, {"x": [0.0, 1.0], "i": i}, n_labels=i, notes=f"v{i}", keep=10))
    versions = log.list_scorer_versions()
    assert len(versions) == 10
    assert [v.id for v in versions] == ids[::-1][:10]
    newest = log.load_scorer_version()
    assert newest.id == ids[-1]
    assert newest.weights.dtype == np.float64
    np.testing.assert_array_equal(newest.weights, np.full(4, 11.0))
    assert newest.calib == {"x": [0.0, 1.0], "i": 11}
    assert newest.n_labels == 11 and newest.notes == "v11"
    assert log.load_scorer_version(ids[0]) is None  # pruned
    assert log.load_scorer_version(ids[-2]).id == ids[-2]


def test_events(conn, log):
    rid = make_request(log)
    log.record_event("sync", rid, {"n": 2})
    row = conn.execute("SELECT kind, request_id, payload_json FROM events").fetchone()
    assert row["kind"] == "sync" and row["request_id"] == rid
    assert '"n": 2' in row["payload_json"] or '"n":2' in row["payload_json"]


def test_backup_restore(db_path, tmp_path, conn, log):
    rid = make_request(log)
    log.record_position(record(rid, 0, Outcome.PICK))
    dest = str(tmp_path / "backup.db")
    backup(conn, dest)
    restored = connect(dest)
    try:
        assert migrate(restored) == 1
        rlog = InteractionLog(restored)
        assert rlog.request_status(rid) == "active"
        assert [r.idx for r in rlog.iter_labeled()] == [0]
    finally:
        restored.close()
