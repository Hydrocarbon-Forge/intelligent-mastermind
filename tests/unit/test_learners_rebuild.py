import copy

import numpy as np

from humanapi.config import Config
from humanapi.learn.features import N_FEATURES
from humanapi.learn.learners import Learners
from humanapi.learn.types import OTHER_ID, Outcome, PositionRecord
from humanapi.store import db
from humanapi.store.log import InteractionLog
from humanapi.store.rebuild import forget_requests, main, rebuild

EMBED_DIM = 16
VOCAB = 300
END_ID = 256
CANDS = (10, 11, 12, END_ID)


def _new_log() -> InteractionLog:
    conn = db.connect(":memory:")
    db.migrate(conn)
    return InteractionLog(conn)


def _cands(rng, ids, p_you):
    out = []
    for tid, p in zip(ids, p_you):
        out.append({
            "id": int(tid),
            "p_base": float(p),
            "p_you": float(p),
            "f": rng.normal(scale=0.5, size=N_FEATURES).tolist(),
            "cls": "end" if tid == END_ID else "word_start",
        })
    out.append({"id": OTHER_ID, "p_base": None, "p_you": 0.1,
                "f": rng.normal(scale=0.5, size=N_FEATURES).tolist(), "cls": "other"})
    return out


def _rec(rng, request_id, idx, outcome, chosen, *, group="content", top_conf=None, tau=None,
         rejected=None, weight=1.0, feat=True):
    p = rng.dirichlet(np.ones(len(CANDS))).tolist()
    return PositionRecord(
        request_id=request_id,
        idx=idx,
        outcome=outcome,
        sample_weight=weight,
        candidates=_cands(rng, CANDS, p),
        chosen_id=chosen,
        group_name=group,
        top_conf=top_conf,
        tau=tau,
        rejected_id=rejected,
        feat_vec=(rng.normal(size=EMBED_DIM).astype(np.float16) if feat else None),
    )


def _session(log, cfg, seed):
    """Create three requests and log a scripted set of positions, applying each live as it is logged.
    Returns the live learners, the records in log order and the request ids."""
    rng = np.random.default_rng(seed)
    learners = Learners(cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB)
    request_ids, records = [], []
    for r in range(3):
        rid = log.create_request(client_info=None, messages=[], params={})
        request_ids.append(rid)
        final = [int(x) for x in rng.choice(CANDS, size=6)]
        log.finish_request(rid, status="done", finish_reason="stop", final_text="", final_ids=final)
        script = [
            _rec(rng, rid, 0, Outcome.AUTO, 10),
            _rec(rng, rid, 1, Outcome.PICK, 11, top_conf=0.9, tau=0.5),
            _rec(rng, rid, 2, Outcome.REWOUND, None, group=None, rejected=10, feat=False),
            _rec(rng, rid, 3, Outcome.TYPED, 12, group="function", top_conf=0.2, tau=0.6, weight=0.3),
            _rec(rng, rid, 4, Outcome.PICK, 11, group=None, feat=False),
            _rec(rng, rid, 5, Outcome.EXPLORE, END_ID, weight=1.0),
        ]
        for rec in script:
            log.record_position(rec)
            learners.apply(rec, final)
            records.append(rec)
    return learners, records, request_ids


def test_rebuild_idempotent():
    cfg = Config()
    log = _new_log()
    _session(log, cfg, seed=1)
    d1 = rebuild(log, cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB).digest()
    d2 = rebuild(log, cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB).digest()
    assert d1 == d2


def test_rebuild_matches_live():
    cfg = Config()
    log = _new_log()
    live, _, _ = _session(log, cfg, seed=2)
    rebuilt = rebuild(log, cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB)
    assert rebuilt.digest() == live.digest()
    assert live.scorer.n_updates > 0 and len(live.knn) > 0


def test_forget_rebuilds():
    cfg = Config()
    log = _new_log()
    _, records, rids = _session(log, cfg, seed=3)
    forgotten = rids[0]
    learners = forget_requests(log, [forgotten], cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB)
    assert log.request_status(forgotten) == "done"
    row = log.conn.execute("SELECT deleted FROM requests WHERE id = ?", (forgotten,)).fetchone()
    assert row["deleted"] == 1
    remaining = [r for r in records if r.request_id != forgotten]
    labeled = [r for r in remaining if r.outcome in (Outcome.PICK, Outcome.TYPED, Outcome.EXPLORE)]
    assert len(learners.knn) == sum(1 for r in labeled if r.feat_vec is not None)
    assert np.isclose(learners.ngram.total_weight, sum(r.sample_weight for r in labeled))


def test_backup_restore(tmp_path):
    cfg = Config()
    log = _new_log()
    _session(log, cfg, seed=4)
    original = rebuild(log, cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB).digest()
    dest = str(tmp_path / "backup.db")
    db.backup(log.conn, dest)
    restored = InteractionLog(db.connect(dest))
    assert rebuild(restored, cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB).digest() == original


def test_prequential_logged():
    cfg = Config()
    rng = np.random.default_rng(5)
    learners = Learners(cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB)
    rec = _rec(rng, 1, 0, Outcome.PICK, 11, top_conf=0.9, tau=0.5)
    feats = np.array([c["f"] for c in rec.candidates])
    row = [c["id"] for c in rec.candidates].index(11)
    expected = float(learners.scorer.probs(feats)[row])
    returned = learners.apply(rec, [])
    assert returned is not None
    assert abs(returned - expected) < 1e-12
    assert learners.scorer.n_updates == 1


def test_snapshot_rollback():
    cfg = Config()
    cfg.learning.snapshot_every = 1
    log = _new_log()
    rid = log.create_request(client_info=None, messages=[], params={})
    learners = Learners(cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB)
    rng = np.random.default_rng(6)
    saved = {}
    for i in range(12):
        rec = _rec(rng, rid, i, Outcome.PICK, 11, group=None, feat=False)
        log.record_position(rec)
        learners.apply(rec, [])
        vid = learners.maybe_snapshot(log)
        assert vid is not None
        saved[vid] = learners.scorer.snapshot()
        assert learners.maybe_snapshot(log) is None  # no duplicate for the same count
    versions = log.list_scorer_versions()
    assert len(versions) == 10
    target = versions[-1].id  # oldest kept
    rec = _rec(rng, rid, 99, Outcome.PICK, 12, group=None, feat=False)
    learners.apply(rec, [])
    assert learners.rollback(log, target) is True
    assert np.array_equal(learners.scorer.weights, saved[target])
    assert learners.rollback(log, 10**9) is False


def test_auto_never_updates():
    cfg = Config()
    learners = Learners(cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB)
    rng = np.random.default_rng(7)
    before = copy.deepcopy(learners.digest())
    rec = _rec(rng, 1, 0, Outcome.AUTO, 11)
    assert learners.apply(rec, [10, 11]) is None
    assert learners.digest() == before


def test_ngram_ignores_auto():
    cfg = Config()
    learners = Learners(cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB)
    rng = np.random.default_rng(8)
    learners.apply(_rec(rng, 1, 0, Outcome.AUTO, 11), [10])
    assert learners.ngram.total_weight == 0.0
    assert len(learners.knn) == 0


def test_calibration_rule():
    cfg = Config()
    learners = Learners(cfg, embed_dim=EMBED_DIM, vocab_size=VOCAB)
    rng = np.random.default_rng(9)

    def obs(group):
        return learners.calibrator.to_json()["obs"].get(group, [])

    # Explore counts with weight 1 / explore_rate.
    learners.apply(_rec(rng, 1, 0, Outcome.EXPLORE, END_ID, group="content"), [])
    assert obs("content")[-1][2] == 1.0 / cfg.gate.explore_rate

    # A pick asked for low confidence counts with weight 1.
    learners.apply(_rec(rng, 1, 1, Outcome.PICK, 11, group="content", top_conf=0.2, tau=0.5), [])
    assert len(obs("content")) == 2 and obs("content")[-1][2] == 1.0

    # A forced ask (confidence above the threshold) is skipped.
    learners.apply(_rec(rng, 1, 2, Outcome.PICK, 11, group="content", top_conf=0.9, tau=0.5), [])
    assert len(obs("content")) == 2

    # A record without a group never calibrates, but still updates the scorer.
    before_updates = learners.scorer.n_updates
    learners.apply(_rec(rng, 1, 3, Outcome.PICK, 11, group=None, top_conf=0.2, tau=0.5), [])
    assert len(obs("content")) == 2 and learners.scorer.n_updates == before_updates + 1


def test_main_smoke(tmp_path):
    dest = str(tmp_path / "smoke.db")
    assert main(["--db", dest, "--embed-dim", str(EMBED_DIM), "--vocab-size", str(VOCAB)]) == 0
