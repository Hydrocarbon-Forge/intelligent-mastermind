"""Rebuild every learner from the interaction log, forget requests, and the `python -m humanapi.store.rebuild` command.

The log is the source of truth. A rebuild replays the training records in row order through
`Learners.apply`, so it reproduces the live learners exactly.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from humanapi.config import Config, load_config
from humanapi.learn.learners import Learners
from humanapi.store import db
from humanapi.store.log import InteractionLog

_DEFAULT_EMBED_DIM = 1024
_DEFAULT_VOCAB_SIZE = 150000


def rebuild(
    log: InteractionLog,
    cfg: Config,
    *,
    embed_dim: int,
    vocab_size: int,
    reset_scorer: bool = False,
) -> Learners:
    learners = Learners(cfg, embed_dim=embed_dim, vocab_size=vocab_size)
    cache: dict[int, list[int]] = {}

    def final_ids(request_id: int) -> list[int]:
        if request_id not in cache:
            cache[request_id] = log.request_final_ids(request_id)
        return cache[request_id]

    if reset_scorer:
        # The base model changed: only the n-gram memory and kNN index are rebuilt. The scorer stays at w0
        # and the calibrator stays empty.
        for rec in log.iter_labeled():
            if rec.chosen_id is None:
                continue
            learners.learn_memories(rec, final_ids(rec.request_id))
            learners.n_labels += 1
    else:
        for rec in log.iter_training():
            learners.apply(rec, final_ids(rec.request_id))
    return learners


def forget_requests(
    log: InteractionLog,
    request_ids: Sequence[int],
    cfg: Config,
    *,
    embed_dim: int,
    vocab_size: int,
) -> Learners:
    for request_id in request_ids:
        log.forget_request(int(request_id))
    return rebuild(log, cfg, embed_dim=embed_dim, vocab_size=vocab_size)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m humanapi.store.rebuild", description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml when present)")
    p.add_argument("--db", default=None, help="database path (default: storage.db_path from the config)")
    p.add_argument("--embed-dim", type=int, default=_DEFAULT_EMBED_DIM)
    p.add_argument("--vocab-size", type=int, default=_DEFAULT_VOCAB_SIZE)
    p.add_argument("--backup", default=None, metavar="DEST", help="copy the database to DEST before rebuilding")
    p.add_argument("--forget", type=int, default=None, metavar="REQUEST_ID", help="forget one request first")
    p.add_argument("--rollback", type=int, default=None, metavar="VERSION_ID",
                   help="restore this scorer version into the rebuilt learners")
    p.add_argument("--reset-scorer", action="store_true", help="reset the scorer to w0 and empty calibrator")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    db_path = args.db if args.db is not None else cfg.storage.db_path

    conn = db.connect(db_path)
    try:
        db.migrate(conn)
        if args.backup is not None:
            db.backup(conn, args.backup)
        log = InteractionLog(conn)
        if args.forget is not None:
            log.forget_request(args.forget)

        learners = rebuild(
            log, cfg, embed_dim=args.embed_dim, vocab_size=args.vocab_size, reset_scorer=args.reset_scorer
        )
        if args.rollback is not None:
            if not learners.rollback(log, args.rollback):
                print(f"no scorer version {args.rollback}", file=sys.stderr)
                return 1
            notes = "rollback"
        else:
            notes = "rebuild"
        log.save_scorer_version(
            learners.scorer.snapshot(),
            learners.calibrator.to_json(),
            learners.n_labels,
            notes=notes,
            keep=cfg.learning.snapshots_kept,
        )
        print(
            f"labels={learners.n_labels} scorer_updates={learners.scorer.n_updates} "
            f"knn_rows={len(learners.knn)}"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
