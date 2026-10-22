"""The interaction log: append-only source of truth for requests, positions, scorer snapshots and events."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterator, Optional, Sequence

import numpy as np

from humanapi.learn.types import Outcome, PositionRecord, ScorerVersion

_LABELED = ("pick", "typed", "explore")
_TRAINING = ("pick", "typed", "explore", "rewound")
_FINAL_STATUSES = ("done", "cancelled", "error")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pack_ids(ids: Sequence[int]) -> bytes:
    return np.asarray(list(ids), dtype="<i4").tobytes()


def _unpack_ids(blob: Optional[bytes]) -> list[int]:
    if not blob:
        return []
    return np.frombuffer(blob, dtype="<i4").astype(int).tolist()


class InteractionLog:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---- requests

    def create_request(self, *, client_info: Optional[str], messages: list[dict], params: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO requests (created_at, client_info, messages_json, params_json, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (_now(), client_info, json.dumps(messages), json.dumps(params)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def activate_request(self, request_id: int, prompt_ids: Sequence[int]) -> None:
        self.conn.execute(
            "UPDATE requests SET status = 'active', prompt_ids = ? WHERE id = ?",
            (_pack_ids(prompt_ids), request_id),
        )
        self.conn.commit()

    def finish_request(
        self,
        request_id: int,
        *,
        status: str,
        finish_reason: Optional[str],
        final_text: Optional[str],
        final_ids: Optional[Sequence[int]],
    ) -> None:
        if status not in _FINAL_STATUSES:
            raise ValueError(f"status must be one of {_FINAL_STATUSES}, got {status!r}")
        self.conn.execute(
            "UPDATE requests SET status = ?, finish_reason = ?, final_text = ?, final_ids = ? WHERE id = ?",
            (
                status,
                finish_reason,
                final_text,
                _pack_ids(final_ids) if final_ids is not None else None,
                request_id,
            ),
        )
        self.conn.commit()

    def request_final_ids(self, request_id: int) -> list[int]:
        row = self.conn.execute("SELECT final_ids FROM requests WHERE id = ?", (request_id,)).fetchone()
        return _unpack_ids(row["final_ids"]) if row is not None else []

    def request_status(self, request_id: int) -> Optional[str]:
        row = self.conn.execute("SELECT status FROM requests WHERE id = ?", (request_id,)).fetchone()
        return row["status"] if row is not None else None

    def forget_request(self, request_id: int) -> None:
        self.conn.execute("UPDATE requests SET deleted = 1 WHERE id = ?", (request_id,))
        self.conn.commit()

    # ---- positions and events

    def record_position(self, rec: PositionRecord) -> int:
        created_at = rec.created_at or _now()
        feat = None if rec.feat_vec is None else np.asarray(rec.feat_vec, dtype=np.float16).tobytes()
        cur = self.conn.execute(
            "INSERT INTO positions (request_id, idx, outcome, chosen_id, typed_text, sample_weight, candidates, "
            "tail_mass, base_k, pre_update_p, top_conf, tau, group_name, rejected_id, feat_vec, latency_ms, "
            "scorer_ver, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.request_id,
                rec.idx,
                Outcome(rec.outcome).value,
                rec.chosen_id,
                rec.typed_text,
                float(rec.sample_weight),
                json.dumps(rec.candidates),
                rec.tail_mass,
                rec.base_k,
                rec.pre_update_p,
                rec.top_conf,
                rec.tau,
                rec.group_name,
                rec.rejected_id,
                feat,
                rec.latency_ms,
                rec.scorer_ver,
                created_at,
            ),
        )
        self.conn.commit()
        rec.id = int(cur.lastrowid)
        rec.created_at = created_at
        return rec.id

    def record_event(self, kind: str, request_id: Optional[int] = None, payload: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, kind, request_id, payload_json) VALUES (?, ?, ?, ?)",
            (_now(), kind, request_id, json.dumps(payload if payload is not None else {})),
        )
        self.conn.commit()

    # ---- reading labels

    def _positions(self, outcomes: tuple[str, ...]) -> Iterator[PositionRecord]:
        marks = ", ".join("?" for _ in outcomes)
        rows = self.conn.execute(
            f"SELECT p.* FROM positions p JOIN requests r ON r.id = p.request_id "
            f"WHERE p.outcome IN ({marks}) AND r.status != 'cancelled' AND r.deleted = 0 "
            f"ORDER BY p.id",
            outcomes,
        ).fetchall()
        for row in rows:
            yield _row_to_record(row)

    def iter_labeled(self) -> Iterator[PositionRecord]:
        return self._positions(_LABELED)

    def iter_training(self) -> Iterator[PositionRecord]:
        return self._positions(_TRAINING)

    # ---- scorer versions

    def save_scorer_version(
        self,
        weights: np.ndarray,
        calib: Optional[dict],
        n_labels: int,
        notes: Optional[str] = None,
        keep: int = 10,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO scorer_versions (created_at, weights, calib_json, n_labels, notes) VALUES (?, ?, ?, ?, ?)",
            (
                _now(),
                np.asarray(weights, dtype="<f8").tobytes(),
                json.dumps(calib) if calib is not None else None,
                int(n_labels),
                notes,
            ),
        )
        new_id = int(cur.lastrowid)
        self.conn.execute(
            "DELETE FROM scorer_versions WHERE id NOT IN "
            "(SELECT id FROM scorer_versions ORDER BY id DESC LIMIT ?)",
            (max(1, int(keep)),),
        )
        self.conn.commit()
        return new_id

    def load_scorer_version(self, version_id: Optional[int] = None) -> Optional[ScorerVersion]:
        if version_id is None:
            row = self.conn.execute("SELECT * FROM scorer_versions ORDER BY id DESC LIMIT 1").fetchone()
        else:
            row = self.conn.execute("SELECT * FROM scorer_versions WHERE id = ?", (version_id,)).fetchone()
        return _row_to_scorer(row) if row is not None else None

    def list_scorer_versions(self) -> list[ScorerVersion]:
        rows = self.conn.execute("SELECT * FROM scorer_versions ORDER BY id DESC").fetchall()
        return [_row_to_scorer(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> PositionRecord:
    feat = None
    if row["feat_vec"] is not None:
        feat = np.frombuffer(row["feat_vec"], dtype=np.float16).copy()
    return PositionRecord(
        request_id=row["request_id"],
        idx=row["idx"],
        outcome=Outcome(row["outcome"]),
        sample_weight=row["sample_weight"],
        candidates=json.loads(row["candidates"]),
        chosen_id=row["chosen_id"],
        typed_text=row["typed_text"],
        tail_mass=row["tail_mass"],
        base_k=row["base_k"],
        pre_update_p=row["pre_update_p"],
        top_conf=row["top_conf"],
        tau=row["tau"],
        group_name=row["group_name"],
        rejected_id=row["rejected_id"],
        feat_vec=feat,
        latency_ms=row["latency_ms"],
        scorer_ver=row["scorer_ver"],
        id=row["id"],
        created_at=row["created_at"],
    )


def _row_to_scorer(row: sqlite3.Row) -> ScorerVersion:
    return ScorerVersion(
        id=row["id"],
        created_at=row["created_at"],
        weights=np.frombuffer(row["weights"], dtype="<f8").copy(),
        calib=json.loads(row["calib_json"]) if row["calib_json"] is not None else None,
        n_labels=int(row["n_labels"] or 0),
        notes=row["notes"],
    )
