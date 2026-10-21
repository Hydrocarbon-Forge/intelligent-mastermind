-- Interaction log: the append-only source of truth. Every learner can be rebuilt from it.
-- Applied as migration 1 by humanapi/store/db.py (PRAGMA user_version = 1).

CREATE TABLE requests (
  id            INTEGER PRIMARY KEY,
  created_at    TEXT NOT NULL,
  client_info   TEXT,
  messages_json TEXT NOT NULL,   -- original messages from the client
  params_json   TEXT,            -- temperature etc., logged but ignored
  prompt_ids    BLOB,            -- packed int32 ids sent to the backend
  status        TEXT NOT NULL,   -- pending | active | done | cancelled | error
  finish_reason TEXT,
  final_text    TEXT,
  final_ids     BLOB,            -- packed int32 ids of the final response
  deleted       INTEGER NOT NULL DEFAULT 0  -- set by forget; deleted requests never produce labels
);

CREATE TABLE positions (
  id            INTEGER PRIMARY KEY,
  request_id    INTEGER NOT NULL REFERENCES requests(id),
  idx           INTEGER NOT NULL,   -- token index within the response
  outcome       TEXT NOT NULL,      -- auto | pick | typed | rewound | explore
  chosen_id     INTEGER,            -- token that ended up committed
  typed_text    TEXT,               -- raw text when outcome = typed
  sample_weight REAL NOT NULL,
  candidates    TEXT NOT NULL,      -- JSON: id, p_base, p_you, features per candidate
  tail_mass     REAL,
  base_k        INTEGER,            -- effective top-k returned by the backend
  pre_update_p  REAL,               -- p_you of the chosen token before the update
  top_conf      REAL,               -- calibrated confidence of the top candidate
  tau           REAL,
  group_name    TEXT,
  rejected_id   INTEGER,            -- auto token removed by a rewind
  feat_vec      BLOB,               -- float16 context embedding (1024 values)
  latency_ms    INTEGER,
  scorer_ver    INTEGER,
  created_at    TEXT NOT NULL
);
CREATE INDEX positions_request ON positions(request_id, idx);
CREATE INDEX positions_outcome ON positions(outcome);

CREATE TABLE scorer_versions (
  id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
  weights BLOB NOT NULL, calib_json TEXT, n_labels INTEGER, notes TEXT
);

CREATE TABLE knowledge_chunks (
  id INTEGER PRIMARY KEY, source TEXT NOT NULL, text TEXT NOT NULL,
  vec BLOB NOT NULL, created_at TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE capabilities (id INTEGER PRIMARY KEY, probed_at TEXT NOT NULL, json TEXT NOT NULL);

CREATE TABLE events (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL,
                     request_id INTEGER, payload_json TEXT);
