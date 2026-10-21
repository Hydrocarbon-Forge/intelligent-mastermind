# Storage schema, outcomes, rebuild (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Storage and rebuild
  The log is the source of truth. It is append-only: rows are never edited, and corrections are
  new rows. Everything the learners know can be recomputed from it.

     CREATE TABLE requests (
       id            INTEGER PRIMARY KEY,
       created_at    TEXT NOT NULL,
          client_info   TEXT,
          messages_json TEXT NOT NULL,                   -- original messages from the client
          params_json                TEXT,               -- temperature etc., logged but ignored
          prompt_ids                 BLOB,               -- packed int32 ids sent to the backend
          status                     TEXT NOT NULL,      -- pending | active | done | cancelled |
     error
       finish_reason TEXT,
       final_text    TEXT,
          final_ids                  BLOB
     );

     CREATE TABLE positions (
       id            INTEGER PRIMARY KEY,
          request_id                 INTEGER NOT NULL REFERENCES requests(id),
          idx                        INTEGER NOT NULL,   -- token index within the response
          outcome                    TEXT NOT NULL,      -- auto | pick | typed | rewound |
     explore
       chosen_id                     INTEGER,            -- token that ended up committed
          typed_text    TEXT,                            -- raw text when outcome = typed
          sample_weight REAL NOT NULL,
          candidates    TEXT NOT NULL,                   -- JSON: id, p_base, p_you, features per
     candidate
       tail_mass                     REAL,

       base_k                        INTEGER,          -- effective top-k returned by the
     backend
       pre_update_p                  REAL,             -- p_you of the chosen token before the
     update
       top_conf                      REAL,             -- calibrated confidence of the top
     candidate
       tau                           REAL,
          group_name                 TEXT,
          rejected_id                INTEGER,          -- auto token removed by a rewind
          feat_vec                   BLOB,             -- float16 context embedding (1024
     values)
       latency_ms                    INTEGER,
       scorer_ver                    INTEGER,
          created_at                 TEXT NOT NULL
     );
     CREATE INDEX positions_request ON positions(request_id, idx);
     CREATE INDEX positions_outcome ON positions(outcome);

     CREATE TABLE scorer_versions (
       id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
       weights BLOB NOT NULL, calib_json TEXT, n_labels INTEGER, notes TEXT
     );

     CREATE TABLE knowledge_chunks (
       id INTEGER PRIMARY KEY, source TEXT NOT NULL, text TEXT NOT NULL,
       vec BLOB NOT NULL, created_at TEXT NOT NULL, deleted INTEGER NOT NULL
     DEFAULT 0
     );

     CREATE TABLE capabilities (id INTEGER PRIMARY KEY, probed_at TEXT NOT NULL,
     json TEXT NOT NULL);
     CREATE TABLE events (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT
     NULL,
                          request_id INTEGER, payload_json TEXT);

  How outcomes are recorded
        A rewind writes the auto-selected position with outcome rewound and the token in
         rejected_id . The re-asked position then gets its own row with outcome pick or
         typed .
        An exploration ask is recorded with outcome explore once you answer it, so
        calibration can find it.

        A labeled position is any row with outcome pick , typed or explore . Auto and
        rewound rows are never labels, although rewound rows carry a penalty in the scorer.
        A typed multi-token word produces one row per token, all sharing the typed text, with
        the lower sample weight for continuation tokens.

  Rebuild
   python -m humanapi.store.rebuild recomputes the learners in this order:

   1. N-gram memory: replay labeled positions in order and count chosen_id under its
        context.
   2. kNN index: load feat_vec and chosen_id for labeled positions and build the index.
   3. Scorer: reset to w0 , then replay labeled and rewound positions using the stored per-
        candidate features, sample weights and rewind penalties. This reproduces the original
        weight trajectory without calling the backend.
   4. Calibration: refit from labeled and exploration positions with their weights.

  When the model changes
  The stored ids and probabilities belong to one tokenizer and one base model. If the base
  model changes but the tokenizer stays the same, the n-gram memory and kNN index
  remain valid, but the stored p_base values do not, so the scorer resets to w0 and
  relearns. If the tokenizer changes, only the text ( final_text and typed text) survives,
  and everything is re-derived from it.

  Operations
        SQLite runs in WAL mode, and the backup API runs on a schedule.
        Migrations are tracked with PRAGMA user_version and numbered SQL files.
        A forget command marks rows or chunks as deleted and rebuilds the affected
        learner, so removals are real.

  Tests

     Test                                              Asserts

      test_migrations_apply                            A fresh database reaches the current
                                                       user_version , and re-running does nothing

     Test                                              Asserts

      test_outcome_recording                           Auto, pick, typed, rewind and explore each write
                                                       the rows described above

      test_typed_word_rows                             A three-token typed word writes three rows with
                                                       the right sample weights

      test_cancelled_not_labeled                       Positions of a cancelled request are kept but
                                                       excluded from labels

      test_rebuild_idempotent                          Running rebuild twice gives identical weights,
                                                       counts and index

      test_rebuild_matches_live                        After a scripted session, a rebuild reproduces the
                                                       live learners

      test_forget_rebuilds                             forget marks the rows deleted and the rebuilt
                                                       learners no longer reflect them

      test_backup_restore                              A backup restores to a database that passes the
                                                       rebuild test
```
