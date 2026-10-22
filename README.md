# Intelligent Mastermind

A local OpenAI-compatible API whose model is a person, assisted by learned memories.

Any OpenAI client sends a chat request to this server. The request waits in a queue until you pick it up in the
browser UI. For each token the program proposes candidates from a local language model, your learned memories
(an n-gram memory of your own choices, a kNN index over context embeddings, and a personal scorer) and your notes.
Confident positions are filled in automatically; the rest are asked. Your decisions are streamed back to the
client, and every decision is logged and used to learn.

## Architecture

Two programs run on the runtime machine:

| Program | What it is | Port |
| --- | --- | --- |
| `llama-server` | The inference server (Qwen3.8-27B, OpenAI-compatible completions, logprobs) | 8081 |
| `humanapi` | This project: the API, the UI WebSocket, the learners and the log | 8080 |

The development machine needs only the Python tests and the UI build; no model runs there.

Inside `humanapi`: `text` (tokenizer wrapper, typed-word commit, token classes), `backend` (the only code that
talks to the inference server), `learn` (n-gram memory, kNN, scorer, gate and calibration), `core` (queue, per-request
driver and engine), `store` (SQLite interaction log, migrations, rebuild), `knowledge` (notes and retrieval),
`api` (OpenAI routes, UI WebSocket) and `eval` (replay harness).

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt        # the test suite and the server (CPU only)
.venv/bin/pip install -r requirements-runtime.txt    # runtime extras: transformers, sentence-transformers
cd ui && npm install && npm run build && cd ..       # writes the UI into humanapi/static
```

Put the Qwen tokenizer files in the folder named by `tokenizer.path` (default `data/tokenizer`).
Without them the server does not start; the error names `tokenizer.path`.

## Running

1. Start the inference server on port 8081, for example:
   `llama-server -m <model.gguf> --port 8081 --top-p 0.80 --top-k 20`
2. Probe what the server supports (writes `data/capabilities.json`):
   `.venv/bin/python -m tools.probe_backend`
3. Start the program:
   `.venv/bin/uvicorn humanapi.main:app --port 8080` (or `.venv/bin/python -m humanapi`, which reads
   `server.host` and `server.port` from the config)
4. Open `http://localhost:8080/` in a browser. Point your OpenAI client at `http://localhost:8080/v1`.

The server starts even when the inference server is down. `GET /health` reports `"backend": false` until it
answers; in the meantime every position is a plain ask.

The UI is a single page. Requests wait in the request list until you open them. Letters go into the type box;
a space or punctuation commits the typed word; Enter also commits a newline; Ctrl+Enter sends the message.
Alt+1 to Alt+8 pick the candidate in that row; Up, Down and Tab pick the highlighted row. Backspace over ␣ sets
`leading_space` to false. Backspace in an empty box rewinds the last auto-selected token; Esc rewinds to your last
decision. The slider shifts every threshold; the status bar shows backend health, scorer version and metrics.

## Operations

- Rebuild every learner from the log (the log is the source of truth):
  `.venv/bin/python -m humanapi.store.rebuild [--backup DEST] [--forget REQUEST_ID] [--rollback VERSION_ID] [--reset-scorer]`
  A plain rebuild saves a scorer version with the note `rebuild`.
- Replay a corpus through the program (JSON Lines, one `{"messages": [...], "response": "..."}` per line):
  `.venv/bin/python -m humanapi.eval.replay --corpus FILE [--fake] [--out result.json]`.
  `--fake` runs on the CPU with the toy tokenizer, the fake backend and the hash embedder.
- Notes ("Teach it this" in the UI, or the knowledge store) are removed with the forget control; removed chunks
  are never retrieved again.
- Scorer snapshots are kept every `learning.snapshot_every` updates, the newest `learning.snapshots_kept` only.
- Logs and databases live under `data/` (git-ignored): `data/human.db` (the log), `data/capabilities.json`,
  `data/tokenizer/`, `data/latency.json` (integration runs).

## Tests

```sh
.venv/bin/python -m pytest -q -m "not integration and not tokenizer_files"   # unit and property tests, CPU only
.venv/bin/python -m pytest -q -m tokenizer_files                              # needs the tokenizer files
.venv/bin/python -m pytest -q -m integration                                  # needs the real server on 8081
cd ui && npm test                                                             # UI tests (Vitest, jsdom)
```

The plain `pytest -q` runs everything except the integration suite, which is excluded by default.
The integration suite (`tests/integration/test_real_server.py`) probes the real server, checks the tokenizer round
trip, runs an end-to-end OpenAI call through the UI socket, records commit-to-list latency to `data/latency.json`
and checks the GPU embedder's dimension.

## Configuration

Settings come from `config.yaml` (or the file named by the `HUMANAPI_CONFIG` environment variable). Every key has a
default in `humanapi/config.py`; an unknown key is an error.

- `server`: `host`, `port`, `model_id` (what `/v1/models` reports, default `human`), `bearer_token` (optional;
  when set, `/v1/*` requires `Authorization: Bearer <token>`), `max_pending` (requests waiting or active before 503).
- `backend`: `base_url` (the inference server), `model`, `top_k` (at most 20), `timeout_s`,
  `capabilities_file`.
- `tokenizer`: `path`, `enable_thinking` (kept false).
- `embed`: `model`, `device`, `context_tokens`.
- `ui`: `max_rows` (candidate rows shown), `heartbeat_seconds`, `send_key`.
- `gate`: thresholds per group (`tau_content`, `tau_function`, `tau_sentence_start`), `explore_rate`,
  `max_auto_run`, `cold_start_min_labels`, rewind controller settings, `lookahead_max`, `autonomy`.
- `learning`: n-gram orders and smoothing, Adagrad rate, L2 pull to the initial weights, sample weights for typed
  words, kNN `k` and temperature, snapshot settings.
- `knowledge`: retrieval threshold, number of chunks, chunk size and overlap.
- `storage`: `db_path`, `index_dir`.

Sampling fields sent by clients (`temperature`, `top_p`, `seed`, ...) are logged with the request and ignored: the
person chooses the words. Tool definitions are shown in the UI for reference; the response is always text.

## Layout

```
humanapi/        the program (see Architecture)
humanapi/static/ the built UI (written by `npm run build`, git-ignored)
ui/              the React UI (Vite, Vitest)
tools/           the capability probe
tests/           unit, contract, integration suites
docs/spec/       the design excerpts the code follows
```
