# Overview, environments, stack, layout, configuration (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Overview and environments
  This document specifies a local OpenAI-compatible API in which a person writes the
  replies. A frozen language model proposes the next tokens, learned memories take over
  the ones that are predictable, and the person decides the rest. It is written test-first: each
  component states its behavior and the tests that define it.

  Two programs run side by side. The inference server on port 8081 is an unmodified
  OpenAI-compatible server (llama.cpp's server) that holds the model. The human program
  on port 8080 is what OpenAI clients call, and it is itself a client of the inference server.

  Environments

                                 Runtime machine                        Development VM

     Hardware                    GPU with room for the 27B model        CPU only, 8 GB RAM
                                 and an embedding model

     Inference                   llama.cpp server on port 8081 with a   None
     server                      Qwen3.8-27B GGUF

     Human                       Real tokenizer, real embedding         Same code with a fake backend, the
     program                     model on the GPU                       real tokenizer files or a toy tokenizer,
                                                                        and a hash embedder

     What runs                   The whole system, the capability       Unit tests, backend contract tests
     there                       probe, the integration suite           against a stub server, UI tests, the
                                                                        replay harness on a synthetic
                                                                        corpus

  No test on the development VM loads a model. What that machine cannot verify is listed
  in Test strategy.

  Base model

     Item                                              Value

     Model                                             Qwen3.8-27B, a dense 27B-parameter model with a
                                                       vision encoder. Only text is used

     Weights                                           GGUF files in williamliao/Qwen3.8-27B-NVFP4-
                                                       GGUF : Qwen3.8-27B-NVFP4-Quality-v2.gguf (16.1
                                                       GB), Qwen3.8-27B-Nvidia-NVFP4.gguf (21.5 GB)
                                                       and Qwen3.8-27B-Unsloth-NVFP4-Q8.gguf (23.2
                                                       GB). The human program does not depend on
                                                       which one is served

     Tokenizer and chat template                       Files from Qwen/Qwen3.8-27B

     Context                                           262,144 tokens natively

     Output vocabulary                                 248,320 entries (padded), so the server can report
                                                       ids that have no tokenizer string. Such ids are
                                                       dropped

     Thinking                                          On by default in the model's template. Every
                                                       prompt this system builds disables it

  Sources: Qwen3.8-27B model card, GGUF repository.

  Decisions
        Sending a message is a keypress (Ctrl+Enter by default). The gate never sends.
        There is no persona text. Style comes from the learned memories, and the prompt to
        the model is the client's messages plus any retrieved knowledge.
        The embedding model is Qwen/Qwen3-Embedding-0.6B, which produces 1024-
        dimensional vectors. It runs inside the human program on the GPU.
        The UI is TypeScript with React and Vite.
        Responses are text. Tool definitions that clients send are shown to you, and no tool
        calls are produced.

  Sampler settings
  The inference server applies these sampling settings itself, and they are fixed. They are
  the model card's values for non-thinking mode, which matches the prompts this system
  builds:

     Parameter                                Value

      temperature                             0.7

      top_p                                   0.80

      top_k                                   20

      min_p                                   0.0

      presence_penalty                        1.5

      repetition_penalty                      1.0

  The human program never sends, overrides or probes any of them: no request to port 8081
  contains a sampler field. Everything that depends on sampling is derived from what the
  server returns under these settings, which has five consequences:

        At most 20 tokens can come back at a position, because the server's own top_k is 20.
        The reported probabilities may already reflect these settings. top_p 0.80 and top_k
        20 may have removed the unseen tail, leaving it at about 0, and temperature 0.7 may
        have sharpened the distribution. The capability probe detects the filtering, and the
        scorer then uses a floor for the "other" class.
         presence_penalty 1.5 may lower the probability of tokens that already appeared in
        the text, which would under-rate words you reuse. The scorer's recent feature lets
        the learned weights absorb this.
        No request can ask for greedy decoding, since that would mean changing
        temperature . Lookahead therefore works by checking tokens the human program
        proposes, not by asking the server to generate.
        The human program has no sampling filters of its own. The candidate list is ranked by
         p_you and capped at ui.max_rows .

  Stack

     Area                         Choice                                 Notes

     Language                     Python 3.11+                           Async throughout the human
                                                                         program

     Web                          FastAPI on uvicorn                     SSE for the OpenAI stream,
     framework                                                           WebSocket for the UI, static files
                                                                         for the built UI

     Backend client                httpx async client                    Connection pooling and timeouts
                                                                         for port 8081

     Tokenizer                     transformers AutoTokenizer ,          Loaded from the Qwen/Qwen3.8-
                                  used only for the tokenizer and chat   27B tokenizer files. No model
                                  template                               weights are loaded

     Embeddings                    sentence-transformers with            On the GPU of the runtime
                                   Qwen/Qwen3-Embedding-0.6B             machine

     Vector index                  faiss-cpu , exact inner-product       Held in CPU memory, which also
                                  index                                  fits the development VM

     Numerics                      numpy                                 Scorer, calibration, n-gram counts

     Storage                      SQLite in WAL mode                     One file for the log, knowledge
                                                                         chunks and scorer snapshots

     UI                           TypeScript, React, Vite                State in Zustand, plain CSS
                                                                         modules, native WebSocket

     Python tests                  pytest , pytest-asyncio ,             Property tests for the numeric and
                                   hypothesis , the official OpenAI      tokenization rules
                                  Python client

     UI tests                     Vitest, Testing Library, jsdom         Run with npm test in ui/

     Inference                    llama.cpp server                       Started separately on port 8081
     server

  Repository layout and configuration
  One repository, one Python package, one TypeScript app. Modules are split by
  responsibility so the backend, tokenizer, embedder and storage can each be replaced by a
  fake in tests.

     humanapi/
       main.py                                     # FastAPI app: routes, static UI, startup and shutdown
       config.py                                   # typed settings loaded from config.yaml
       api/
         openai_routes.py                          # /v1/chat/completions, /v1/models, /health
         ui_ws.py                                  # WebSocket endpoint for the UI
         schemas.py                                # pydantic models for OpenAI payloads and UI messages
       core/
         session.py                                # Request and Position state machine
         driver.py                                 # the per-position loop: query, score, gate, ask or
     commit
         queue.py                                  # pending requests, one active request in the UI
       backend/
         base.py                                   # InferenceBackend protocol and Capabilities
         openai_compat.py                          # httpx client for port 8081
         fake.py                                   # in-process scripted backend for tests
       text/
         tokenizer.py                              # HumanTokenizer around the Qwen tokenizer
         toy_tokenizer.py                          # tiny byte-level tokenizer with the same interface
         prompt.py                                 # chat template rendering and retrieved knowledge
       learn/
         candidates.py                             # candidate union and base-probability bounds
         features.py                               # per-candidate feature vectors
         scorer.py                                 # log-linear scorer with online update
         gate.py                                   # thresholds, calibration, exploration, rewind control
         ngram.py                                  # n-gram memory of your choices
         knn.py                                    # kNN over context embeddings
         embed.py                                  # Embedder protocol, SentenceTransformerEmbedder,
     HashEmbedder
       knowledge/
         store.py                                  # chunks, embeddings, retrieval
       store/
         db.py                                     # SQLite connection and migrations
         schema.sql                                # tables
         rebuild.py                                # rebuild learners from the log
       eval/
         replay.py                                 # replay harness

         metrics.py                                # autonomy, calibration, accuracy
     tools/
       probe_backend.py                            # capability probe, run against the real server
     ui/                                           # TypeScript, React, Vite app (builds into
     humanapi/static)
     tests/
       unit/ contract/                       integration/     data/
     config.yaml

  Running the system
   1. Start the inference server on port 8081 with one of the GGUF files, for example llama-
        server -m /models/Qwen3.8-27B-Nvidia-NVFP4.gguf --port 8081 . The server's
        default port is 8080, so --port is required to keep 8080 free.
   2. Run python -m tools.probe_backend once per server build. It writes
         data/capabilities.json .
   3. Start the human program with uvicorn humanapi.main:app --port 8080 .
   4. Point any OpenAI client at http://localhost:8080/v1 and open
         http://localhost:8080/ for the UI. For UI development, npm run dev in ui/
        proxies to port 8080.

  The human program starts even when the backend is down. The UI shows the backend's
  health, and until the backend answers every position is a plain ask with no candidates.

  Configuration

     server:
       host: 127.0.0.1
       port: 8080
     backend:
       base_url: http://127.0.0.1:8081/v1
       model: qwen3.8-27b          # name sent to the server
       top_k: 20                   # logprob alternatives requested; the server's
     own top_k of 20 limits what comes back
       capabilities_file: data/capabilities.json
     tokenizer:
       path: data/tokenizer        # files from Qwen/Qwen3.8-27B
       enable_thinking: false
     embed:
       model: Qwen/Qwen3-Embedding-0.6B
       device: cuda

         context_tokens: 128
     ui:
       max_rows: 8
         heartbeat_seconds: 15
         send_key: Ctrl+Enter
     gate:
       tau_content: 0.80
         tau_function: 0.60                            # punctuation and mid-word continuations
         tau_sentence_start: 0.85
         explore_rate: 0.05
         max_auto_run: 40                              # forces an ask after this many auto-selected
     tokens
       cold_start_min_labels: 200
       abstention_margin: 0.10                         # added to tau_content when retrieval has no
     hit
     learning:
       ngram_orders: [1, 2, 3]
         l2_to_init: 0.001
         tail_floor: 0.01                              # base mass for OTHER when the server's
     probabilities are filtered
       weight_typed_first: 1.0
       weight_typed_continuation: 0.3
     knowledge:
         min_similarity: 0.35
         top_chunks: 4
         chunk_tokens: 300
         chunk_overlap_tokens: 40
     storage:
       db_path: data/human.db
         index_dir: data/index

  Threshold values, retrieval settings, top_k and the sample weights are initial settings.
  The replay harness is how they get tuned.
```
