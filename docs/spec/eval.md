# Test strategy, doubles, corpus, replay harness, integration suite (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Test strategy
  The test suite is split by what each test needs, so everything that can run on the CPU-only
  development VM does, and the rest runs on the runtime machine.

  Where each suite runs

     Suite                     Marker or command                    Needs                     Runs on

     Unit and                   pytest -m "not                      Python only               Development VM
     property                  integration and not                                            and runtime
     tests                     tokenizer_files"                                               machine

     Tokenizer                  pytest -m                           The Qwen/Qwen3.8-         Anywhere the files
     file tests                tokenizer_files                      27B tokenizer files in    are present.
                                                                    tokenizer.path            Skipped with a
                                                                                              message otherwise

     Suite                     Marker or command                  Needs                     Runs on

     Contract                  Included in unit tests             The stub HTTP server      Development VM
     tests                                                                                  and runtime
                                                                                            machine

     UI tests                   npm test in ui/                   Node                      Development VM
                                                                                            and runtime
                                                                                            machine

     Replay                     python -m                         A corpus and cached       Synthetic corpus
     harness                   humanapi.eval.replay               backend outputs           on the VM, your
                                                                                            own corpus on the
                                                                                            runtime machine

     Integration                pytest -m integration             The real server on port   Runtime machine
     suite                                                        8081, the GPU             only
                                                                  embedder

  No test on the development VM loads model weights. Tests keep index sizes and corpora
  small so memory stays well under 8 GB.

  Test doubles

     Double                           Replaces                     How it behaves

      FakeBackend                     The inference server, in     A tiny scripted language model with
                                      process                      deterministic distributions. It can inject
                                                                   failures: timeouts, a lower top-k cap,
                                                                   missing ids, no prompt logprobs. It can also
                                                                   apply a presence penalty to tokens already
                                                                   in the context

     Stub HTTP                        The inference server over    A small FastAPI app that mimics the
     server                           HTTP                         OpenAI completions endpoint with
                                                                   configurable quirks, so
                                                                   OpenAICompatBackend is tested as real
                                                                   HTTP code

      ToyTokenizer                    The Qwen tokenizer           A byte-level tokenizer with a few hundred
                                                                   entries and the same interface

     Double                           Replaces              How it behaves

      HashEmbedder                    The embedding model   Deterministic hash-based vectors,
                                                            normalized, no model

     Scripted person                  The human             A test driver that sends pick , typed ,
                                                            rewind and send messages over the
                                                            WebSocket from a script

  Corpus
  The corpus is a JSON Lines file. Each line is one thing you replied to and what you wrote:

     {"messages": [{"role": "user", "content": "want to grab lunch tomorrow?"}],
     "response": "Sure, sounds good!"}

   messages is the conversation up to your reply, in OpenAI message format. response is
  your reply exactly as you wrote it. Any writing of yours with a clear "what I was replying
  to" fits: email or chat threads, notes that answer a question, and the replies the program
  logs itself. The replay harness uses it to measure how often the system would have asked
  and how often its guesses would have been right.

  Tests use a small synthetic corpus checked in at tests/data/corpus_small.jsonl . Your
  own corpus stays on the runtime machine, and no test depends on it.

  Replay harness
   eval/replay.py runs the real candidate assembly, scorer and gate over a corpus, with a
  simulated person who always chooses the true next token (and counts as "typed" when
  that token is not among the candidates).

   1. Split the corpus by time: the first 80% is for learning and the last 20% is frozen for
      evaluation.
   2. For each response, tokenize it and obtain the backend's top-k at every position.
        Results are cached to disk so reruns are free. With a FakeBackend the distributions
        come from the scripted model.
   3. Apply the same update rules as live use, including exploration asks and calibration
        refits.

     Output                                            Reads as

     Learning curve                                    Auto-select rate against the number of responses
                                                       seen

     Precision at the threshold                        Share of auto-selected tokens that matched the
                                                       true token, which should sit at or above the stated
                                                       τ

     Threshold sweep                                   Autonomy against precision for several τ values,
                                                       used to tune the thresholds

     Calibration curve                                 Stated confidence against measured accuracy, per
                                                       threshold group

     Class breakdown                                   Which token classes auto-select and which keep
                                                       asking

  On the synthetic corpus the harness is itself under test:
  test_replay_learning_curve_rises asserts that auto-select rate increases as the
  scripted person repeats habits, test_replay_precision_holds asserts precision stays
  at or above τ on the frozen slice, and test_replay_deterministic asserts that a fixed
  seed reproduces the outputs.

  Integration suite
  These tests need the real server and run on the runtime machine only.

     Test                                                         Asserts

      test_probe_matches_server                                   The probe's capabilities.json agrees with
                                                                  direct requests to the server

      test_probs_match_probe                                      The probe's filtered_probs value agrees with
                                                                  the sum of the probabilities returned on live
                                                                  requests, and no request sent a sampler field

      test_ids_roundtrip_real_server                              Prompt ids sent to the server decode to the same
                                                                  text the local tokenizer produced

      test_vocab_padding_ids_dropped                              Ids without a tokenizer string never reach the
                                                                  candidate list

     Test                                                       Asserts

      test_end_to_end_sdk                                       The OpenAI client gets a reply produced by a
                                                                scripted person through the WebSocket

      test_latency_recorded                                     Commit-to-list latency is measured and written
                                                                next to the budgets

      test_embedder_dimension                                   The GPU embedder returns 1024-dimensional
                                                                vectors

  What the development VM cannot verify

     Behavior                                          Why it is not testable     Where it is verified
                                                       there

     Real logprob behavior of                          No model can run on 8      The capability probe and the
     the llama.cpp server for                          GB of CPU memory           integration suite
     this model

     Quality of the real                               Needs the GPU model        Replay harness on your own
     embedding model for                                                          corpus, on the runtime machine
     kNN

     Latency and throughput                            Depends on the GPU         test_latency_recorded
                                                       and the server build

     Chat template output of                           Needs the tokenizer        tokenizer_files tests wherever
     Qwen3.8                                           files, which can be        the files are copied
                                                       copied to the VM

  The code paths that handle each unverifiable behavior are tested with the stub server
  and FakeBackend failure modes, so a different result on the real server changes
  configuration, not code.
```
