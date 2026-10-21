# Performance budgets, risks, differences from design doc (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Performance budgets and risks
  The budgets below are targets chosen for this design, not measurements.
   test_latency_recorded writes the measured values next to them.

     Budget                                     Target                             Notes

     Commit to next                             Under about 300 ms                 One backend call, feature building
     candidate list                                                                and scoring. Scoring itself should
                                                                                   take a few milliseconds

     Scorer update                              Under 5 ms, off the critical       Runs on a background task
                                                path

     Auto-run pace                              Faster than you can read           Relies on speculative lookahead
                                                                                   when the server supports prompt
                                                                                   logprobs

     kNN memory                                 About 410 MB per 100,000           100,000 vectors of 1024 values at 4
                                                labels as float32                  bytes each; half that if kept as
                                                                                   float16

     Log growth                                 Roughly 4 to 5 KB per              About 450 MB per 100,000
                                                position                           positions, mostly candidate JSON
                                                                                   and the 2 KB stored vector

     Development VM                             Under 8 GB for the whole           No model weights, small corpora,
     memory                                     test run                           small indexes

  Risks

     Risk                      Effect                    Mitigation          Caught by

     Server caps               Few                       Verified cap,         test_k_clamped_to_cap , probe
     top-k low                 candidates,               more weight
                               many                      on n-gram and
                               unknown base              kNN sources
                               probabilities

     Reported                  Tail mass                 Detect                test_no_sampler_fields ,
     probabilities             reads as about            filtering, use a      test_filtered_probs_detected
     are already               0, so tokens              tail floor, never
     filtered                  outside the list          change
                               look                      sampler
                               impossible                settings

     Risk                      Effect                  Mitigation          Caught by

     No token ids              Ambiguous               Bytes-to-id         test_bytes_to_id_fallback
     in responses              token identity          table, drop
                                                       ambiguous
                                                       entries

     Padded                    Ids with no             Unresolvable        test_unresolvable_dropped ,
     output                    string reach            ids are             test_vocab_padding_ids_dropped
     vocabulary                the                     dropped
                               candidates

     Local and                 Wrong                   Round-trip          test_ids_roundtrip_real_server
     server                    prompts,                assertion
     tokenizers                silent quality          before every
     disagree                  loss                    call, tokenizer
                                                       files tied to the
                                                       model

     Prompt ends               The model               Thinking            test_render_chat_no_open_think
     inside a                  starts                  disabled in
     thinking                  reasoning               render_chat
     block                     instead of
                               replying

     Per-position              Auto-run lags           kNN only at         test_knn_word_start_only , latency
     embedding                                         word starts         test
     is too slow

     Long                      A wrong early           max_auto_run        test_max_auto_run
     unattended                token carries
     auto-runs                 far

     Scorer                    Habits that are         L2 to w0 ,          test_l2_pulls_to_init , replay
     overfits a                not real                sample              precision
     few labels                                        weights,
                                                       snapshots

     Calibration               Overconfident           Exploration         test_rewind_controller
     drifts                    gate                    asks, rewind-
     between                                           rate controller
     refits

     Risk                      Effect                   Mitigation          Caught by

     Client                    Request fails            Streaming with      test_stream_golden
     timeout                   on the client            keepalives
     while you                 side
     type

     Many                      Stale requests            max_pending        test_max_pending_503
     queued                    pile up                  and a 503
     requests

     Database                  Training data            WAL mode,           test_backup_restore
     corruption                gone                     scheduled
     or loss                                            backups

     Presence                  Base                     The recent          test_recent_feature_learns_penalty
     penalty                   probabilities            feature and
     lowers the                under-rate               learned
     probability               words you                weights
     of reused                 repeat                   absorb the
     tokens                                             effect

  Differences from the design doc
  Splitting the system into two programs forced several changes to the design doc's
  learning stack, and a few decisions were made since. Everything else carries over
  unchanged.

     Design doc                                        This specification               Reason

     Hidden state h as the                             Context embedding from a         A standard OpenAI-
     context feature                                   GPU embedder plus logprob        compatible API does not
                                                       features                         return hidden states

     Low-rank head f(h, w) on                          Log-linear scorer over           The head needs the hidden
     the unembedding                                   candidate features               state and per-token
                                                                                        embeddings

     Cross-entropy over the full                       Softmax over candidates          Only top-k probabilities are
     vocabulary                                        plus an "other" class            available

     Design doc                                        This specification               Reason

     Candidates from the                               Union of backend top-k, n-       Tokens outside the server's
     corrected full distribution                       gram, kNN and vocabulary         top-k still need a way in
                                                       matches

     Auto-selected tokens                              Provisional buffer, flushed at   Streamed tokens cannot be
     stream as they are chosen                         your next decision               retracted after a rewind

     Sending is one of the gated                       Sending is a keypress and        Ending a message stays a
     decisions                                         the gate never selects END       human action

     Exploration asks                                  Weighted 1 / ε in                Confident positions are
                                                       calibration                      otherwise never labeled

     Retrieval at each response                        Retrieval once per request       Keeps the server's prefix
                                                                                        cache valid

     Persona or style prompt                           None                             The n-gram and kNN
                                                                                        memories carry style

     Prompts in the model's                            Prompts with thinking            The model replies directly
     default mode                                      disabled                         instead of opening a
                                                                                        reasoning block

     Candidate filters (top-k, top-                    None. The server's fixed         The sampler belongs to the
     p, min-p, temperature)                            sampler settings are the         server and is never changed
     applied by this system                            only ones
```
