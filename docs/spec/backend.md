# Backend contract and capability probe (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Backend contract
  Every call to port 8081 goes through one interface. The adapter works in token ids and
  returns probabilities, never text, so the rest of the program does not depend on how a
  given server spells its API. Server quirks live in one file, and tests replace the adapter with
  a fake.

     @dataclass
     class TokenLP:
             token_id: int | None                      # None if the server returns no ids
             text: str
             raw: bytes                                # token bytes, used to recover ids locally
             logprob: float

     @dataclass
     class NextTokenResult:
         top: list[TokenLP]                            # sorted descending, at most k entries
             tail_mass: float                          # 1 - sum(exp(lp)) over the returned tokens

     @dataclass
     class Capabilities:
             max_top_k: int                            # never above the server's own top_k of 20
             prompt_as_ids: bool
             returns_ids: bool
             filtered_probs: bool                      # probabilities already filtered by the
     server's top_p / top_k
         prompt_logprobs: bool
             prefix_cache: bool

     class InferenceBackend(Protocol):
         async def health(self) -> bool: ...
         async def next_token(self, prompt_ids: list[int], k: int) ->
     NextTokenResult: ...
             async def score_continuation(
                 self, prompt_ids: list[int], cont_ids: list[int], k: int
         ) -> list[NextTokenResult]: ...                             # one result per continuation
     position

   tail_mass is the probability the server did not show. When the server's probabilities are
  already filtered it is about 0, and the scorer uses tail_floor for the "other" class
  instead.

  Rules the adapter enforces
        No request carries a sampler field. The server applies its own fixed settings (see
        Sampler settings). The adapter never sends, changes or probes them, and takes the
        probabilities as the server reports them under those settings.

        A returned token string is never an identifier. The adapter resolves each entry to an id
        through the id field or through the bytes-to-id table, and drops entries it cannot
        resolve instead of guessing. This also covers the padded output vocabulary of
        Qwen3.8, which has ids without strings.
         k is clamped to max_top_k , which can never exceed the server's top_k of 20. The
        effective k is part of the result so candidate code knows how much it saw.
        Every call has a timeout. A connection error is retried once. Any other failure raises
         BackendUnavailable , and the driver turns that position into a plain ask with no
        candidates instead of failing the request.
        The adapter never asks the server to generate a continuation. It only reads
        probabilities for token sequences the human program supplies.

  Capability probe
   tools/probe_backend.py runs against the real server on the runtime machine and writes
   capabilities.json , which the adapter reads at startup. Each check has a defined
  adapter behavior when it fails:

     Check                                 How it is tested                 Adapter behavior when it fails

     Top-logprob cap                       Request k = 5, 20 and 50 and     Use the verified cap, which cannot
                                           record what comes back or        exceed the server's top_k of 20.
                                           errors                           Candidates lean more on the n-
                                                                            gram and kNN sources

     Prompt as token ids                   Send an id array and compare     Send text, and assert
                                           logprobs with the same text      encode(decode(ids)) == ids
                                           prompt                           before every call

     Ids in the response                   Look for an id on each           Recover ids from raw bytes with
                                           returned entry                   the bytes-to-id table, and drop
                                                                            ambiguous bytes

     Filtered or raw                       Request one token and check      Set filtered_probs . When true,
     probabilities                         whether the returned             tail_mass is treated as about 0
                                           probabilities sum to about 1     and the scorer uses tail_floor
                                           (already filtered by top_p
                                           and top_k ) or to clearly less
                                           (raw)

     Check                                 How it is tested                 Adapter behavior when it fails

     Prompt logprobs                       Request zero new tokens with     score_continuation falls back to
                                           echo and logprobs                one single-token call per position,
                                                                            and speculative lookahead is
                                                                            disabled

     Prefix caching                        Time two requests that share     Cap the context sent to the server
                                           a long prefix

     Determinism                           Repeat an identical request      Calibration and cache keys use a
                                           and compare logprobs             tolerance

     Latency                               Time a one-token request at      Recorded for the performance
                                           about 500 and about 4000         budgets
                                           prompt tokens

  Tests
  Contract tests run OpenAICompatBackend against a stub HTTP server
  ( tests/contract/stub_server.py ) that mimics the OpenAI completions endpoint with
  configurable quirks. Driver tests use the in-process FakeBackend .

     Test                                                       Asserts

      test_next_token_sorted_and_tail                           The list is sorted descending and tail_mass
                                                                equals one minus the sum of the returned
                                                                probabilities, within tolerance

      test_k_clamped_to_cap                                     A request above max_top_k is clamped,
                                                                max_top_k never exceeds 20, and the effective
                                                                k is reported

      test_unresolvable_dropped                                 Entries without a resolvable id are dropped,
                                                                never guessed

      test_bytes_to_id_fallback                                 With returns_ids false, ids come from bytes,
                                                                and ambiguous bytes are logged and dropped

      test_text_prompt_fallback                                 With prompt_as_ids false the prompt is sent
                                                                as text and the round-trip assertion runs

     Test                                              Asserts

      test_no_sampler_fields                           No request body sent to the server, probe
                                                       requests included, contains temperature ,
                                                        top_p , top_k , min_p , presence_penalty ,
                                                        frequency_penalty or repetition_penalty

      test_filtered_probs_detected                     When the returned probabilities sum to about 1
                                                       across at most 20 entries, filtered_probs is
                                                       true and tail_mass reads as about 0. When
                                                       they sum to clearly less, it is false

      test_prompt_logprobs_fallback                    With prompt_logprobs false,
                                                       score_continuation returns one result per
                                                       position using single-token calls

      test_retry_once_on_connection_error              A dropped connection is retried once, and a
                                                       second failure raises BackendUnavailable

      test_timeout_degrades                            A timeout raises BackendUnavailable , and the
                                                       driver asks with no candidates

      test_probe_writes_capabilities                   The probe run against the stub in each quirk
                                                       mode writes the expected JSON
```
