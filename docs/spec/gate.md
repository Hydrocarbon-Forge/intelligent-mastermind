# Gate, calibration, provisional buffer, rewind, lookahead (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Gate and calibration
  The gate turns p_you into one of two actions at each position: auto-select a token or ask
  you. It is conservative, because a wrong auto-selection costs more than an extra question.
  Sending a message is a keypress, so the gate never selects END .

     def decide(pos) -> Action:
             p = scorer.p_you(pos)                      # candidates plus OTHER
             top, p_top = best_non_other_non_end(p)
             conf = calibrator.apply(p_top, group_of(top, pos.state))
             tau = thresholds.for_(top, pos.state, slider=settings.autonomy)
             if rand() < cfg.explore_rate:              # exploration ask, logged as
     such
                 return Ask(explore=True)
             return Auto(top) if conf >= tau else Ask

  Thresholds

     Group                                             Config key           Initial   Reason
                                                                            value

     Content words                                     tau_content          0.80      A wrong word derails the
                                                                                      sentence

     Punctuation and mid-                              tau_function         0.60      Cheap to fix, and the base
     word continuations                                                               model is rarely wrong

     Sentence starts                                   tau_sentence_start   0.85      Errors compound from
                                                                                      here

  The autonomy slider shifts every threshold in logit space: a setting of δ turns τ into
   sigmoid(logit(τ) + δ) . The slider range is about plus or minus 2.

  Cold start
  Until a group has cold_start_min_labels labeled positions, the gate auto-selects only
   CONTINUATION tokens with a base probability of at least 0.95. Everything else asks.

  Calibration
  An isotonic (monotonic) fit maps p_top to the measured chance that the top candidate
  was your choice, one fit per threshold group. Positions are only asked when confidence is
  low, so asked positions alone say nothing about the confident range.

        Exploration asks fix this. At rate ε ( explore_rate ) the gate asks even when it would
        auto-select. Each such label is weighted 1 / ε , so it stands in for the confident
        positions that were not sampled.
        Asked positions below τ are weighted 1.
        Auto-selected tokens you did not rewind are not labels, because not noticing an error
        is not the same as approving it.
        The fit is refreshed every 50 new weighted labels, and each fit is stored with its date in
        scorer_versions .

  Exploration positions look like normal asks, so you may anchor on the top candidate. The
  log records the order shown, and the small bias is accepted in exchange for unbiased
  labels in the confident range.

  Provisional buffer and rewind
  Tokens already streamed to the client cannot be retracted, so auto-selected tokens are
  provisional until you make your next decision:

   1. Tokens chosen automatically since your last decision sit in a provisional buffer. They
      are shown to you but not streamed to the client.
   2. When you pick, type or send, the buffer is flushed to the client together with that
        decision.
   3. A rewind inside the buffer truncates it back to the chosen position, logs the rewind
      with the rejected token, and forces an ask there.

  This bounds streaming delay by your own decision rate and makes rewind free inside a
  run. max_auto_run forces an ask after that many auto-selected tokens.

  Rewind-rate controller
  A rolling window tracks rewinds per 100 auto-selected tokens. If the rate rises above a
  configured ceiling, every threshold is raised by a small step until the rate recovers. This
  catches calibration drift between refits.

  Speculative lookahead
  One backend request per auto-selected token is slow, and the sampler settings cannot be
  changed to ask the server for a greedy continuation. Lookahead therefore checks tokens
  the human program proposes. When the gate chooses Auto , the driver drafts up to 8
  tokens ahead from the n-gram memory (the most likely next token for each context,
  stopping where the memory has no confident answer) and obtains the server's
  probabilities along that drafted path with one score_continuation call. It runs decide
  at each drafted position and stops at the first position that asks, where the scorer's top
  token differs from the drafted token, or where an END would be chosen. Unused
  positions are discarded. Lookahead runs only when the capability probe reports
   prompt_logprobs .

  Tests

     Test                                              Asserts

      test_thresholds_by_group                         Content words, function tokens and sentence
                                                       starts each use their own threshold

      test_end_never_auto                              END is never auto-selected, whatever its
                                                       confidence

      test_cold_start                                  Before a group has the minimum labels, only
                                                        CONTINUATION tokens with base probability of at
                                                       least 0.95 auto-select

      test_slider_logit_shift                          The slider shifts thresholds by the logit formula

      test_explore_rate                                Over many decisions with a seeded generator,
                                                       exploration asks occur at ε within a binomial
                                                       tolerance

      test_isotonic_monotonic                          The calibration fit is nondecreasing

      test_importance_weights                          Exploration labels carry weight 1 / ε , and
                                                       unrewound auto-selected tokens carry none

      test_provisional_not_streamed                    Auto-selected tokens are not streamed until the
                                                       next human decision

      test_rewind_in_buffer                            A rewind inside the buffer streams nothing, logs
                                                       the rejected token and forces an ask

     Test                                                 Asserts

      test_rewind_controller                              A rewind rate above the ceiling raises thresholds
                                                          step by step and relaxes them when it recovers

      test_max_auto_run                                   An ask is forced after max_auto_run auto-
                                                          selected tokens

      test_lookahead_draft_from_ngram                     Drafts come from the n-gram memory and stop
                                                          where it has no confident answer

      test_lookahead_stops                                Lookahead stops at the first ask, where the
                                                          scorer's top token differs from the drafted one and
                                                          at END , and discards unused positions

      test_lookahead_disabled                             Without prompt_logprobs , each auto-selected
                                                          token costs one backend call
```
