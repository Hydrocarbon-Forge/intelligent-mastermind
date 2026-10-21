# Candidates and the log-linear scorer (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Candidates and the scorer
  The server returns no hidden state, so the user model is a log-linear scorer over
  candidates. Each candidate token gets a feature vector, a weight vector turns it into a
  score, and a softmax over the candidates plus an "other" class gives p_you . It replaces
  the design doc's low-rank head. It starts equal to the base model and updates in
  microseconds after each label, so it needs no GPU.

  Candidate sources
   1. The backend's top-k tokens, each with its base logprob.
   2. The top tokens from the n-gram memory for the current context (what you chose after
        the same last tokens).
   3. Tokens chosen by the nearest kNN neighbors.
   4. Vocabulary matches for the partial word you have typed.
   5. The END token, always.
   6. A pseudo-candidate OTHER , meaning "you will type something that is not listed."

  A candidate that did not come from the backend has an unknown base probability. It gets
  the lowest returned probability as an upper bound and the base_unknown flag, so the
  scorer can learn how far to trust such tokens.

  Features per candidate

     Feature                                Definition

      log_p_base                            Backend logprob, or the bound when unknown

      base_unknown                          1 when the candidate was not in the backend's top-
                                            k

     Feature                                Definition

      base_rank                             Rank in the backend list, capped at k + 1

      ng1 , ng2 , ng3                       Log of the smoothed n-gram probability of this
                                            token given the last 0, 1 or 2 response tokens, with
                                            backoff

      knn_vote                              Similarity-weighted share of neighbors that chose
                                            this token

      user_unigram                          Log frequency of the token among your labeled
                                            tokens, smoothed

      class_x_state                         One-hot of token class crossed with position state
                                            (sentence start, mid-sentence, after punctuation,
                                            mid-word)

      recent                                1 if the token appeared in your last 20 response
                                            tokens

      prefix_match                          1 if the token matches the partial word you typed

      retrieval_overlap                     1 if the token occurs in a retrieved knowledge
                                            chunk

      end_x_length                          For END only: a bucket of response length so far

   OTHER uses its own feature slots: log(max(tail_mass, tail_floor)) , base entropy, the
  highest kNN similarity, the longest n-gram order seen in this context, and a bias. The floor
  matters because the server's fixed sampler settings ( top_p 0.80, top_k 20) may already
  have removed the tail from the reported probabilities, leaving tail_mass near 0. Low
  similarity and unseen contexts should raise the chance that you type something new.

  Model

     s_c = w \cdot \phi_c, \qquad p_{you}(c) = \frac{e^{s_c}}{\sum_{j} e^{s_j}}

  The weights start at w0 , which puts weight 1 on log_p_base and on log(max(tail_mass,
  tail_floor)) for OTHER , and 0 everywhere else. At that point p_you equals the base
  model's distribution over what the server returned, plus the unseen tail (or tail_floor
  when the server's probabilities are already filtered). Learning moves weights away from
   w0 only when your labels say so.

  Online update
  After each labeled position, with chosen candidate y and sample weight α , take one
  step on the regularized loss. The chosen token is added to the candidate set first if it was
  missing, so it always has features.

     \nabla = \alpha\Big(\sum_j p_j \phi_j - \phi_y\Big) + \lambda (w - w_0)

        Use Adagrad-style per-weight step sizes. λ is learning.l2_to_init and pulls
        weights back toward w0 .
        Sample weights: weight_typed_first (1.0) for a pick or the first token of a typed
        word, weight_typed_continuation (0.3) for the other tokens of a typed word, and 1.0
        for an exploration ask.
        Rewind: the token you chose instead is a normal label, and the wrongly auto-selected
        token adds a penalty term -log(1 - p_bad) .
        Auto-selected tokens that were not rewound never update the weights.
        Updates run on a background task after the commit, so they never delay the next list.

  Running metrics
  The model is evaluated prequentially: the prediction made before each update is logged,
  then the update happens. These pre-update predictions give a running accuracy and
  calibration curve that is never contaminated by training on the example being scored.

  Snapshots
  The weights are saved every N updates ( scorer_versions ), and the last ten are kept. A
   rollback command restores a snapshot, and the scorer can always be rebuilt by
  replaying the log from w0 .

  Tests

     Test                                              Asserts

      test_initial_equals_base                         With weights at w0 , p_you equals the backend
                                                       probabilities when tail_mass is at least
                                                        tail_floor , and is renormalized with
                                                        tail_floor otherwise (property test over
                                                       random distributions)

     Test                                              Asserts

      test_filtered_tail_floor                         With filtered_probs true, OTHER starts near
                                                       tail_floor and rises as labels show you typing
                                                       unlisted tokens

      test_gradient_numeric                            The analytic gradient matches finite differences

      test_update_raises_chosen                        One update raises the chosen token's probability
                                                       and lowers the others'

      test_l2_pulls_to_init                            With no informative labels, weights decay
                                                       toward w0

      test_missing_chosen_added                        A chosen token absent from the candidates is
                                                       added with base_unknown set before the update

      test_unknown_base_bound                          An unknown base probability equals the lowest
                                                       returned probability and sets the flag

      test_sources_union                               Candidates from the top-k, n-gram, kNN, prefix
                                                       matches and END merge without duplicates, and
                                                       END is always present

      test_sample_weights                              Typed continuation tokens use 0.3, and picks and
                                                       first typed tokens use 1.0

      test_rewind_penalty                              A rewind adds the -log(1 - p_bad) term and a
                                                       normal label for the replacement token

      test_auto_never_updates                          Auto-selected tokens that were not rewound
                                                       leave the weights unchanged

      test_prequential_logged                          The pre-update probability is logged before the
                                                       weights change

      test_snapshot_rollback                           A snapshot restores the weights exactly, and
                                                       only ten are kept

      test_replay_determinism                          The same labeled positions in the same order
                                                       produce identical weights

      test_recent_feature_learns_penalty               When the fake backend applies a presence
                                                       penalty to tokens already in the context, the
                                                       weight on recent becomes positive and
                                                       precision on repeated tokens recovers

```
