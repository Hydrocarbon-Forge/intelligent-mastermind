# Design plan (plan.pdf, extracted as text)

Background and rationale. Where it differs from the other spec files, the other spec files win (see docs/spec/risks.md, "Differences from the design doc").

```text
Human-in-the-Loop OpenAI-Compatible
  API: Design Plan
    Oct 21, 2026        · ​@Nathan Smith

  Overview and goals
  The goal is a local OpenAI-compatible API whose "model" is you: requests land in a
  response window and you answer them. Over time a small learned assistant types the
  predictable parts for you, so you are only asked at the points where your input actually
  matters.

       Endpoint: /v1/chat/completions (streaming and tool calls) and /v1/models , so any
       OpenAI client works unchanged.
       Manual mode: you write every response yourself. This is the baseline, and it produces
       the first training data.
       Assisted mode: a frozen local LLM proposes next tokens, you pick or type your own,
       and a learned model of your choices auto-selects tokens it is confident you would pick.
       Success criterion: the share of auto-selected tokens rises over time while your
       unedited-accept rate stays high. In a blind test you should be unable to tell your real
       replies from the system's.
       Non-goal: a local model that replaces you. Every real decision stays yours; the system
       only learns where you do not need to be asked.
       Deferred: LoRA fine-tuning of the base model. The LoRA parallels section shows which
       of its techniques this design already reuses.

  Core concept: you are the verifier
  Assisted mode is speculative decoding with a human as the target model. In ordinary
  speculative decoding a small model drafts tokens and a large model verifies them,
  stepping in only when the draft is wrong. Here a frozen base LLM drafts, a learned model
  of your preferences scores the drafts, and you are consulted only when that score is
  uncertain.

  Three layers decide what gets written, and each has a different job:

     Layer                 Role                                            Changes over time?

     Base LLM              Language competence and a candidate             No, frozen
                           distribution at every position

     User                  Learns which candidate you would pick in this   Yes, updated after every
     model                 context                                         pick

     You                   Ground truth for every real decision            n/a

  The organizing principle is that everything except you is a predictor that must earn the
  right to speak for you, through calibrated confidence. Training this is still ordinary
  language-model training (cross-entropy on the next token of text you wrote). The
  difference is that only a small add-on learns while the base model stays untouched.

  System architecture

  architecture · 8 components

  The human program (port 8080) is an ordinary client of a separate OpenAI-compatible
  server (port 8081), so it never touches inference internals, and its loop runs once per
  token position. Every outcome lands in the log, and the kNN index, head and calibration
  can all be rebuilt from it.

  Two programs and what crosses the boundary
  The human program and the inference server are separate programs that talk only
  through the standard OpenAI-compatible API. Clients call the human program on port
  8080. The human program in turn calls an inference server on port 8081, such as
  llama.cpp's server, which stays unmodified and unaware of the human program. This
  keeps llama.cpp integration out of the human program entirely, and the server can be
  swapped for any other OpenAI-compatible one.

  The cost is that the human program only gets what the standard API returns. Each need
  maps like this:

     Need                                       Standard-API route              Caveat

     Candidate tokens and                       Completion request with         The number of alternatives is
     probabilities                              logprobs and top-k              capped, and the cap varies by
                                                alternatives                    server. Sampling filters (top-p,
                                                                                min-p, temperature) are
                                                                                applied locally to the returned
                                                                                list

     Tokenization                               The model's tokenizer, loaded   Must match the server's model
                                                locally in the human program    exactly. A server's own tokenize
                                                                                route would be server-specific

     Scoring a typed word                       Send prompt plus typed text     Support varies by server. If
                                                and ask for prompt logprobs     missing, fall back to one
                                                (echo)                          request per token, or skip base
                                                                                logprobs for typed tokens (the
                                                                                labels are still your text)

     KV cache reuse                             The server's prefix caching,    Depends on the server. The
                                                since each request resends      human program just keeps
                                                the same prefix                 committed text stable

     Hidden state h                             Not part of the API             See the next subsection

  What replaces h
  The user model was designed around the base model's final hidden state h , which a
  standard API does not return. Everywhere this doc says h , read "the position's feature
  vector." Two sources are available:

   1. A context embedding, from a separate embedding model run inside the human
       program (or an embeddings route on the server), used by the kNN index and the head.
   2. Features from the logprobs themselves: top-k probabilities, entropy, rank, the
      candidate tokens, and recent-token features. These feed the gate directly.

  Start with both. The consequences are real and worth knowing up front:

       Candidates are the server's top-k, plus tokens that kNN neighbors picked or that you
       typed. A token outside the top-k can still be a candidate, scored by memory alone.

       Training loss is computed over the returned top-k, the picked token and an "other"
       bucket, not the full vocabulary.
       The head works on an embedding rather than the base model's internal state, so it can
       capture less. If that proves limiting, a thin server-side extension that exposes h is a
       later option, but it is not needed for v1.

  The per-position loop

  per-position loop · 1 decision, 1 loop back

  The loop ends when an end-of-message token is selected or picked. Only the path
  through Ask you produces labels; auto-selected tokens are never used for training.

  Worked example
  You are replying to "want to grab lunch tomorrow?" and have typed "Sure,". The list at the
  next position might look like this (illustrative numbers):

     Token               p_you

      ␣yeah              0.46

      ␣that              0.22

     Token               p_you

      ␣let               0.15

      ␣sounds            0.09

     other               0.08

  Nothing clears 0.8, so you are asked, and you pick ␣sounds . At the next position ␣good
  comes back at 0.91 and ! at 0.88, so both auto-select. After a few hundred responses,
  "Sure, sounds good!" may run on its own once you choose the one word that sets the tone,
  and you are only asked at the fork where you decide how enthusiastic to be.

  Candidate generation
  At each position the list you see is the output of the model's softmax, narrowed by
  sampling filters. The numbers are probabilities, not logits: logits are the raw scores for
  every vocabulary entry (roughly 150k), and softmax turns them into probabilities that sum
  to 1. Every entry in the list is a real token, shown as text.

  The filters decide how many tokens survive, so the list length varies and is itself a signal:

     Filter                   What it does                           Effect on list length

     Temperature              Reshapes the distribution before       Lower leaves fewer survivors,
                              filtering                              higher leaves more

     top-k                    Keeps at most k tokens                 Hard cap

     top-p                    Keeps the smallest set whose           1 token when the model is sure,
                              probabilities sum to p                 many when unsure

     min-p                    Drops tokens below a fraction of the   Trims the long tail
                              top token's probability

  Because you are the sampler, the filters only decide what is worth showing, not what gets
  randomly drawn. That changes the settings:

       Use a wide net. Start around top-p 0.95 to 0.98, min-p around 0.01 and temperature
       near 1.0, so your real choice is rarely cut. These are starting points to tune, not fixed
       values.

       Filter the corrected distribution. Apply the filters to the user model's p_you (see The
       user model), not the base probabilities. Otherwise a token you favor but the base
       model underrates would never appear.
       Cap the display (for example 8 rows) and show whatever survives. If a single token
       survives there is nothing to ask, which gives auto-select for free.
       Do not feed the gate renormalized numbers. After filtering, a lone survivor
       renormalizes to 1.0 even if its raw probability was 0.6. The gate gets raw probabilities
       and pre-filter entropy as features, so it can tell genuine certainty from everything else
       being filtered away.

  Token-level UI and input handling
  The interface works at the token level, with one forward pass per position. Whole-word
  assembly was considered and rejected for v1: it needs two or three batched passes and
  cache forking at every ask, for what is mostly a display nicety. You type the word you have
  in mind, and the system turns it into tokens.

  The list
  Every token is shown exactly as the tokenizer would emit it, with whitespace made
  visible:

     Marker            Meaning

      ␣the             Token with a leading space (a new word)

      the              No leading space: a continuation, or a start after a
                       newline or open quote

      ↵                Newline

      ⇥                Tab

      ␣␣               Double space

      <0xF0>           Incomplete byte sequence that does not decode to
                       valid text

  Tokens are decoded through the tokenizer, never rendered from raw vocabulary strings
  (byte-level BPE stores a space as Ġ or similar). ␣I and I are different tokens with
  different probabilities, and both appear only when both have real mass.

  Typing your own word
   1. You type the word. Whatever you type is your intent, and the list can narrow to tokens
      matching it.
   2. A leading space is inserted automatically and shown as ␣ , unless the context ends in a
       newline, open quote or bracket, or you start with punctuation. Backspace over the ␣
       to override.
   3. When you finish (space, Enter or punctuation), the committed text is tokenized
      canonically, for example ␣un + believ + able .
   4. All of those tokens are committed in one teacher-forced forward pass, which also
      yields the hidden state and logits at every position. The UI can animate the word filling
      in.
   5. The next position asks again, or auto-selects if the gate is confident.

  Picking a list item and typing the same word produce the same label, so there is no
  separate "other" option. Typing is the primary path, and the list works as autocomplete
  with probabilities.

  Rules that keep tokenization consistent
       Re-tokenize text, never splice tokens. Committed text is a string, tokenized
       canonically each time. Reuse the KV cache for the longest common prefix and
       recompute only the differing tail.
       Hold trailing whitespace as pending. A typed trailing space is not committed on its
       own, because a standalone space token is out of distribution. The next word absorbs it
       as its leading space, and a trailing space at the end of a message is dropped.
       Do not strip spaces into a standalone space token. The model was trained with the
       space merged into the next word's token, so separate space tokens degrade every
       prediction.
       Hold the partial word out of the context (token healing). If you have typed pr , the
       model should predict words that start with pr , not treat pr as a finished token.
       Never force a continuation. Nothing locks believ and able after ␣un . At every
       position the full distribution is live, so "unhappy" remains possible on another day.

  Off-list picks
  A word missing from the list was cut by the filters, not deleted from the model. It still has a
  probability in the full distribution, so an off-list pick is the most informative label you can
  give: the training loss is computed over the full vocabulary, and a large gap between the
  base probability and your choice produces a large update. If the typed word is several
  tokens, each token becomes its own label via the single teacher-forced pass.

  Optional live feedback
  While you type, run the partial text through one teacher-forced pass on a pause and show
  its probability (for example "your word: p = 0.002"). This is one pass per pause, not per
  keystroke.

  The user model
  The user model predicts which token you would pick, and it reuses the base LLM instead
  of adding a second transformer. The base model has already read the context, and its final
  hidden state h at the current position summarizes it. The learned part is a small
  correction on top of the base probabilities, scored only over the candidate tokens plus an
  "other" class:

     p_{you}(w \mid ctx) \;\propto\; p_{base}(w \mid ctx)\cdot \exp\big(f(h,
     w)\big)

  A from-scratch transformer would need far more data than you will ever generate and
  would be slow to update, so it is not used.

  Three ways to build the correction f

     Approach                   How it works            Update cost           Notes

     kNN over                   Store (h, token you     One index write, no   Learns instantly and is
     past picks                 picked) for every       gradient              easy to debug ("here are
                                decision. At a new                            the 5 moments this
                                position, find the                            resembles"). Start here
                                nearest stored states
                                and see what you
                                picked there

     Approach                   How it works              Update cost          Notes

     Low-rank                    f(h, w) = e_wᵀ · U ·     A few milliseconds   Weight-decayed toward
     output head                h , where e_w is the      per gradient step    zero, so it starts equal to
                                token embedding and                            the base model and only
                                 U is a rank-r matrix                          drifts as evidence
                                                                               accumulates

     Gate-level                 A tiny model that takes   Milliseconds         Feeds the gate, see The
     mixing                     kNN and head outputs                           gate
                                as features and learns
                                how much to trust each

  Training details
       Loss over the full vocabulary. Cross-entropy on your picked token against
       softmax(base_logits + f) . The head is cheap even at 150k tokens because U·h is a
       small vector and the embedding matrix is already in memory.
       Update after every active pick, plus all tokens of a typed word from its single teacher-
       forced pass.
       Candidates come from the corrected distribution. Compute p_you over the full
       vocabulary, then apply the filters. Tokens that nearby kNN neighbors picked become
       candidates even if the base model gave them almost no mass.
       The head and kNN are rebuildable caches. The interaction log is the source of truth,
       so either can be recomputed from it.

  Limit of the frozen hidden state
  Because h is fixed, the correction can only reweight features the base model already
  computes. If the base model's h does not distinguish two situations you treat differently,
  the head cannot either. Deep habits that need the model to notice different things would
  require adapting internal layers, which is the deferred LoRA step.

  The gate
  The gate decides at each position whether to auto-select a token or ask you. The rule is
  simple: if the top candidate's calibrated p_you exceeds a threshold τ (start at 0.8),
  append it and move on without showing the UI; otherwise show the list. With τ above 0.5

  at most one candidate can pass, so the rule is just "top choice above τ, else ask." Auto-run
  stops at the first position that falls below τ.

  Gate inputs
       Raw base probability of the top token and pre-filter entropy (not renormalized values)
        p_you from the head and the kNN agreement and neighbor distance
       Whether the position is mid-word, after punctuation, or at a sentence start
       The "other" mass, meaning the probability you would type something not on the list
       Retrieval similarity, once the knowledge store exists

  Thresholds by token class

     Token class                                       Threshold         Reason

     Mid-word continuations and pure                   Looser, around    A wrong comma is cheap to fix and
     punctuation                                       0.6               the base model is nearly always right

     Content words                                     0.8 or stricter   A wrong content word derails the
                                                                         rest of the sentence

     Sentence starts                                   Stricter          Errors compound from here

  These are starting values. Expose τ as an autonomy slider so you can trade effort against
  risk directly.

  Calibration
  Raw softmax output is not an honest probability: these models tend to be overconfident,
  and a small online-trained head more so. Fit a temperature or isotonic curve on your
  recent picks so that "0.8" means you really picked it about 80% of the time, and check it
  with a reliability plot. Picking something outside the displayed list counts as a miss against
  the "other" bucket.

  Safeguards
   1. Exploration asks. Randomly ask on about 5% of confident positions anyway. This is the
       only way to keep collecting calibration data where the model believes it is right.

   2. Rewind key. Auto-run tokens stream visibly. A keypress stops the run, rewinds to that
       point and shows the options. A rewind is a strong negative label for the auto-selected
       token, while silently not interrupting is only a weak positive.
   3. Audit mode. Show a sample of auto-completed responses afterward so you can
       measure drift.
   4. End of message is a candidate. EOS must be a pickable option, so "stop here" is a
      decision you make. Tool-call JSON is highly predictable and should mostly auto-select.

  What keeps asking, on purpose
  Grammar-driven tokens (spaces inside words, apostrophes, the period after a complete
  sentence) stop asking within the first few responses. Your own habits (ending casual
  messages with a period, comma density, ! vs . , lowercase starts) converge over the
  first hundred or so responses. What the gate keeps asking about is the real decisions: yes
  or no, which example, how blunt or enthusiastic, and whether to end the message.

  Data and labels
  Only tokens you actively chose or typed are ground truth. The model's own auto-
  selections are never treated as labels, because training on them would reinforce its
  guesses and slowly pull it away from you. Everything is logged to SQLite, which is the
  source of truth from which every learned component can be rebuilt.

  What gets logged per position
       The context (or a hash and pointer into the conversation)
       The hidden-state vector h and the base logits for the candidates shown
       The candidate list with p_base and p_you for each
       What happened: auto-selected, picked from the list, typed, or rewound
       The final token ids and typed string, plus time spent

  How each outcome is used

     Outcome                                      Training label                  Strength

     You typed a word                             Every token of the word, from   Strong, and the first token is
                                                  one teacher-forced pass         the most informative

     Outcome                                      Training label                  Strength

     You picked from the list                     The chosen token                Strong

     You rewound an auto-                         Negative for that token, plus   Strong
     selected token                               the token you then chose

     Auto-selected and not                        None for the head and kNN;      Weak
     interrupted                                  usable as a weak signal for
                                                  calibration

     Exploration ask on a                         The chosen token, also a        Strong
     confident position                           calibration data point

  The continuation tokens of a word you typed are valid labels because they are your text,
  not the model's guess. They mostly teach spelling and vocabulary, which is cheap and
  harmless.

  Offline bootstrap
  Before touching the UI, run your existing writing (emails, chats, notes, and your manual-
  mode responses) through the base model with teacher forcing. At every position you can
  record whether your actual token was rank 1, 2 or 3, or outside the top-k. That produces
  thousands of labeled examples for the first kNN index, the head and the gate calibration.

  Preference pairs for later
  The log also records what was shown and what you chose, so every rejected draft versus
  your final text is a preference pair. These are not used by the v1 learners, but they are
  exactly the data a later DPO-style LoRA step would want.

  LoRA parallels
  This design is a family member of LoRA, not a separate invention, so most of LoRA's
  practice transfers. The low-rank head is literally a LoRA applied to the output layer: the
  model's unembedding matrix W maps h to logits, and the head replaces it with W + E·U ,
  a frozen weight plus a trainable low-rank delta.

  Where the methods sit

     Method                                What changes                   Update speed       Status here

     kNN over past picks                   Nothing in the model, only a   Instant            v1
                                           memory

     Low-rank output                       Low-rank delta on the          Milliseconds       v1
     head                                  unembedding

     LoRA on top layers                    Internal representations       Minutes,           Deferred
                                                                          batched

     Full fine-tune                        Everything                     Hours, and risky   Not
                                                                                             planned

  LoRA practices to borrow now
       Low rank. Keep the head's rank small (the LoRA norm is single digits to a few tens),
       which limits capacity and overfitting on little data.
       Start at identity. LoRA initializes its delta so the model begins unchanged. Initialize the
       head to zero and weight-decay it toward zero, so it only moves as evidence
       accumulates.
       Held-out validation. Keep the most recent slice of your responses (about 10%) out of
       training and evaluate on it. Never evaluate on data the learner has seen.
       Early stopping with patience. When held-out loss stops improving for a few
       evaluations, stop and keep the best checkpoint. Use this when refitting the head in
       batch.
       Versioned adapters with rollback. Save each head and kNN snapshot with a version,
       so a bad update can be reverted.
       Promotion gate. A new version replaces the live one only if it beats it on held-out pick
       accuracy and calibration.
       Replay data. If the head is ever trained in batch, mix in general text so it does not drift
       into odd behavior outside your usual topics.

  How this differs from a normal LoRA
   1. Frozen hidden state. A LoRA on internal layers changes what the model notices. This
      design only reweights outputs given features the base model already computes. That
       makes it stable and fast, and also limits it (see The user model).
   2. A human is the labeler. The gate decides which data gets collected, so this is closer to
      active learning than to training on a static dataset.
   3. Updates are per-pick. A LoRA is retrained in batches. The head and kNN update after
      every pick, which is what makes the system feel like it is learning as you use it.

  If a real LoRA is added later
  The deferred step would be a scheduled job: supervised fine-tuning on your final
  responses, then preference optimization on the pick and edit pairs, with early stopping
  and the promotion gate above. One consequence is worth planning for now. A new LoRA
  changes h , so every stored hidden state in the kNN index becomes stale, and the head
  and calibration were fit against the old states. After any swap, recompute states from the
  log, retrain the head and refit calibration. Keeping the log as the source of truth and
  treating those components as rebuildable caches makes that routine.

  Knowledge: what it knows
  Knowledge is added through retrieval, not weights. The user model changes what the
  system prefers to say, not what it knows, and a LoRA is good at style but unreliable at
  facts: training on a few thousand tokens of new facts tends to teach the phrasing without
  dependable recall and can increase confident hallucination.

  Retrieval store
  Keep a store of things you know: notes, documents, past messages, and facts you tell it
  directly (for example a "teach it this" box). Chunk and embed them, and at each response
  retrieve the relevant chunks into the context.

       No design changes elsewhere. Retrieved passages change the context, so h changes,
       and the head, kNN and gate work on it as before.
       Instant. A new note is known on the next request.
       Deletable. Removing a chunk removes the knowledge, which matters for "only knows
       what you know."
       A gate signal. Retrieval similarity is a gate feature.

  Approximating "only knows what you know"
  A pretrained model knows far more than you do, and it cannot truly be made to forget. The
  practical approximation is abstention: ground factual answers in your own retrieval store,
  and when a factual question retrieves nothing relevant, the gate escalates to you instead
  of letting the base model improvise from pretraining. A second, softer lever is to rerank or
  bias candidates with a small n-gram or language model trained only on your own text, so
  phrasing leans toward your vocabulary.

  Deferred: knowledge in the weights
  Consolidating knowledge into a LoRA is possible later, but only with care: augment
  documents with paraphrases and generated Q&A so the model learns to answer from
  them, mix in general replay data, verify with a recall test generated from the documents,
  and keep retrieval as the fallback and source of truth. It is best treated like consolidation
  of memory, where retrieval is primary and the weights slowly absorb whatever is retrieved
  most often.

  Evaluation
  The key question is whether the system is becoming more autonomous without becoming
  less you. Track both sides, on a held-out set of your most recent responses and on live
  traffic.

     Metric                                            What it measures                            Healthy direction

     Auto-select rate                                  Share of tokens selected without asking     Up

     Asks per 100 tokens                               How often you are interrupted               Down

     Unedited-accept rate                              Share of auto-selected tokens you did not   Stays high
                                                       rewind

     Rewind rate                                       How often you stop an auto-run              Down

     Calibration error                                 Gap between stated confidence and           Down, near zero
                                                       actual pick rate, from a reliability plot

     Held-out pick accuracy                            How often the top candidate matches         Up
                                                       your real token

     Metric                                            What it measures                              Healthy direction

     Bits contributed per                              Surprisal of your picks under the user        Down as it learns
     response                                          model                                         you

     Blind A/B                                         Can you tell your real reply from the         Cannot tell
                                                       system's

  Bits contributed is the clearest measure of how much of a response is genuinely yours. If
  it falls while the unedited-accept rate holds and the blind test stays passing, the system is
  converging on a model of you that you still sign off on at every real decision. If autonomy
  rises while the blind test fails, it is drifting toward the base model, so raise τ and review
  the exploration asks.

  Drift monitoring relies on the exploration asks and audit mode from The gate: they supply
  fresh, unbiased labels even where the model is confident.

  Pitfalls and risks
  Most risks come from the system learning from its own output or from errors
  compounding within a sentence.

     Risk                                              Why it matters                   Mitigation

     Training on auto-selected                         The model reinforces its own     Only active picks, typed
     tokens                                            guesses and drifts from you      words and rewinds are labels

     Overconfident gate                                Wrong tokens slip through        Calibration on recent picks,
                                                       unasked                          exploration asks, rewind key

     Compounding errors                                One wrong auto-token derails     Stricter τ at sentence starts
                                                       the sentence                     and for content words

     Renormalization inflates                          A lone survivor shows 1.0        Gate uses raw probabilities
     confidence                                        even at raw 0.6                  and pre-filter entropy

     Off-distribution                                  Standalone space tokens          Canonical tokenization, hold
     tokenization                                      degrade predictions              trailing whitespace pending

     Candidate filters cut your                        You fall into typing more        Wide filters on the corrected
     real choice                                       often than needed                distribution

     Risk                                              Why it matters                  Mitigation

     Head overfits a few picks                         A single example overpowers     Low rank, weight decay
                                                       the base model                  toward zero, kNN averaging

     Stale caches after a                              Stored states no longer         Log as source of truth, rebuild
     model change                                      match h                         kNN, head and calibration

     Base model knows more                             It improvises facts you would   Retrieval grounding and
     than you                                          not know                        abstention to you

     Client timeouts                                   Slow human replies break API    Heartbeats and streaming
                                                       clients                         keepalives in the shim

  One early decision affects several of these: whether your main clients are chatty (simple
  Q&A) or agentic (tool calls and long system prompts). Agentic traffic is more repetitive, so
  the gate pays off sooner, but the UI then needs to collapse very large contexts and let you
  emit tool-call JSON.

  Build roadmap

  roadmap · 5 phases + deferred LoRA, each with a gate

  Each gate is a measurable check from the Evaluation section, and a phase is only done
  when its gate passes. The deferred LoRA step stays out of scope until the earlier phases
  work, and would replace the live version only if it wins on held-out data.

  Open decisions and next steps
  The first build needs one serving choice settled; the rest can be decided once data is
  flowing.

       Serving stack. llama.cpp is simpler and faster for top-k logprobs. Decided: the
       inference server is a separate OpenAI-compatible program on port 8081, and the
       human program on port 8080 is its client. Still open: whether the chosen server's
       logprob and prompt-logprob support is enough, and which embedding model supplies
       the feature vector.
       Base model. Choose the local model and size (a Qwen-class model around 27B was
       discussed). Larger models give better candidates but slower passes.

       Client type. Chatty Q&A, agentic with tool calls, or both. This shapes the UI and how
       fast the gate pays off.
       Candidate display cap and starting filters. Confirm the row cap and the wide-net
       filter values.
       Starting thresholds. Confirm τ near 0.8 for content words and a looser value for
       punctuation and mid-word tokens.
       UI platform. Local web UI over WebSocket is assumed. Confirm hotkeys for pick, type,
       rewind and the autonomy slider.
       Bootstrap data. Decide which of your existing writing to use for the offline bootstrap.

  The first concrete step is Milestone 0 in the roadmap: the shim, a manual-mode UI and
  logging, so training data starts accumulating from the first request.
```
