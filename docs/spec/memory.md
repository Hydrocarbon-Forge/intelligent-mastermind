# N-gram, kNN, knowledge store (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Memory components
  Three memories feed the scorer, and each is a cache that can be rebuilt from the log. The
  embedder is an interface with two implementations: SentenceTransformerEmbedder ,
  which runs Qwen/Qwen3-Embedding-0.6B on the GPU and returns 1024-dimensional
  vectors, and HashEmbedder , a deterministic hash-based embedder with no model, used in
  tests.

     Component                  Key                    Stores              Updated             Rebuilt from

     N-gram                     Last 0, 1 or 2         Counts of the       On every label      Labeled
     memory                     response tokens        token you chose                         positions
                                                       next

     kNN index                  Context                (vector, chosen     On every label      Stored vectors
                                embedding              token, weight)                          in the log

     Knowledge                  Chunk                  Text chunks of      When you add        The notes
     store                      embedding              what you know       or remove a         themselves
                                                                           note

  N-gram memory
  This is the cheapest and most useful learner for style. For orders 1 to 3 it counts which
  token you chose after each context of the last 0, 1 or 2 response tokens, using only labeled
  tokens and their sample weights. Probabilities use interpolated backoff with additive
  smoothing, so an unseen two-token context falls back to the one-token context. The first
  position of a response uses a start-of-message marker as its context.

  It makes habits such as how you open messages, how you punctuate and which words you
  favor learnable within the first few dozen responses.

  kNN over context embeddings
        Feature vector. Take the last embed.context_tokens tokens of the conversation tail
        (the latest user message plus your response so far), decode them to text, embed them,
        and normalize the vector.
        Index. An exact inner-product index ( faiss-cpu ) held in memory.
        Query. Retrieve the nearest 16 neighbors. Convert similarities to weights with a
        softmax at a fixed temperature, and set knn_vote[c] to the total weight of neighbors
        that chose token c . The highest similarity feeds the OTHER class.
        Where it runs. kNN features are computed at positions that start a word and reused
        for the continuation tokens inside it, where the n-gram memory and base model
        already do the work.
        Persistence. Each position's vector is stored in the log as float16, so the index rebuilds
        without re-embedding.

  A sentence embedding captures what the conversation is about, not the exact syntactic
  slot of the next token. That is why the n-gram memory sits beside it and why the base
  model's probability stays the main feature.

  Knowledge store
   1. Sources. Pasted notes, a folder of text or markdown files, and a "Teach it this" box in
      the UI.
   2. Chunking. Split into chunks of a few hundred tokens with a small overlap, embed them,
        and store text and vector.
   3. Retrieval. Once at the start of each request, query with the latest user message and a
      few preceding turns. Keep the top few chunks above a minimum similarity and pass
        them to render_chat as knowledge.

   4. Why once. Retrieved text changes the prompt prefix, which invalidates the server's
      prefix cache, so the prompt stays fixed for the whole request.
   5. Forgetting. Deleting a chunk removes it from the database and the index, so removed
        knowledge is never retrieved again.

  Abstention as a gate input
  Retrieval records retrieval_hit (any chunk above the minimum similarity) and
   retrieval_max_sim . At positions that look like factual content (numbers, and capitalized
  word starts), when there is no hit, the gate adds a configurable margin to tau_content .
  The system then asks you more often exactly where it cannot ground its guess.

  Tests

     Test                                              Asserts

      test_ngram_counts_and_backoff                    Counts accumulate per order, and an unseen
                                                       context backs off to a lower order

      test_ngram_start_marker                          The first response position uses the start marker
                                                       context

      test_ngram_ignores_auto                          Auto-selected tokens do not change the counts

      test_knn_vote                                    Neighbors that chose the same token pool their
                                                       weights, and the highest similarity is exposed

      test_knn_word_start_only                         kNN features are computed at word-start
                                                       positions and reused for continuation tokens

      test_knn_rebuild_from_log                        An index rebuilt from stored vectors returns the
                                                       same neighbors as the live one

      test_hash_embedder_deterministic                 The same text always gives the same normalized
                                                       vector

      test_embedder_dimension                          The real embedder returns 1024 dimensions
                                                       (integration suite)

      test_chunking_overlap                            Chunks have the configured size range and overlap

      test_retrieval_threshold                         Only chunks above the minimum similarity are
                                                       returned, capped at the top few

     Test                                              Asserts

      test_retrieval_once_per_request                  The prompt prefix is identical across all positions
                                                       of a request

      test_forget_removes_chunk                        A deleted chunk is gone from the database and
                                                       index and is never retrieved

      test_abstention_margin                           With no retrieval hit, factual-looking positions use
                                                       the raised content threshold
```
