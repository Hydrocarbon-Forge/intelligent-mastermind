# Tokenizer, prompt rendering, typed-text commit (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Tokenizer and prompt rendering
  The human program loads the Qwen tokenizer files itself. It needs exact ids to log labels,
  to build prompts and to turn typed text into tokens. text/tokenizer.py wraps the
  tokenizer in one class, and text/toy_tokenizer.py provides a tiny byte-level tokenizer
  with the same interface so logic tests run without any files.

     class HumanTokenizer:
             def encode(self, text: str) -> list[int]: ...
             def decode(self, ids: list[int]) -> str: ...
         def render_chat(self, messages: list[dict], knowledge: str | None) ->
     list[int]: ...
         def display(self, token_id: int) -> str: ...                         # visible markers
     for the UI
             def token_class(self, token_id: int) -> TokenClass: ...
             def bytes_to_ids(self, raw: bytes) -> list[int]: ...    # for backends
     without ids
         eos_ids: set[int]                                                    # derived from
     the tokenizer files

  Prompt rendering
   render_chat builds the prompt the model sees, and it is the only place a prompt is made.

        It applies the tokenizer's own chat template with add_generation_prompt on and
         enable_thinking set to false, which is how the model card documents a direct, non-
        thinking response. The model therefore continues straight into the reply instead of
        opening a reasoning block.
        Reasoning fields in client messages ( reasoning_content , reasoning ) are dropped.
        Non-text content parts become the placeholder text [image] .
        Retrieved knowledge, when present, is appended to the system message, or becomes
        a system message when the client sent none.
        The result is encoded to ids once and stored with the request. Those ids are the fixed
        prompt for every position.
         eos_ids is read from the tokenizer files (the eos_token and the chat template's end-
        of-turn token), never hardcoded.

  Token classes
  The gate and the scorer key on the kind of token at a position, so the class is computed
  once per vocabulary entry and cached.

     Class                             Examples         Used for

      WORD_START                        ␣yeah , ␣the    Content and function words

      CONTINUATION                      believ , able   Mid-word, looser threshold

     Class                             Examples                   Used for

      PUNCT                             , . ! 't                  Looser threshold, habit features

      NEWLINE / SPACE                   ↵ , ␣␣                    Layout habits

      DIGIT                             2 , ␣42                   Numbers

      END                              end-of-turn, end-of-text   The "stop here" candidate

      FRAGMENT                         partial UTF-8 bytes        Shown as <0xNN> , never a typed target

  Display rules
   display() decodes through the tokenizer, never from raw vocabulary strings. It replaces
  a leading space with ␣ , a newline with ↵ , a tab with ⇥ , and a run of spaces with one ␣
  per space. Incomplete byte sequences render as <0xNN> . A token with and without a
  leading space stays two different candidates.

  Turning typed text into tokens
  The response is tokenized on its own, apart from the prompt, so merges never cross the
  prompt boundary. This matches how the model generates it.

     def commit_typed(committed_text, committed_ids, typed, leading_space):
             typed = (" " if leading_space else "") + typed
             new_text = committed_text + typed
             new_ids = tok.encode(new_text)                                   # response only, canonical
             k = longest_common_prefix(committed_ids, new_ids)
             return k, new_ids[k:]                     # drop committed_ids[k:], append new_ids[k:]

        The UI sends leading_space with the typed word. The server proposes a default
        ( space_default in the ask message): true unless the response is empty, the text so
        far ends in whitespace, an open bracket or an open quote, or the typed text starts with
        closing punctuation. Backspacing over the shown ␣ in the UI sets it to false.
        Trailing whitespace stays pending. A typed trailing space is not committed on its own,
        because a standalone space token is out of distribution. The next word absorbs it as its
        leading space, and a trailing space at the end of a message is dropped.
        The longest-common-prefix step handles canonical retokenization. When added text
        changes how the previous token splits, that tail is replaced and only the new tokens
        get labels.

        The partial word you are typing is never sent to the server, so the server predicts from
        the committed text only. The UI filters candidates by the typed prefix, and a trie over
        the vocabulary supplies tokens that start with it when none of the candidates do.
        Pressing Send is logged as choosing the END token at that position.
        Streamed text goes through an incremental UTF-8 decoder, so a character split across
        tokens is held back until it is complete.

  Tests

     Test                                              Asserts

      test_display_markers                             Space, newline, tab, double space and byte
                                                       fragments render as specified

      test_roundtrip_property                          For random Unicode strings (including emoji and
                                                       mixed scripts), decode(encode(s)) == s

      test_token_class_total                           Every vocabulary id has exactly one class

      test_bytes_to_ids                                For a sample of tokens, bytes_to_ids returns a
                                                       list containing the token's id

      test_commit_first_word                           The first word of a response gets no leading
                                                       space by default

      test_commit_after_punctuation                    Closing punctuation takes no leading space, and
                                                       a new word after it does

      test_commit_leading_space_override               leading_space false produces the no-space
                                                       token variant

      test_commit_trailing_space_pending               A trailing space is held, absorbed by the next
                                                       word, and dropped at the end of a message

      test_commit_resplit                              When added text re-splits the previous token,
                                                       only the tail from the common prefix on is
                                                       replaced

      test_commit_multi_token_word                     A word of several tokens produces all of them in
                                                       one result

      test_prefix_filter                               The typed prefix narrows candidates, and the
                                                       vocabulary trie supplies matches when none are
                                                       listed

     Test                                                         Asserts

      test_incremental_utf8                                       A multi-byte character split across tokens is
                                                                  emitted only when complete

      test_render_chat_no_open_think                              With real tokenizer files, the rendered prompt
                                                                  does not end inside an unclosed reasoning block

      test_render_chat_matches_template                           With real tokenizer files, the output equals the
                                                                  tokenizer's own template with thinking disabled

      test_render_chat_drops_reasoning                            Reasoning fields are absent from the prompt,
                                                                  and image parts become [image]

      test_render_chat_knowledge                                  Retrieved knowledge lands in the system
                                                                  message, creating one when needed

      test_eos_from_files                                         eos_ids is non-empty and every id is in the
                                                                  vocabulary

  The tests that use real tokenizer files are marked tokenizer_files and skip with a clear
  message when tokenizer.path has no files. Everything else runs against the toy
  tokenizer.
```
