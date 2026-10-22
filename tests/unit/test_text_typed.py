from __future__ import annotations

import pytest

from humanapi.text.toy_tokenizer import ToyTokenizer
from humanapi.text.typed import (
    Utf8Stream,
    commit_typed,
    longest_common_prefix,
    position_state,
    space_default,
)
from humanapi.text.types import PosState


@pytest.fixture(scope="module")
def tok() -> ToyTokenizer:
    return ToyTokenizer()


def apply(tok, committed_ids, k, new_ids):
    return list(committed_ids[:k]) + list(new_ids)


def commit(tok, committed_text, typed, leading_space):
    committed_ids = tok.encode(committed_text)
    k, new = commit_typed(tok, committed_text, committed_ids, typed, leading_space)
    ids = apply(tok, committed_ids, k, new)
    return ids, k


def test_longest_common_prefix():
    assert longest_common_prefix([1, 2, 3], [1, 2, 9]) == 2
    assert longest_common_prefix([], [1]) == 0
    assert longest_common_prefix([4], [4]) == 1


def test_space_default():
    assert space_default("", "yeah") is False
    assert space_default("Hi there", "you") is True
    assert space_default("Hi ", "you") is False
    assert space_default("(", "you") is False
    assert space_default('say "', "hi") is False
    assert space_default('say "hi"', "there") is True
    assert space_default("Hi", ",") is False
    assert space_default("Hi", "!") is False
    assert space_default("Hi", "'") is False


def test_commit_first_word(tok):
    ids, _ = commit(tok, "", "yeah", leading_space=False)
    assert tok.decode(ids) == "yeah"
    assert tok.display(ids[0]) == "y"


def test_commit_after_punctuation(tok):
    ids, _ = commit(tok, "Sure.", "yeah", leading_space=True)
    assert tok.decode(ids) == "Sure. yeah"
    assert ids == tok.encode("Sure. yeah")
    ids, _ = commit(tok, "Sure", ",", leading_space=False)
    assert tok.decode(ids) == "Sure,"


def test_commit_leading_space_override(tok):
    with_space, _ = commit(tok, "I", "want", leading_space=True)
    without, _ = commit(tok, "I", "want", leading_space=False)
    assert tok.decode(with_space) == "I want"
    assert tok.decode(without) == "Iwant"
    assert with_space != without


def test_commit_trailing_space_pending(tok):
    committed_ids = tok.encode("I")
    k, new = commit_typed(tok, "I", committed_ids, "want  ", True)
    assert tok.decode(apply(tok, committed_ids, k, new)) == "I want"
    assert commit_typed(tok, "I", committed_ids, "   ", True) == (len(committed_ids), [])


def test_commit_resplit(tok):
    committed_ids = tok.encode("believ")
    assert committed_ids == [tok.bytes_to_ids(b"believ")[0]]
    k, new = commit_typed(tok, "believ", committed_ids, "able", leading_space=False)
    assert k == longest_common_prefix(committed_ids, tok.encode("believable"))
    assert tok.decode(apply(tok, committed_ids, k, new)) == "believable"
    assert apply(tok, committed_ids, k, new) == tok.encode("believable")


def test_commit_multi_token_word(tok):
    ids, _ = commit(tok, "I", "unbelievable", leading_space=True)
    assert tok.decode(ids) == "I unbelievable"
    committed_ids = tok.encode("I")
    k, new = commit_typed(tok, "I", committed_ids, "unbelievable", True)
    assert len(new) >= 3
    assert new == tok.encode("I unbelievable")[k:]


def test_position_state(tok):
    assert position_state(tok, []) == PosState.SENTENCE_START
    assert position_state(tok, tok.encode("Hi.")) == PosState.SENTENCE_START
    assert position_state(tok, tok.encode("Hi\n")) == PosState.SENTENCE_START
    assert position_state(tok, tok.encode("Hi (")) == PosState.AFTER_PUNCT
    assert position_state(tok, tok.encode("Hi the")) == PosState.MID_WORD
    assert position_state(tok, tok.encode("believ")) == PosState.MID_WORD
    assert position_state(tok, tok.encode("n 42")) == PosState.MID_WORD
    assert position_state(tok, tok.encode("Hi  ")) == PosState.MID_SENTENCE


def test_incremental_utf8(tok):
    emoji = list("\U0001F600".encode("utf-8"))
    ids = [tok.bytes_to_ids(bytes([b]))[0] for b in emoji]
    stream = Utf8Stream(tok)
    assert stream.sync(ids[:1]) == ""
    assert stream.sync(ids[:2]) == ""
    assert stream.sync(ids[:3]) == ""
    assert stream.sync(ids) == "\U0001F600"
    assert stream.sync(ids) == ""


def test_utf8_finish_flushes_leftovers(tok):
    ids = [tok.bytes_to_ids(b"\xf0")[0]]
    stream = Utf8Stream(tok)
    assert stream.sync(ids) == ""
    assert stream.finish(ids) == "�"


def test_utf8_no_reemit_after_resplit(tok):
    stream = Utf8Stream(tok)
    a = tok.bytes_to_ids(b"a")[0]
    b = tok.bytes_to_ids(b"b")[0]
    assert stream.sync([a]) == "a"
    assert stream.sync([b]) == ""
    assert stream.sync([a, b]) == "b"


def test_utf8_mixed_script_text(tok):
    text = "naïve 日本 \U0001F600"
    ids = tok.encode(text)
    stream = Utf8Stream(tok)
    collected = ""
    for end in range(1, len(ids) + 1):
        collected += stream.sync(ids[:end])
    collected += stream.finish(ids)
    assert collected == text
