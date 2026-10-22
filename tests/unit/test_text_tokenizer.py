from __future__ import annotations

import copy
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from humanapi.text.base import ByteVocabTokenizer
from humanapi.text.prompt import prepare_messages
from humanapi.text.toy_tokenizer import ToyTokenizer
from humanapi.text.types import TokenClass
from tests.conftest import tokenizer_dir


@pytest.fixture(scope="module")
def tok() -> ToyTokenizer:
    return ToyTokenizer()


def one_id(tok: ToyTokenizer, raw: bytes) -> int:
    ids = tok.bytes_to_ids(raw)
    assert ids, raw
    return ids[0]


def test_display_markers(tok):
    assert tok.display(one_id(tok, b" the")) == "␣the"
    assert tok.display(one_id(tok, b"\n")) == "↵"
    assert tok.display(one_id(tok, b"\t")) == "⇥"
    assert tok.display(one_id(tok, b"  ")) == "␣␣"
    assert tok.display(0xF0) == "<0xF0>"
    assert tok.display(one_id(tok, b" I")) != tok.display(one_id(tok, b"I"))


def test_end_token(tok):
    assert tok.eos_ids == {256}
    assert tok.end_id == 256
    assert tok.token_class(256) == TokenClass.END
    assert tok.display(256) == "<|end|>"
    assert 256 in set(tok.valid_ids())


@settings(max_examples=100, deadline=None)
@given(st.text())
def test_roundtrip_property(text):
    assert tok_for_property.decode(tok_for_property.encode(text)) == text


tok_for_property = ToyTokenizer()


def test_token_class_total(tok):
    valid = list(tok.valid_ids())
    assert valid
    for token_id in valid:
        assert isinstance(tok.token_class(token_id), TokenClass)
    assert tok.token_class(one_id(tok, b" the")) == TokenClass.WORD_START
    assert tok.token_class(one_id(tok, b"believ")) == TokenClass.CONTINUATION
    assert tok.token_class(one_id(tok, b",")) == TokenClass.PUNCT
    assert tok.token_class(one_id(tok, b"'t")) == TokenClass.PUNCT
    assert tok.token_class(one_id(tok, b"42")) == TokenClass.DIGIT
    assert tok.token_class(one_id(tok, b" 42")) == TokenClass.DIGIT
    assert tok.token_class(one_id(tok, b"\n")) == TokenClass.LAYOUT
    assert tok.token_class(0xF0) == TokenClass.FRAGMENT


def test_padding_ids(tok):
    padding = tok.vocab_size - 1
    assert padding not in set(tok.valid_ids())
    assert tok.token_text(padding) is None
    with pytest.raises(KeyError):
        tok.display(padding)
    with pytest.raises(KeyError):
        tok.token_class(padding)


def test_vocab_size_and_pieces(tok):
    assert tok.vocab_size == len(list(tok.valid_ids())) + 8


def test_bytes_to_ids(tok):
    for token_id in list(tok.valid_ids())[::7]:
        assert token_id in tok.bytes_to_ids(tok.token_bytes(token_id))
    assert tok.bytes_to_ids(b"no-such-token-here") == []


def test_encode_canonical(tok):
    assert tok.encode(" the") == [one_id(tok, b" the")]
    assert tok.encode("believable") == [one_id(tok, b"believ"), one_id(tok, b"able")]


def test_tokens_with_prefix(tok):
    with_space = tok.tokens_with_prefix("ok", True)
    assert one_id(tok, b" ok") in with_space
    assert one_id(tok, b" okay") in with_space
    assert one_id(tok, b"believ") in tok.tokens_with_prefix("believ", False)
    assert one_id(tok, b" believ") not in tok.tokens_with_prefix("believ", False)
    assert 256 not in tok.tokens_with_prefix("", False)
    assert tok.tokens_with_prefix("zzqq", False) == []


def test_prompt_drops_reasoning_and_maps_images():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "look "}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "!"}]},
        {"role": "assistant", "content": "ok", "reasoning_content": "secret", "reasoning": "x", "name": "bot"},
        {"role": "user", "content": None},
    ]
    before = copy.deepcopy(messages)
    out = prepare_messages(messages, None)
    assert out == [
        {"role": "user", "content": "look [image]!"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": ""},
    ]
    assert messages == before


def test_render_chat_drops_reasoning(tok):
    ids = tok.render_chat(
        [{"role": "user", "content": "hi", "reasoning": "private"}],
        None,
    )
    text = tok.decode(ids)
    assert "private" not in text
    assert text == "<|user|>\nhi\n<|assistant|>\n"


def test_render_chat_knowledge_into_system(tok):
    out = prepare_messages([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "x"}], "fact")
    assert out[0] == {"role": "system", "content": "Be brief.\n\nRelevant notes:\nfact"}
    out = prepare_messages([{"role": "system", "content": ""}], "fact")
    assert out[0]["content"] == "Relevant notes:\nfact"
    out = prepare_messages([{"role": "user", "content": "x"}], "fact")
    assert out[0] == {"role": "system", "content": "Relevant notes:\nfact"}
    assert len(out) == 2
    rendered = tok.render_chat([{"role": "user", "content": "x"}], "fact")
    assert "Relevant notes:\nfact" in tok.decode(rendered)


@pytest.mark.tokenizer_files
def test_eos_from_files():
    from humanapi.text.tokenizer import HumanTokenizer

    real = HumanTokenizer(str(tokenizer_dir()))
    assert real.eos_ids
    assert all(i in set(real.valid_ids()) for i in real.eos_ids)


@pytest.mark.tokenizer_files
def test_render_chat_matches_template():
    from humanapi.text.tokenizer import HumanTokenizer

    real = HumanTokenizer(str(tokenizer_dir()), enable_thinking=False)
    messages = [{"role": "user", "content": "hello"}]
    text = real._hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert real.render_chat(messages, None) == real._hf.encode(text, add_special_tokens=False)
