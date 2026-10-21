"""Shared text-layer types: token classes, position states and the tokenizer interface.

Implementations: HumanTokenizer (humanapi/text/tokenizer.py) and ToyTokenizer (humanapi/text/toy_tokenizer.py).
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Protocol, Sequence


class TokenClass(str, Enum):
    WORD_START = "word_start"  # leading space + letters: " yeah", " the"
    CONTINUATION = "continuation"  # letters without leading space: "believ", "able"
    PUNCT = "punct"  # punctuation and contraction tails: , . ! 't
    LAYOUT = "layout"  # newline, tab, runs of spaces
    DIGIT = "digit"  # numbers: "2", " 42"
    END = "end"  # end-of-turn, end-of-text
    FRAGMENT = "fragment"  # partial UTF-8 bytes; shown as <0xNN>, never a typed target


class PosState(str, Enum):
    SENTENCE_START = "sentence_start"  # empty response, after a newline, or after . ! ?
    AFTER_PUNCT = "after_punct"  # previous token is other punctuation
    MID_WORD = "mid_word"  # previous token is a word piece or digit
    MID_SENTENCE = "mid_sentence"  # anything else


class Tokenizer(Protocol):
    eos_ids: set[int]  # derived from the tokenizer files, never hardcoded
    end_id: int  # the canonical END candidate id (one member of eos_ids)
    vocab_size: int  # number of ids, including ids that have no string

    def encode(self, text: str) -> list[int]:
        """Canonical tokenization of a response, without any prompt and without special tokens."""
        ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def render_chat(self, messages: list[dict], knowledge: Optional[str]) -> list[int]:
        """The only place a prompt is made. Thinking is disabled; reasoning fields dropped; non-text parts
        become [image]; knowledge is appended to the system message (creating one when absent)."""
        ...

    def display(self, token_id: int) -> str:
        """UI text with visible whitespace markers: leading space -> '␣', newline -> '↵', tab -> '⇥',
        a run of spaces -> one '␣' per space, incomplete bytes -> '<0xNN>'."""
        ...

    def token_class(self, token_id: int) -> TokenClass: ...

    def token_text(self, token_id: int) -> Optional[str]:
        """Decoded text of one token, or None for ids without a string (padding ids) and for fragments."""
        ...

    def token_bytes(self, token_id: int) -> Optional[bytes]:
        """Raw bytes of one token (fragments included), or None for ids without a string."""
        ...

    def bytes_to_ids(self, raw: bytes) -> list[int]:
        """Every id whose token bytes equal raw (usually 0 or 1 ids, more when ambiguous)."""
        ...

    def valid_ids(self) -> Iterable[int]:
        """Ids that have a token string (padding ids excluded)."""
        ...

    def tokens_with_prefix(self, prefix: str, leading_space: bool) -> list[int]:
        """Ids whose decoded text equals (' ' if leading_space else '') + something starting with prefix.
        Backed by a trie over the vocabulary."""
        ...
