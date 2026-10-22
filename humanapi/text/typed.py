"""Typed-text commit, leading-space default, position state and incremental UTF-8 streaming."""
from __future__ import annotations

from typing import Sequence

from humanapi.text.types import PosState, TokenClass, Tokenizer

_CLOSING_START = (",", ".", "!", "?", ":", ";", ")", "]", "}", "'")
_OPENING_BRACKETS = ("(", "[", "{")
_QUOTES = ('"', "'")
_SENTENCE_END = (".", "!", "?")


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


def space_default(committed_text: str, typed: str) -> bool:
    if not committed_text:
        return False
    last = committed_text[-1]
    if last.isspace() or last in _OPENING_BRACKETS:
        return False
    if last in _QUOTES and committed_text.count(last) % 2 == 1:
        return False
    if typed[:1] in _CLOSING_START and typed:
        return False
    return True


def commit_typed(
    tok: Tokenizer,
    committed_text: str,
    committed_ids: Sequence[int],
    typed: str,
    leading_space: bool,
) -> tuple[int, list[int]]:
    """Return (k, new_ids): the caller drops committed_ids[k:] and appends new_ids."""
    stripped = typed.rstrip(" \t")
    if not stripped:
        return len(committed_ids), []
    addition = (" " if leading_space else "") + stripped
    new_ids = tok.encode(committed_text + addition)
    k = longest_common_prefix(committed_ids, new_ids)
    return k, list(new_ids[k:])


def position_state(tok: Tokenizer, response_ids: Sequence[int]) -> PosState:
    if not response_ids:
        return PosState.SENTENCE_START
    if tok.decode([response_ids[-1]]).endswith("\n"):
        return PosState.SENTENCE_START
    decoded = tok.decode(response_ids).rstrip()
    if decoded and decoded[-1] in _SENTENCE_END:
        return PosState.SENTENCE_START
    token_class = tok.token_class(response_ids[-1])
    if token_class == TokenClass.PUNCT:
        return PosState.AFTER_PUNCT
    if token_class in (TokenClass.WORD_START, TokenClass.CONTINUATION, TokenClass.DIGIT):
        return PosState.MID_WORD
    return PosState.MID_SENTENCE


def _complete_length(data: bytes) -> int:
    """Length of the prefix that does not end inside an incomplete multi-byte UTF-8 sequence."""
    total = len(data)
    for back in range(1, min(4, total) + 1):
        byte = data[total - back]
        if (byte & 0xC0) == 0x80:  # continuation byte, keep looking for the lead
            continue
        if (byte & 0x80) == 0:
            return total
        if (byte & 0xE0) == 0xC0:
            need = 2
        elif (byte & 0xF0) == 0xE0:
            need = 3
        elif (byte & 0xF8) == 0xF0:
            need = 4
        else:
            return total
        return total if need <= back else total - back
    return total


class Utf8Stream:
    """Streams response text by bytes; an incomplete trailing character is held back until complete."""

    def __init__(self, tok: Tokenizer) -> None:
        self._tok = tok
        self._emitted = b""

    def _full_bytes(self, all_ids: Sequence[int]) -> bytes:
        return b"".join(self._tok.token_bytes(token_id) or b"" for token_id in all_ids)

    def sync(self, all_ids: Sequence[int]) -> str:
        full = self._full_bytes(all_ids)
        if not full.startswith(self._emitted):
            return ""
        cut = _complete_length(full)
        if cut < len(self._emitted):
            return ""
        text = full[len(self._emitted) : cut].decode("utf-8", errors="replace")
        self._emitted = full[:cut]
        return text

    def finish(self, all_ids: Sequence[int]) -> str:
        full = self._full_bytes(all_ids)
        if not full.startswith(self._emitted):
            return ""
        text = full[len(self._emitted) :].decode("utf-8", errors="replace")
        self._emitted = full
        return text
