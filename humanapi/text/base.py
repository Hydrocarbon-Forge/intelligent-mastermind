"""Shared byte-vocabulary tokenizer base: display markers, token classes and a prefix trie."""
from __future__ import annotations

from typing import Iterable, Optional

from humanapi.text.types import TokenClass

_DISPLAY_MARKERS = {" ": "␣", "\n": "↵", "\t": "⇥"}  # ␣ ↵ ⇥
_DIGITS = frozenset("0123456789")
_APOSTROPHES = ("'", "’")


def _strict_text(raw: bytes) -> Optional[str]:
    """Decode a whole token as UTF-8; None when the bytes are not valid UTF-8 (a fragment)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


class _TrieNode:
    __slots__ = ("children", "ids")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.ids: list[int] = []


class ByteVocabTokenizer:
    """Everything in the Tokenizer protocol except encode, decode and render_chat."""

    eos_ids: set[int]
    end_id: int
    vocab_size: int

    def __init__(
        self,
        vocab_bytes: dict[int, bytes],
        eos_ids: set[int],
        vocab_size: int,
        end_id: int | None = None,
    ) -> None:
        self._bytes: dict[int, bytes] = dict(vocab_bytes)
        self.eos_ids = set(eos_ids)
        self.vocab_size = vocab_size
        self.end_id = min(self.eos_ids) if end_id is None else end_id
        self._by_bytes: dict[bytes, list[int]] = {}
        for token_id in sorted(self._bytes):
            self._by_bytes.setdefault(self._bytes[token_id], []).append(token_id)
        self._classes: dict[int, TokenClass] = {}
        self._trie: Optional[_TrieNode] = None

    # -- display and classes -------------------------------------------------

    def display(self, token_id: int) -> str:
        raw = self._bytes[token_id]  # KeyError for padding ids
        text = _strict_text(raw)
        if text is None:
            return "".join(f"<0x{byte:02X}>" for byte in raw)
        return "".join(_DISPLAY_MARKERS.get(ch, ch) for ch in text)

    def token_class(self, token_id: int) -> TokenClass:
        cached = self._classes.get(token_id)
        if cached is None:
            cached = self._classify(token_id)  # KeyError for padding ids
            self._classes[token_id] = cached
        return cached

    def _classify(self, token_id: int) -> TokenClass:
        raw = self._bytes[token_id]
        if token_id in self.eos_ids:
            return TokenClass.END
        text = _strict_text(raw)
        if text is None:
            return TokenClass.FRAGMENT
        if text.isspace():
            return TokenClass.LAYOUT
        stripped = text.lstrip(" ")
        if stripped and all(ch in _DIGITS for ch in stripped):
            return TokenClass.DIGIT
        if not any(ch.isalnum() for ch in stripped) or stripped.startswith(_APOSTROPHES):
            return TokenClass.PUNCT
        return TokenClass.WORD_START if text.startswith(" ") else TokenClass.CONTINUATION

    # -- lookups -------------------------------------------------------------

    def token_text(self, token_id: int) -> Optional[str]:
        raw = self._bytes.get(token_id)
        if raw is None:
            return None
        return _strict_text(raw)  # None for fragments

    def token_bytes(self, token_id: int) -> Optional[bytes]:
        return self._bytes.get(token_id)

    def bytes_to_ids(self, raw: bytes) -> list[int]:
        return list(self._by_bytes.get(raw, []))

    def valid_ids(self) -> Iterable[int]:
        return sorted(self._bytes)

    def tokens_with_prefix(self, prefix: str, leading_space: bool) -> list[int]:
        if self._trie is None:
            self._trie = self._build_trie()
        target = (" " if leading_space else "") + prefix
        node = self._trie
        for ch in target:
            node = node.children.get(ch)
            if node is None:
                return []
        found: list[int] = []
        stack = [node]
        while stack:
            current = stack.pop()
            found.extend(current.ids)
            stack.extend(current.children.values())
        return sorted(found)

    def _build_trie(self) -> _TrieNode:
        root = _TrieNode()
        for token_id in sorted(self._bytes):
            if token_id in self.eos_ids:
                continue
            text = self.token_text(token_id)
            if text is None:  # fragment
                continue
            node = root
            for ch in text:
                node = node.children.setdefault(ch, _TrieNode())
            node.ids.append(token_id)
        return root
