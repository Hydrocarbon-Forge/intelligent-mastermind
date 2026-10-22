"""Deterministic byte-level toy tokenizer with the same interface as HumanTokenizer (CPU-only tests)."""
from __future__ import annotations

from typing import Sequence

from humanapi.text.base import ByteVocabTokenizer
from humanapi.text.prompt import prepare_messages

_END_ID = 256
_END_TEXT = b"<|end|>"
_PIECES: tuple[str, ...] = (
    " the", " a", " I", "I", " to", " and", " of", " is", " it", " you",
    " yes", " yeah", " no", " sounds", " good", " great", " let", " that", " this", " sure",
    " Sure", "Sure", "Yes", "No", "Hello", " Hello", "Thanks", " thanks", " lunch", " tomorrow",
    " want", " grab", " happy", " un", "un", "believ", "able", " believ", " pr", "pr",
    "ogram", " program", " ok", " okay", " maybe", " can", " will", " not", " do", " see",
    " later", " soon", " we", " are", " have", " so", " but", " what", " when", " where",
    " how", " why", " me", " my", " your", " for", " with", " at", " on", " in",
    " be", "ing", "ed", "s", "ly", "'t", "'s", "'m", ",", ".",
    "!", "?", ":", ";", "-", "\n", "\n\n", "  ", " 2", "2",
    "42", " 42", " 10", "0", "1", "3", "(", ")", '"', "'",
)


class ToyTokenizer(ByteVocabTokenizer):
    """Ids: 0-255 raw bytes, 256 END, then _PIECES in order, then padding ids with no string."""

    def __init__(self, padding_ids: int = 8) -> None:
        vocab: dict[int, bytes] = {byte: bytes([byte]) for byte in range(256)}
        vocab[_END_ID] = _END_TEXT
        for index, piece in enumerate(_PIECES):
            vocab[_END_ID + 1 + index] = piece.encode("utf-8")
        real_ids = _END_ID + 1 + len(_PIECES)
        super().__init__(
            vocab,
            eos_ids={_END_ID},
            vocab_size=real_ids + padding_ids,
            end_id=_END_ID,
        )
        # Encoding prefers a listed piece over the raw byte with the same bytes.
        self._encode_table: dict[bytes, int] = {bytes([byte]): byte for byte in range(256)}
        for token_id in range(_END_ID + 1, real_ids):
            self._encode_table[vocab[token_id]] = token_id
        self._encode_table[_END_TEXT] = _END_ID
        self._max_piece_len = max(len(key) for key in self._encode_table)

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        out: list[int] = []
        position = 0
        total = len(data)
        while position < total:
            longest = min(self._max_piece_len, total - position)
            for length in range(longest, 0, -1):
                token_id = self._encode_table.get(data[position : position + length])
                if token_id is not None:
                    out.append(token_id)
                    position += length
                    break
        return out

    def decode(self, ids: Sequence[int]) -> str:
        raw = b"".join(self._bytes.get(int(token_id), b"") for token_id in ids)
        return raw.decode("utf-8", errors="replace")

    def render_chat(self, messages: list[dict], knowledge: str | None) -> list[int]:
        prepared = prepare_messages(messages, knowledge)
        text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in prepared)
        return self.encode(text + "<|assistant|>\n")
