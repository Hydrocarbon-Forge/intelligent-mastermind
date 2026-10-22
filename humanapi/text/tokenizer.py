"""HumanTokenizer: the real Qwen tokenizer files, wrapped behind the ByteVocabTokenizer interface."""
from __future__ import annotations

from typing import Any, Sequence

from humanapi.text.base import ByteVocabTokenizer
from humanapi.text.prompt import prepare_messages

_PROBE_USER = "zqxuser"
_PROBE_ASSISTANT = "zqxreply"


def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table used by byte-level BPE vocabularies."""
    printable = list(range(ord("!"), ord("~") + 1))
    printable += list(range(ord("¡"), ord("¬") + 1))
    printable += list(range(ord("®"), ord("ÿ") + 1))
    codes = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, codes)}


def _model_vocab_size(path: str) -> int | None:
    try:
        from transformers import AutoConfig  # lazy

        size = AutoConfig.from_pretrained(path).vocab_size
    except Exception:
        return None
    return int(size) if size else None


def _end_tokens(hf: Any, special_ids: set[int]) -> tuple[set[int], int]:
    """eos ids from the files: eos_token plus specials the chat template emits after an assistant turn."""
    eos: set[int] = set()
    end_id: int | None = None
    if hf.eos_token_id is not None:
        eos.add(int(hf.eos_token_id))
        end_id = int(hf.eos_token_id)
    try:
        text = hf.apply_chat_template(
            [
                {"role": "user", "content": _PROBE_USER},
                {"role": "assistant", "content": _PROBE_ASSISTANT},
            ],
            tokenize=False,
        )
    except Exception:
        text = None
    if text and _PROBE_ASSISTANT in text:
        tail = text[text.rfind(_PROBE_ASSISTANT) + len(_PROBE_ASSISTANT):]
        specials = [i for i in hf.encode(tail, add_special_tokens=False) if i in special_ids]
        if specials:
            eos.update(specials)
            end_id = specials[0]
    if end_id is None or not eos:
        raise ValueError("could not determine end-of-turn token from the tokenizer files")
    return eos, end_id


class HumanTokenizer(ByteVocabTokenizer):
    def __init__(self, path: str, enable_thinking: bool = False) -> None:
        from transformers import AutoTokenizer  # lazy: keeps this module importable without transformers

        self.path = path
        self.enable_thinking = enable_thinking
        self._hf = AutoTokenizer.from_pretrained(path)
        special_ids = set(self._hf.all_special_ids) | set(self._hf.get_added_vocab().values())
        eos_ids, end_id = _end_tokens(self._hf, special_ids)

        decoder = _byte_decoder()
        vocab_bytes: dict[int, bytes] = {}
        for token, token_id in self._hf.get_vocab().items():
            if token_id in eos_ids:
                vocab_bytes[token_id] = token.encode("utf-8")
                continue
            if token_id in special_ids:
                continue
            try:
                vocab_bytes[token_id] = bytes(decoder[ch] for ch in token)
            except KeyError:
                continue
        vocab_size = _model_vocab_size(path) or len(self._hf)
        super().__init__(vocab_bytes, eos_ids, vocab_size, end_id)

    def encode(self, text: str) -> list[int]:
        return list(self._hf.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self._hf.decode(list(ids))

    def render_chat(self, messages: list[dict], knowledge: str | None) -> list[int]:
        prepared = prepare_messages(messages, knowledge)
        text = self._hf.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return list(self._hf.encode(text, add_special_tokens=False))
