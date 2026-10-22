"""Prompt message normalisation shared by every tokenizer's render_chat."""
from __future__ import annotations

_IMAGE_PLACEHOLDER = "[image]"


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                pieces.append(part.get("text") or "")
            else:
                pieces.append(_IMAGE_PLACEHOLDER)
        return "".join(pieces)
    return str(content)


def prepare_messages(messages: list[dict], knowledge: str | None) -> list[dict]:
    """Keep only role and text content; append retrieved knowledge to the system message.

    The input list and its dicts are never mutated.
    """
    prepared: list[dict] = []
    for message in messages:
        prepared.append(
            {
                "role": message.get("role", "user"),
                "content": _content_text(message.get("content")),
            }
        )
    if knowledge is None:
        return prepared
    notes = "Relevant notes:\n" + knowledge
    for index, message in enumerate(prepared):
        if message["role"] == "system":
            existing = message["content"]
            combined = existing + "\n\n" + notes if existing else notes
            prepared[index] = {"role": "system", "content": combined}
            return prepared
    return [{"role": "system", "content": notes}, *prepared]
