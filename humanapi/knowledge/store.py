"""Knowledge store: chunked notes and documents, embedded, retrieved once per request, forgettable.

Rows live in the `knowledge_chunks` table of the given connection; the embeddings are also kept in an
in-memory matrix for retrieval. Forgetting blanks the row and removes the vector from the matrix.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

from humanapi.config import KnowledgeConfig
from humanapi.learn.embed import Embedder
from humanapi.text.types import Tokenizer

_FOLDER_EXTENSIONS = (".txt", ".md")


@dataclass
class Chunk:
    id: int
    source: str
    text: str
    sim: Optional[float] = None


@dataclass
class Retrieval:
    chunks: list[Chunk]  # best first; only chunks with sim >= min_similarity, at most top_chunks
    hit: bool  # True when chunks is non-empty
    max_sim: float  # best similarity over all stored chunks, 0.0 when the store is empty


def chunk_text(text: str, tok: Tokenizer, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into windows of chunk_tokens token ids with stride chunk_tokens - overlap_tokens."""
    if chunk_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("need chunk_tokens > 0 and 0 <= overlap_tokens < chunk_tokens")
    ids = tok.encode(text)
    if not ids:
        return []
    stride = chunk_tokens - overlap_tokens
    chunks: list[str] = []
    start = 0
    while True:
        window = ids[start : start + chunk_tokens]
        piece = tok.decode(window)
        if piece:
            chunks.append(piece)
        if start + chunk_tokens >= len(ids):
            break
        start += stride
    return chunks


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeStore:
    def __init__(self, conn: sqlite3.Connection, embedder: Embedder, tok: Tokenizer, cfg: KnowledgeConfig):
        self.conn = conn
        self.embedder = embedder
        self.tok = tok
        self.cfg = cfg
        self._ids: list[int] = []
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self.load()

    def load(self) -> None:
        """Rebuild the in-memory matrix from the live rows."""
        rows = self.conn.execute(
            "SELECT id, vec FROM knowledge_chunks WHERE deleted = 0 ORDER BY id"
        ).fetchall()
        self._ids = [int(r[0]) for r in rows]
        dim = self.embedder.dim
        if rows:
            self._matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).reshape(len(rows), dim)
        else:
            self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, text: str, source: str = "note") -> list[int]:
        if text.strip() == "":
            return []
        pieces = chunk_text(text, self.tok, self.cfg.chunk_tokens, self.cfg.chunk_overlap_tokens)
        if not pieces:
            return []
        vecs = np.asarray(self.embedder.embed(pieces), dtype=np.float32).reshape(len(pieces), self.embedder.dim)
        created = _now()
        new_ids: list[int] = []
        for piece, vec in zip(pieces, vecs):
            cur = self.conn.execute(
                "INSERT INTO knowledge_chunks (source, text, vec, created_at, deleted) VALUES (?, ?, ?, ?, 0)",
                (source, piece, vec.tobytes(), created),
            )
            new_ids.append(int(cur.lastrowid))
        self.conn.commit()
        self._ids.extend(new_ids)
        self._matrix = np.vstack([self._matrix, vecs])
        return new_ids

    def add_folder(self, path: str) -> list[int]:
        """Add every *.txt and *.md file directly inside path (non-recursive, sorted by name)."""
        ids: list[int] = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if not name.endswith(_FOLDER_EXTENSIONS) or not os.path.isfile(full):
                continue
            with open(full, "r", encoding="utf-8") as fh:
                ids.extend(self.add(fh.read(), source=name))
        return ids

    def forget(self, chunk_id: int) -> bool:
        row = self.conn.execute("SELECT deleted FROM knowledge_chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None or int(row[0]) == 1:
            return False
        self.conn.execute(
            "UPDATE knowledge_chunks SET deleted = 1, text = '', vec = x'' WHERE id = ?", (chunk_id,)
        )
        self.conn.commit()
        if chunk_id in self._ids:
            pos = self._ids.index(chunk_id)
            self._ids.pop(pos)
            self._matrix = np.delete(self._matrix, pos, axis=0)
        return True

    def retrieve(self, query: str) -> Retrieval:
        if not self._ids:
            return Retrieval(chunks=[], hit=False, max_sim=0.0)
        q = np.asarray(self.embedder.embed([query]), dtype=np.float32).reshape(-1)
        sims = (self._matrix @ q).astype(np.float64)
        max_sim = float(sims.max())
        order = np.argsort(-sims, kind="stable")
        chunks: list[Chunk] = []
        for pos in order:
            sim = float(sims[pos])
            if sim < self.cfg.min_similarity or len(chunks) >= self.cfg.top_chunks:
                break
            chunk = self._fetch(self._ids[int(pos)])
            if chunk is not None:
                chunk.sim = sim
                chunks.append(chunk)
        return Retrieval(chunks=chunks, hit=bool(chunks), max_sim=max_sim)

    def _fetch(self, chunk_id: int) -> Optional[Chunk]:
        row = self.conn.execute(
            "SELECT id, source, text FROM knowledge_chunks WHERE id = ? AND deleted = 0", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return Chunk(id=int(row[0]), source=str(row[1]), text=str(row[2]))

    def notes(self) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT id, source, text FROM knowledge_chunks WHERE deleted = 0 ORDER BY id"
        ).fetchall()
        return [Chunk(id=int(r[0]), source=str(r[1]), text=str(r[2])) for r in rows]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE deleted = 0").fetchone()
        return int(row[0])

    def knowledge_text(self, chunks: Sequence[Chunk]) -> Optional[str]:
        if not chunks:
            return None
        return "\n\n".join(c.text for c in chunks)

    def token_ids(self, chunks: Sequence[Chunk]) -> frozenset[int]:
        ids: set[int] = set()
        for c in chunks:
            ids.update(self.tok.encode(c.text))
            ids.update(self.tok.encode(" " + c.text))
        return frozenset(ids)
