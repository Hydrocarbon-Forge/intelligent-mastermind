"""Knowledge store: chunking overlap, retrieval threshold, forgetting, folder ingestion."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from humanapi.config import KnowledgeConfig
from humanapi.knowledge.store import KnowledgeStore, chunk_text
from humanapi.learn.embed import HashEmbedder
from humanapi.text.toy_tokenizer import ToyTokenizer

SCHEMA = Path(__file__).resolve().parents[2] / "humanapi" / "store" / "schema.sql"


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA.read_text(encoding="utf-8"))
    yield c
    c.close()


@pytest.fixture
def tok():
    return ToyTokenizer()


def make_store(conn, tok, **overrides) -> KnowledgeStore:
    base = dict(min_similarity=0.5, top_chunks=4, chunk_tokens=300, chunk_overlap_tokens=40)
    base.update(overrides)
    return KnowledgeStore(conn, HashEmbedder(dim=128), tok, KnowledgeConfig(**base))


def test_chunking_overlap(tok):
    # Letters only: one token per byte, so window positions are exact.
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 tokens
    chunks = chunk_text(text, tok, chunk_tokens=40, overlap_tokens=10)
    stride = 30
    # windows start at 0, 30, ..., 240; the last one is a 20-token tail
    assert len(chunks) == 9
    for i, c in enumerate(chunks):
        assert c == text[stride * i : stride * i + 40]
    assert len(chunks[-1]) == 20
    # consecutive full chunks share exactly the overlap
    for a, b in zip(chunks, chunks[1:]):
        assert a[-10:] == b[:10]
    assert all(len(tok.encode(c)) <= 40 for c in chunks)
    assert all(c for c in chunks)


def test_chunking_short_and_exact(tok):
    assert chunk_text("abc", tok, chunk_tokens=40, overlap_tokens=10) == ["abc"]
    exact = "x" * 40
    assert chunk_text(exact, tok, chunk_tokens=40, overlap_tokens=10) == [exact]
    assert chunk_text("   ", tok, chunk_tokens=40, overlap_tokens=10) == ["   "]  # tokens exist; add() skips blanks
    with pytest.raises(ValueError):
        chunk_text("abc", tok, chunk_tokens=10, overlap_tokens=10)


def test_retrieval_threshold(conn, tok):
    store = make_store(conn, tok, top_chunks=4)
    ids = store.add("the sky is blue today", source="a")
    store.add("cats sleep all day long", source="b")
    assert len(ids) == 1

    hit = store.retrieve("the sky is blue today")
    assert hit.hit is True
    assert [c.id for c in hit.chunks] == [ids[0]]
    assert hit.chunks[0].sim == pytest.approx(1.0, abs=1e-5)
    assert hit.max_sim == pytest.approx(1.0, abs=1e-5)

    miss = store.retrieve("qqqq zzzz wwww")
    assert miss.hit is False
    assert miss.chunks == []
    assert 0.0 <= miss.max_sim < 0.5

    other = sqlite3.connect(":memory:")
    other.executescript(SCHEMA.read_text(encoding="utf-8"))
    empty = make_store(other, tok)
    assert empty.retrieve("anything").max_sim == 0.0
    other.close()


def test_empty_store_skips_embedder(conn, tok):
    class Spy(HashEmbedder):
        calls = 0

        def embed(self, texts):
            Spy.calls += 1
            return super().embed(texts)

    store = KnowledgeStore(conn, Spy(dim=32), tok, KnowledgeConfig())
    r = store.retrieve("hello")
    assert r.hit is False and r.max_sim == 0.0 and r.chunks == []
    assert Spy.calls == 0


def test_forget_removes_chunk(conn, tok):
    store = make_store(conn, tok)
    keep, drop = store.add("the sky is blue today", source="a")[0], store.add("cats sleep all day", source="b")[0]
    assert store.count() == 2

    assert store.forget(drop) is True
    row = conn.execute("SELECT deleted, text, length(vec) FROM knowledge_chunks WHERE id = ?", (drop,)).fetchone()
    assert row[0] == 1 and row[1] == "" and row[2] == 0
    assert drop not in store._ids
    assert store.forget(drop) is False  # already deleted
    assert store.forget(9999) is False  # unknown

    assert [c.id for c in store.notes()] == [keep]
    assert store.count() == 1

    store.load()
    assert store.count() == 1
    r = store.retrieve("cats sleep all day")
    assert drop not in [c.id for c in r.chunks]
    assert store.retrieve("the sky is blue today").chunks[0].id == keep


def test_forget_best_match_lowers_max_sim(conn, tok):
    store = make_store(conn, tok)
    best = store.add("the sky is blue today", source="a")[0]
    store.add("cats sleep all day long", source="b")
    before = store.retrieve("the sky is blue today").max_sim
    store.forget(best)
    after = store.retrieve("the sky is blue today").max_sim
    assert after < before


def test_add_folder_reads_both_extensions(conn, tok, tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha note", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta note", encoding="utf-8")
    (tmp_path / "c.py").write_text("ignored", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.txt").write_text("nested", encoding="utf-8")

    store = make_store(conn, tok)
    ids = store.add_folder(str(tmp_path))
    assert len(ids) == 2
    assert [c.source for c in store.notes()] == ["a.txt", "b.md"]
    assert store.add("   ") == []


def test_knowledge_text_and_token_ids(conn, tok):
    store = make_store(conn, tok)
    store.add("one", source="a")
    store.add("two", source="b")
    chunks = store.notes()
    assert store.knowledge_text([]) is None
    assert store.knowledge_text(chunks) == "one\n\ntwo"
    ids = store.token_ids(chunks)
    assert set(tok.encode("one")) <= ids and set(tok.encode(" two")) <= ids
    assert isinstance(ids, frozenset)
    assert np.isfinite(store.retrieve("one").max_sim)
