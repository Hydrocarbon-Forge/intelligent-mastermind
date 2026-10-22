import numpy as np

from humanapi.learn.embed import HashEmbedder
from humanapi.learn.knn import KnnIndex


def _unit(rng, dim):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_knn_vote():
    idx = KnnIndex(dim=3, k=16, temperature=0.05)
    e0 = np.array([1, 0, 0], dtype=np.float32)
    e1 = np.array([0, 1, 0], dtype=np.float32)
    idx.add(e0, token_id=7)
    idx.add(e0, token_id=7)
    idx.add(e1, token_id=3)
    res = idx.query(e0)
    assert res.max_sim == 1.0
    assert res.neighbors[0] == (0, 1.0)  # ties broken by lower row id
    assert abs(sum(res.votes.values()) - 1.0) < 1e-9
    # Two neighbours chose token 7 and their weights pool.
    assert res.votes[7] > res.votes[3]
    assert set(res.votes) == {7, 3}
    assert len(idx) == 3


def test_knn_rebuild_from_log():
    rng = np.random.default_rng(0)
    dim = 32
    vecs = [_unit(rng, dim) for _ in range(60)]
    tokens = [int(t) for t in rng.integers(0, 5, size=60)]
    weights = [float(w) for w in rng.uniform(0.3, 1.0, size=60)]

    live = KnnIndex(dim=dim, k=8, temperature=0.05)
    for v, t, w in zip(vecs, tokens, weights):
        live.add(v, t, w)

    # Rebuild from float16-stored vectors, replaying in the same order.
    rebuilt = KnnIndex(dim=dim, k=8, temperature=0.05)
    for v, t, w in zip(vecs, tokens, weights):
        rebuilt.add(v.astype(np.float16), t, w)

    for _ in range(10):
        q = _unit(rng, dim)
        a = live.query(q)
        b = rebuilt.query(q)
        assert [r for r, _ in a.neighbors] == [r for r, _ in b.neighbors]
        assert set(a.votes) == set(b.votes)
        for t in a.votes:
            assert abs(a.votes[t] - b.votes[t]) < 1e-2
        assert abs(a.max_sim - b.max_sim) < 1e-2


def test_knn_empty_index_and_fewer_rows_than_k():
    empty = KnnIndex(dim=4, k=16)
    res = empty.query(np.ones(4, dtype=np.float32) / 2)
    assert res.votes == {}
    assert res.max_sim == 0.0
    assert res.neighbors == []

    one = KnnIndex(dim=4, k=16)
    one.add(np.array([1, 0, 0, 0], dtype=np.float32), token_id=9)
    res = one.query(np.array([1, 0, 0, 0], dtype=np.float32))
    assert res.votes == {9: 1.0}
    assert res.neighbors == [(0, 1.0)]


def test_hash_embedder_deterministic():
    emb = HashEmbedder(dim=128)
    texts = ["Hello there, friend.", "", "quantum chromodynamics", "Hello there, friend."]
    a = emb.embed(texts)
    b = HashEmbedder(dim=128).embed(texts)
    assert a.shape == (4, 128)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(a[0], a[3])
    # Shared words make texts closer than unrelated ones.
    q, near, far = emb.embed(["the cat sat on the mat", "a cat sat on a mat", "quantum chromodynamics"])
    assert float(q @ near) > float(q @ far)
