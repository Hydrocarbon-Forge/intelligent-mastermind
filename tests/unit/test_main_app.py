import subprocess
import sys

from fastapi.testclient import TestClient

from humanapi.backend.base import BackendUnavailable, Capabilities
from humanapi.backend.fake import FakeBackend
from humanapi.config import Config
from humanapi.learn.embed import HashEmbedder
from humanapi.main import Deps, create_app
from humanapi.text.toy_tokenizer import ToyTokenizer

TOK = ToyTokenizer()


def _deps(backend=None):
    backend = backend or FakeBackend(
        lambda ctx: {TOK.end_id: 1.0}, capabilities=Capabilities(), token_bytes=lambda i: TOK.token_bytes(i) or b""
    )
    return Deps(backend=backend, tokenizer=TOK, embedder=HashEmbedder(dim=32))


def _cfg():
    cfg = Config()
    cfg.gate.cold_start_min_labels = 0
    return cfg


def test_health_models_and_placeholder(tmp_path):
    app = create_app(_cfg(), deps=_deps(), static_dir=str(tmp_path / "missing"))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["process"] == "ok" and health.json()["database"] == "ok"
        models = client.get("/v1/models").json()
        assert models["data"][0]["id"] == "human"
        page = client.get("/")
        assert page.status_code == 200
        assert "npm run build" in page.text
        assert app.state.engine is not None


def test_serves_built_ui_and_assets(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>built</title>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    app = create_app(_cfg(), deps=_deps(), static_dir=str(tmp_path))
    with TestClient(app) as client:
        assert "<title>built</title>" in client.get("/").text
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/assets/../config.yaml").status_code == 404
        assert client.get("/v1/models").status_code == 200  # API routes are unaffected


def test_empty_static_dir_counts_as_not_built(tmp_path):
    app = create_app(_cfg(), deps=_deps(), static_dir=str(tmp_path))
    with TestClient(app) as client:
        assert "npm run build" in client.get("/").text


def test_boots_with_backend_down():
    backend = FakeBackend(
        lambda ctx: {TOK.end_id: 1.0}, capabilities=Capabilities(), token_bytes=lambda i: TOK.token_bytes(i) or b""
    )
    backend.fail_with = BackendUnavailable("server is down")
    app = create_app(_cfg(), deps=_deps(backend), static_dir="/nonexistent")
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["process"] == "ok"
        assert health["backend"] is False


def test_two_apps_in_one_process(tmp_path):
    first = create_app(_cfg(), deps=_deps(), static_dir=str(tmp_path))
    second = create_app(_cfg(), deps=_deps(), static_dir=str(tmp_path))
    with TestClient(first) as a, TestClient(second) as b:
        assert a.get("/health").status_code == 200
        assert b.get("/health").status_code == 200
        assert first.state.engine is not second.state.engine


def test_import_is_lightweight():
    """Importing the app must not load transformers or sentence-transformers (checked in a fresh process)."""
    code = (
        "import sys, humanapi.main; "
        "heavy = [m for m in ('transformers', 'sentence_transformers') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
