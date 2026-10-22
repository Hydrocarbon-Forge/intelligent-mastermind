"""The application: OpenAI routes, the UI WebSocket and the built UI, wired together at startup.

Importing this module needs no tokenizer, no transformers, no database file and no network: the real components
are built in the lifespan (startup) from the configuration, and only when no `deps` are given.

Run with:  uvicorn humanapi.main:app --port 8080   (or: python -m humanapi)
"""
from __future__ import annotations

import os
import random
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from humanapi.api.openai_routes import build_openai_router
from humanapi.api.ui_ws import build_ui_router
from humanapi.backend.base import InferenceBackend
from humanapi.backend.openai_compat import OpenAICompatBackend, load_capabilities
from humanapi.config import Config, load_config
from humanapi.core.engine import Engine
from humanapi.learn.embed import Embedder, SentenceTransformerEmbedder
from humanapi.store.db import connect
from humanapi.text.tokenizer import HumanTokenizer
from humanapi.text.types import Tokenizer

_DEFAULT_STATIC = Path(__file__).resolve().parent / "static"
_NOT_BUILT = "The UI is not built. Run `npm run build` in the ui/ folder, then restart the server.\n"


@dataclass
class Deps:
    backend: InferenceBackend
    tokenizer: Tokenizer
    embedder: Embedder
    conn: Optional[sqlite3.Connection] = None  # None: an in-memory database


def _config_path() -> Optional[str]:
    return os.environ.get("HUMANAPI_CONFIG") or None


def build_real_deps(cfg: Config) -> Deps:
    """The production components. Raises a clear error when the tokenizer files are missing."""
    tokenizer_dir = Path(cfg.tokenizer.path)
    if not (tokenizer_dir / "tokenizer.json").exists() and not (tokenizer_dir / "tokenizer_config.json").exists():
        raise FileNotFoundError(
            f"tokenizer files not found: tokenizer.path={cfg.tokenizer.path!r} "
            "(expected tokenizer.json or tokenizer_config.json in that folder)"
        )
    tokenizer = HumanTokenizer(cfg.tokenizer.path, cfg.tokenizer.enable_thinking)
    capabilities = load_capabilities(cfg.backend.capabilities_file)
    backend = OpenAICompatBackend(
        cfg.backend.base_url,
        cfg.backend.model,
        tokenizer,
        capabilities,
        timeout_s=cfg.backend.timeout_s,
    )
    embedder = SentenceTransformerEmbedder(cfg.embed.model, cfg.embed.device)
    Path(cfg.storage.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(cfg.storage.db_path)
    return Deps(backend=backend, tokenizer=tokenizer, embedder=embedder, conn=conn)


def create_app(
    cfg: Optional[Config] = None,
    *,
    deps: Optional[Deps] = None,
    static_dir: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> FastAPI:
    cfg = cfg if cfg is not None else load_config(_config_path())
    static_root = Path(static_dir) if static_dir is not None else _DEFAULT_STATIC

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        components = deps if deps is not None else build_real_deps(cfg)
        engine = Engine.create(
            cfg,
            backend=components.backend,
            tokenizer=components.tokenizer,
            embedder=components.embedder,
            conn=components.conn,
            rng=rng,
        )
        app.state.engine = engine
        # Routers need the engine, so they are attached at startup. Starlette consults the route list per
        # request, so routes added here are served; each app instance is started once.
        app.include_router(build_openai_router(engine, cfg))
        app.include_router(build_ui_router(engine, cfg))
        await engine.start()
        try:
            yield
        finally:
            await engine.stop()
            close = getattr(components.backend, "aclose", None)
            if close is not None:
                await close()
            engine.conn.close()

    app = FastAPI(title="Intelligent Mastermind", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index() -> object:
        index_file = static_root / "index.html"
        if index_file.is_file():
            return FileResponse(index_file, media_type="text/html")
        # An empty static folder (no index.html) counts as not built.
        return PlainTextResponse(_NOT_BUILT)

    @app.get("/assets/{path:path}", include_in_schema=False)
    async def asset(path: str) -> FileResponse:
        root = (static_root / "assets").resolve()
        target = (root / path).resolve()
        if not target.is_file() or not str(target).startswith(str(root) + os.sep):
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(target)

    return app


# Module-level application for `uvicorn humanapi.main:app`. Nothing heavy happens until startup.
app: FastAPI = create_app()
