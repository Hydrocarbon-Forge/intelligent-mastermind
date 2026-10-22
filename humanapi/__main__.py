"""`python -m humanapi`: serve humanapi.main:app with uvicorn on server.host and server.port from the config."""
from __future__ import annotations

import uvicorn

from humanapi.config import load_config
from humanapi.main import _config_path


def main() -> None:
    cfg = load_config(_config_path())
    uvicorn.run("humanapi.main:app", host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
