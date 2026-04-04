from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariba_mcp.server import mcp


def _http_path() -> str:
    path = os.getenv("MCP_HTTP_PATH", "/mcp").strip() or "/mcp"
    if not path.startswith("/"):
        return f"/{path}"
    return path


app = mcp.http_app(path=_http_path())


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
