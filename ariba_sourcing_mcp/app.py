from __future__ import annotations

import os

import uvicorn

from main import mcp


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
