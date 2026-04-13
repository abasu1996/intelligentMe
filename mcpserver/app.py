"""ASGI entry point for Azure App Service."""

from __future__ import annotations

from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

try:
    from mcpserver.mcp_server import mcp
except ModuleNotFoundError:
    from mcp_server import mcp

# Azure App Service assigns a public hostname dynamically, so the localhost-only
# defaults from FastMCP would reject valid requests. We disable the host filter
# here for the HTTP deployment entry point while keeping the MCP app unchanged.
mcp.settings.transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)


@mcp.custom_route("/", methods=["GET"], include_in_schema=False)
async def root_status(_request):
    return JSONResponse(
        {
            "service": "invoice-po-tax-analyzer",
            "status": "ok",
            "transport": "streamable-http",
            "mcp_path": mcp.settings.streamable_http_path,
            "health_path": "/health",
        }
    )


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health_check(_request):
    return JSONResponse({"status": "healthy"})


app = mcp.streamable_http_app()
