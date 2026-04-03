"""SAP Ariba MCP Server — entrypoint.

API Reference: https://help.sap.com/docs/ariba-apis
Developer Portal: https://developer.ariba.com
"""

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.config import get_settings
from ariba_mcp.tools import register_all_tools

# Create client at module level — no event loop needed here since
# httpx.AsyncClient is created per-request inside each method.
_client = AribaClient(get_settings())

mcp = FastMCP(
    "ariba-mcp",
    instructions=(
        "SAP Ariba Procurement MCP Server. "
        "Use ariba_* tools to query procurement reporting, sourcing projects, "
        "analytical data, supplier information, contract compliance, "
        "document approvals, audit logs, and supplier risk from your SAP Ariba realm. "
        "API docs: https://help.sap.com/docs/ariba-apis"
    ),
)

# Register all tools at import time so tools/list works immediately.
register_all_tools(mcp, _client)

if __name__ == "__main__":
    mcp.run()
