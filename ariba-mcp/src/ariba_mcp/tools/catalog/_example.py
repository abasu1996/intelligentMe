"""EXAMPLE — Internal Catalogs Shop API.

Owner: Anil
Docs: https://help.sap.com/doc/9d210ac7774a48c7a50e064db3c932e9/cloud/en-US/index.html

Steps to add your own API:
  1. Create a new .py file in this folder (e.g. public_catalogs_shop.py)
  2. Define a register(mcp, client) function
  3. Add your @mcp.tool functions inside it
  4. Import and call register() in __init__.py
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# TODO: Replace with actual API path from Developer Portal
INTERNAL_CATALOG_API = "internal-catalogs/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_catalog_internal_search",
        description=(
            "Search items in internal catalogs. "
            "Retrieve catalog data from internal catalogs in the SAP Ariba Catalog solution."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def catalog_internal_search(query: str, page_token: str | None = None) -> str:
        try:
            params = {"q": query}
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(INTERNAL_CATALOG_API, "items", params)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
