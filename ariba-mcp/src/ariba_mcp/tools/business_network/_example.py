"""EXAMPLE — Ariba Network Invoice Header Data Extraction API.

Owner: Ayub
Docs: https://help.sap.com/doc/9e361d7c0ccc40f5840012785f249811/cloud/en-US/index.html

This is a working example showing the pattern every team member should follow
when adding their API file to this folder.

Steps to add your own API:
  1. Create a new .py file in this folder (e.g. purchase_orders_supplier.py)
  2. Define a register(mcp, client) function
  3. Add your @mcp.tool functions inside it
  4. Import and call register() in __init__.py
  Order Change Requests API for Suppliers
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# TODO: Replace with actual API path from Developer Portal
INVOICE_HEADER_API = "invoice-header-extraction/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_invoice_header_list",
        description=(
            "List invoice headers from Ariba Network. "
            "Get header info of invoices associated with a Business Network user ID (ANID)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def invoice_header_list(anid: str, page_token: str | None = None) -> str:
        try:
            params = {"anid": anid}
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(INVOICE_HEADER_API, "invoices", params)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
