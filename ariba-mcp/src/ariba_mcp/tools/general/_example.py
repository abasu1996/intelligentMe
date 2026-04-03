"""EXAMPLE — Document Approval API.

Owner: Anim
Docs: https://help.sap.com/doc/f9cd5fe02da34e5a9c0ddd8161ee04d1/cloud/en-US/index.html

Steps to add your own API:
  1. Create a new .py file in this folder (e.g. audit_search.py)
  2. Define a register(mcp, client) function
  3. Add your @mcp.tool functions inside it
  4. Import and call register() in __init__.py
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# TODO: Replace with actual API path from Developer Portal
DOCUMENT_APPROVAL_API = "document-approval/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_approval_list_pending",
        description=(
            "List pending document approvals. "
            "Returns requisitions, invoices, and user profiles awaiting approval."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def approval_list_pending(page_token: str | None = None) -> str:
        try:
            params = {}
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(DOCUMENT_APPROVAL_API, "pendingApprovals", params)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
