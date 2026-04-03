"""EXAMPLE — Operational Reporting API for Strategic Sourcing.

Owner: Pranathi
Docs: https://help.sap.com/doc/c4f46b9331834a0b970f834c20c9c73b/cloud/en-US/index.html

Endpoint pattern:
  GET {base}/sourcing-reporting-view/v1/prod/views/{viewName}?realm={realm}&filters={json}

Common views: SourcingProjectSourcingSystemView, SourcingRequestSourcingSystemView,
  EventItemSourcingSystemView, EventParticipationSourcingSystemView

Steps to add your own API:
  1. Create a new .py file in this folder (e.g. sourcing_project_management.py)
  2. Define a register(mcp, client) function
  3. Add your @mcp.tool functions inside it
  4. Import and call register() in __init__.py
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

SOURCING_REPORTING_API = "sourcing-reporting-view/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_sourcing_query_view",
        description=(
            "Query a sourcing operational reporting view. "
            "Common views: SourcingProjectSourcingSystemView, SourcingRequestSourcingSystemView, "
            "EventItemSourcingSystemView. "
            "Filters: '{\"createdDateFrom\":\"2025-01-01\",\"createdDateTo\":\"2025-01-31\"}'."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def sourcing_query_view(
        view_name: str,
        filters: str | None = None,
        page_token: str | None = None,
    ) -> str:
        try:
            filter_dict = json.loads(filters) if filters else None
            result = await client.fetch_view(SOURCING_REPORTING_API, view_name, filter_dict, page_token)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
