"""EXAMPLE — Operational Reporting API for Procurement.

Owner: Vanshika
Docs: https://help.sap.com/doc/42cb9e6fb65a4fa7b03f5e0ec7d406f9/cloud/en-US/index.html

Endpoint patterns:
  Sync:  GET  {base}/procurement-reporting-details/v1/prod/views/{viewName}?realm={realm}&filters={json}
  Count: GET  {base}/procurement-reporting-details/v1/prod/views/{viewName}/count?realm={realm}
  Meta:  GET  {base}/procurement-reporting-view/v2/prod/viewTemplates
  Job:   POST {base}/procurement-reporting-job/v2/prod/jobs
  Job:   GET  {base}/procurement-reporting-jobresult/v2/prod/jobs/{jobId}

Common views: POLineItemSystemView, RequisitionLineItemSystemView,
  InvoiceLineItemSystemView, ReceiptSystemView, PaymentSystemView

Steps to add your own API:
  1. Create a new .py file in this folder (e.g. analytical_reporting.py)
  2. Define a register(mcp, client) function
  3. Add your @mcp.tool functions inside it
  4. Import and call register() in __init__.py
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

PROCUREMENT_SYNC_API = "procurement-reporting-details/v1/prod"
PROCUREMENT_META_API = "procurement-reporting-view/v2/prod"
PROCUREMENT_JOB_API = "procurement-reporting-job/v2/prod"
PROCUREMENT_RESULT_API = "procurement-reporting-jobresult/v2/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_procurement_query_view",
        description=(
            "Query a procurement operational reporting view (synchronous). "
            "Common views: POLineItemSystemView, RequisitionLineItemSystemView, "
            "InvoiceLineItemSystemView, ReceiptSystemView, PaymentSystemView. "
            "Filters: '{\"createdDateFrom\":\"2025-01-01\",\"createdDateTo\":\"2025-01-31\"}'."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def procurement_query_view(
        view_name: str,
        filters: str | None = None,
        page_token: str | None = None,
    ) -> str:
        try:
            filter_dict = json.loads(filters) if filters else None
            result = await client.fetch_view(PROCUREMENT_SYNC_API, view_name, filter_dict, page_token)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
