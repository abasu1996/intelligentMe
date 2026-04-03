"""Integration Monitoring API for Procurement.
 
Owner: <assign owner>
Docs: https://help.sap.com/doc/4a02653030b44d628d8b0d1d423cf58f/cloud/en-US/index.html
 
This API allows developers to monitor the execution status of integration
events (data import/export tasks) in SAP Ariba Procurement. It returns
the current running status of a specified event, or the last run status
if the event is not currently running.
 
Use cases:
  - Automate batch import/export job monitoring
  - Detect failed or stalled integration tasks
  - Track ITK (Integration Toolkit) and file-channel task progress
 
Endpoints implemented:
  GET /eventStatus – retrieve status of a specific integration event
 
Prerequisites:
  - SAP Ariba Developer Portal access (Procurement tab)
  - Application.Base.EnableIntegrationEventMonitoringAPI parameter
    must be enabled in Intelligent Configuration Manager
  - Client ID must be added to Application.OpenApi.ClientIds parameter
    in SAP Ariba Administrator
  - OAuth authentication configured (handled by AribaClient)
  - API intended for customer systems only — not for third-party connections
 
Notes:
  - Vanity URLs are NOT supported by this API
  - Only events run on the queried site are returned; child-site replicated
    data is NOT accessible via parent site queries
  - v2 is the active version — do not use v1
"""
 
import json
import os

import httpx
from fastmcp import FastMCP

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

BASE_URL = "https://openapi.ariba.com/api/procurement-eventstatus/v2/prod"


def _make_auth() -> DirectAuthClient:
    return DirectAuthClient(
        client_id=os.getenv("INTEGRATION_MONITORING_CLIENT_ID", ""),
        client_secret=os.getenv("INTEGRATION_MONITORING_CLIENT_SECRET", ""),
        api_key=os.getenv("INTEGRATION_MONITORING_API_KEY", ""),
    )
 
 
def register(mcp: FastMCP, client: AribaClient) -> None:

    _auth = _make_auth()

    @mcp.tool(
        name="ariba_procurement_integration_event_status",
        description=(
            "Retrieve the status of a specific integration event (data import or export task) "
            "in SAP Ariba Procurement. Returns the current running status of the event, "
            "or the last run status if the event is not currently running. "
            "Useful for automating and monitoring batch ITK and file-channel integration jobs."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def procurement_integration_event_status(
        realm: str,
        task_name: str,
        filter: str | None = None,
        top: int | None = None,
        skip: int | None = None,
    ) -> str:
        """
        Args:
            realm:      Required. The target SAP Ariba realm (e.g. 'MyCompanyS4').
                        Must match the realm configured in the client application.
            task_name:  Required. The name of the integration task/event to monitor
                        (e.g. 'UserImport', 'InvoiceExport', 'PurchaseOrderImport').
                        This is the ITK task name or file-channel job name as configured
                        in SAP Ariba Administrator.
            filter:     Optional. OData $filter expression. Supported fields:
                          taskName   – name of the integration task
                          status     – event status. Possible values:
                                         'Running'    – task is currently executing
                                         'Successful' – last run completed successfully
                                         'Failed'     – last run failed
                                         'Queued'     – task is queued to run
                          startTime  – task start time (ISO 8601, e.g. '2024-01-01T00:00:00Z')
                          endTime    – task end time (ISO 8601)
                        Example:
                          "taskName eq 'UserImport' and status eq 'Failed'"
                          "startTime ge '2024-01-01T00:00:00Z' and endTime le '2024-03-31T23:59:59Z'"
            top:        Optional. Max records to return per page (default 10).
            skip:       Optional. Number of records to skip for pagination (default 0).
        """
        try:
            headers = await _auth.get_headers()
            params: dict = {
                "realm": realm,
                "taskName": task_name,
            }
            if filter:
                params["$filter"] = filter
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/eventStatus",
                    headers=headers,
                    params=params,
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)