"""EXAMPLE — Risk Exposure API.

Owner: Anim
Prod URL: https://openapi.ariba.com/api/risk-exposure/v3/prod
Docs: https://help.sap.com/doc/2e493ebf33de4b2bb813c8de0fb9ed9b/cloud/en-US/index.html

Steps to add your own API:
  1. Create a new .py file in this folder (e.g. supplier_data_extraction.py)
  2. Define a register(mcp, client) function
  3. Add your @mcp.tool functions inside it
  4. Import and call register() in __init__.py
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

RISK_EXPOSURE_API = "risk-exposure/v3/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_risk_get_exposure",
        description=(
            "Get supplier risk exposure data. Pass an SM vendor ID or ERP vendor ID "
            "to retrieve the supplier's overall and category risk scores."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def risk_get_exposure(vendor_id: str) -> str:
        try:
            result = await client.fetch_resource(RISK_EXPOSURE_API, f"riskExposure/{vendor_id}")
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
