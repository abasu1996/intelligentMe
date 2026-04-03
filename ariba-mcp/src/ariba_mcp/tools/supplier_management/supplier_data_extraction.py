"""Supplier Data Extraction API.

Owner: Nitish SM
Prod URL: https://openapi.ariba.com/api/supplier-extraction/v1/prod
Docs: https://help.sap.com/doc/0dfbbe78ddb647628f4b0acb3afe723d/cloud/en-US/4d4ad4075abb466b8077a7f064975e3a.html

Retrieves data about a specific set of suppliers from a specific realm.

Status: STUB — awaiting OAuth credentials from the SAP Ariba Developer Portal.
To activate: add ARIBA_EXTRACTION_CLIENT_ID / ARIBA_EXTRACTION_CLIENT_SECRET /
ARIBA_EXTRACTION_API_KEY to .env and config.py, then replace the stub error below.
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient

BASE_URL = "https://openapi.ariba.com/api/supplier-extraction/v1/prod"

_NOT_CONFIGURED = json.dumps({
    "error": True,
    "message": (
        "Supplier Data Extraction API is not yet configured. "
        "Add ARIBA_EXTRACTION_CLIENT_ID, ARIBA_EXTRACTION_CLIENT_SECRET, "
        "and ARIBA_EXTRACTION_API_KEY to .env and Horizon environment variables, "
        "then update this file to use DirectAuthClient."
    ),
})


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register Supplier Data Extraction API tools (stubs until credentials provided)."""

    @mcp.tool(
        name="ariba_supplier_extraction_by_ids",
        description=(
            "Extract supplier data for a specific set of suppliers by their vendor IDs. "
            "Returns full supplier profile data from the Supplier Data Extraction API. "
            "NOTE: Requires separate API credentials (not yet configured)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def extract_suppliers_by_ids(vendor_ids: str) -> str:
        # TODO: replace stub once ARIBA_EXTRACTION_* credentials are provisioned
        return _NOT_CONFIGURED

    @mcp.tool(
        name="ariba_supplier_extraction_all",
        description=(
            "Extract all supplier data from the realm using the Supplier Data Extraction API. "
            "Returns comprehensive supplier profiles. "
            "NOTE: Requires separate API credentials (not yet configured)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def extract_all_suppliers() -> str:
        # TODO: replace stub once ARIBA_EXTRACTION_* credentials are provisioned
        return _NOT_CONFIGURED
