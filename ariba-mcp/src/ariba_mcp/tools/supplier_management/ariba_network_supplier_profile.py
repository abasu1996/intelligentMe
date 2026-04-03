"""Ariba Network Supplier Profile API.

Owner: Nitish SM
Prod URL: https://openapi.ariba.com/api/ariba-network-supplier-profile/v1/prod
          (Note: the index lists invoice-extraction/v1 — verify the correct path
           on the SAP Ariba Developer Portal before activating)
Docs: https://help.sap.com/doc/bba3a69a443a46ee80382c8d8c0610ae/cloud/en-US/c03277fb38224c05a38cdd980bc816db.html

Retrieves SAP Business Network supplier profile information based on supplier ANID.
Requires a valid SAP Business Network Buyer Account.

Status: STUB — awaiting OAuth credentials and confirmed production URL.
To activate: provision credentials on the SAP Ariba Developer Portal, add
ARIBA_NETWORK_PROFILE_CLIENT_ID / SECRET / API_KEY to .env and config.py,
confirm the correct prod URL, then replace the stub error below.
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient

_NOT_CONFIGURED = json.dumps({
    "error": True,
    "message": (
        "Ariba Network Supplier Profile API is not yet configured. "
        "Provision credentials on the SAP Ariba Developer Portal and add "
        "ARIBA_NETWORK_PROFILE_CLIENT_ID, ARIBA_NETWORK_PROFILE_CLIENT_SECRET, "
        "and ARIBA_NETWORK_PROFILE_API_KEY to .env and Horizon environment variables. "
        "Also confirm the correct production URL (the index URL may be incorrect)."
    ),
})


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register Ariba Network Supplier Profile API tools (stubs until credentials provided)."""

    @mcp.tool(
        name="ariba_network_supplier_profile_get",
        description=(
            "Get SAP Business Network supplier profile information by ANID (Ariba Network ID). "
            "Returns company profile, contact details, certifications, and network status. "
            "Requires a valid SAP Business Network Buyer Account. "
            "NOTE: Requires separate API credentials (not yet configured)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_supplier_profile(anid: str) -> str:
        # TODO: replace stub once ARIBA_NETWORK_PROFILE_* credentials are provisioned
        return _NOT_CONFIGURED

    @mcp.tool(
        name="ariba_network_supplier_profile_search",
        description=(
            "Search for SAP Business Network supplier profiles by name or category. "
            "Returns matching supplier profiles with ANID, company name, and contact info. "
            "NOTE: Requires separate API credentials (not yet configured)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def search_supplier_profiles(query: str) -> str:
        # TODO: replace stub once ARIBA_NETWORK_PROFILE_* credentials are provisioned
        return _NOT_CONFIGURED
