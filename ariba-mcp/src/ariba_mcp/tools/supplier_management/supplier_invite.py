"""Supplier Invite API.

Owner: Nitish SM
Prod URL: https://openapi.ariba.com/api/supplier-invite/v2/prod
Docs: https://help.sap.com/doc/8903b23ba81845efb4760ba9ad2096bb/cloud/en-US/03064470056c477288514f0d68b69bf3.html

Creates vendor records in SAP Business Network and manages supplier invitations.

Authentication: OAuth 2.0 Bearer token + apiKey header (own credentials)
"""

import json

import httpx
from fastmcp import FastMCP

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.config import get_settings
from ariba_mcp.errors import handle_ariba_error

BASE_URL = "https://openapi.ariba.com/api/supplier-invite/v2/prod"


def _make_auth() -> DirectAuthClient:
    s = get_settings()
    return DirectAuthClient(
        client_id=s.ariba_invite_client_id,
        client_secret=s.ariba_invite_client_secret,
        api_key=s.ariba_invite_api_key,
        timeout=s.request_timeout,
    )


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register Supplier Invite API tools."""

    _auth = _make_auth()

    @mcp.tool(
        name="ariba_invite_list_invitations",
        description=(
            "List all supplier invitations sent from the realm. "
            "Returns invitation ID, supplier name, email, status (Pending, Accepted, Rejected), "
            "and creation date."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_invitations(top: int = 100, skip: int = 0) -> str:
        try:
            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/suppliers",
                    headers=headers,
                    params={"realm": client.realm, "$top": top, "$skip": skip},
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_invite_get_invitation",
        description=(
            "Get the status and details of a specific supplier invitation by invitation ID. "
            "Returns supplier name, email, ANID (if accepted), status, and timestamps."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_invitation(invitation_id: str) -> str:
        try:
            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/suppliers/{invitation_id}",
                    headers=headers,
                    params={"realm": client.realm},
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_invite_supplier",
        description=(
            "Invite a new supplier to register in SAP Business Network. "
            "Required: supplier_name and email. "
            "Optional: first_name, last_name, phone, country_code (ISO 3166, e.g. 'US'), "
            "language_code (e.g. 'en'), custom_message. "
            "Returns the invitation ID on success."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def invite_supplier(
        supplier_name: str,
        email: str,
        first_name: str | None = None,
        last_name: str | None = None,
        phone: str | None = None,
        country_code: str | None = None,
        language_code: str = "en",
        custom_message: str | None = None,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            headers["Content-Type"] = "application/json"

            body: dict = {
                "supplierName": supplier_name,
                "contactEmail": email,
                "languageCode": language_code,
            }
            if first_name:
                body["contactFirstName"] = first_name
            if last_name:
                body["contactLastName"] = last_name
            if phone:
                body["contactPhone"] = phone
            if country_code:
                body["countryCode"] = country_code
            if custom_message:
                body["customMessage"] = custom_message

            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    f"{BASE_URL}/suppliers",
                    headers=headers,
                    params={"realm": client.realm},
                    json=body,
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)
