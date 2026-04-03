"""Supplier Data API (supplierdataaccess/v1).

Owner: Nitish SM
Prod URL: https://openapi.ariba.com/api/supplierdataaccess/v1/prod
Docs: https://help.sap.com/doc/ae35ad2426d54acca48272f08dbfbe73/cloud/en-US/c1f5f6056132476db23a1251add642ae.html

Retrieves supplier data including registration status, qualification status,
preferred status, and questionnaire details.

Authentication: OAuth 2.0 Bearer token + apiKey header (own credentials)
"""

import json

import httpx
from fastmcp import FastMCP

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.config import get_settings
from ariba_mcp.errors import handle_ariba_error

BASE_URL = "https://openapi.ariba.com/api/supplierdataaccess/v1/prod"


def _make_auth() -> DirectAuthClient:
    s = get_settings()
    return DirectAuthClient(
        client_id=s.ariba_sda_client_id,
        client_secret=s.ariba_sda_client_secret,
        api_key=s.ariba_sda_api_key,
        timeout=s.request_timeout,
    )


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register Supplier Data API tools."""

    _auth = _make_auth()

    @mcp.tool(
        name="ariba_sda_list_suppliers",
        description=(
            "List suppliers from the Supplier Data API. "
            "Returns supplier registration status, qualification status, preferred status, "
            "contact details, and associated questionnaire data. "
            "Supports pagination via top (page size, default 100) and skip (offset)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_suppliers(top: int = 100, skip: int = 0) -> str:
        try:
            headers = await _auth.get_headers()
            params = {
                "realm": client.realm,
                "$top": top,
                "$skip": skip,
            }
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/suppliers",
                    headers=headers,
                    params=params,
                    timeout=60,
                )
                resp.raise_for_status()
            data = resp.json()
            return json.dumps(data, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_sda_get_supplier",
        description=(
            "Get full details for a specific supplier by their SM Vendor ID. "
            "Returns registration status, qualification status, preferred status, "
            "address, contact details, and questionnaire responses."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_supplier(vendor_id: str) -> str:
        try:
            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/suppliers/{vendor_id}",
                    headers=headers,
                    params={"realm": client.realm},
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_sda_get_supplier_questionnaires",
        description=(
            "Get all questionnaire responses for a specific supplier by SM Vendor ID. "
            "Returns questionnaire titles, completion status, scores, and individual responses."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_supplier_questionnaires(vendor_id: str) -> str:
        try:
            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/suppliers/{vendor_id}/questionnaires",
                    headers=headers,
                    params={"realm": client.realm},
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_sda_filter_suppliers",
        description=(
            "Filter suppliers by registration or qualification status using the Supplier Data API. "
            "registrationStatus options: Registered, Invited, NotInvited, RegistrationDenied. "
            "qualificationStatus options: Qualified, QualifiedForSome, NotQualified, Unknown. "
            "preferredStatus options: Preferred, NotPreferred."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def filter_suppliers(
        registration_status: str | None = None,
        qualification_status: str | None = None,
        preferred_status: str | None = None,
        top: int = 100,
        skip: int = 0,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            params: dict = {
                "realm": client.realm,
                "$top": top,
                "$skip": skip,
            }
            if registration_status:
                params["registrationStatus"] = registration_status
            if qualification_status:
                params["qualificationStatus"] = qualification_status
            if preferred_status:
                params["preferredStatus"] = preferred_status

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/suppliers",
                    headers=headers,
                    params=params,
                    timeout=60,
                )
                resp.raise_for_status()
            data = resp.json()
            return json.dumps(data, default=str)
        except Exception as e:
            return handle_ariba_error(e)
