"""Asset Management API.

Owner: Rohit Naik
Prod URL: https://openapi.ariba.com/api/asset-management/v1/prod

Manage asset line items in SAP Ariba purchase requisitions.
List requisitions with asset items, count them, and update asset numbers.

Authentication: OAuth 2.0 Bearer token + apiKey header (own credentials)

Note: Requires the Asset Management feature to be enabled in the realm.
"""

import json
import os

import httpx
from fastmcp import FastMCP

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

BASE_URL = "https://openapi.ariba.com/api/asset-management/v1/prod"


def _make_auth() -> DirectAuthClient:
    return DirectAuthClient(
        client_id=os.getenv("API_ASSET_MANAGEMENT_CLIENT_ID", ""),
        client_secret=os.getenv("API_ASSET_MANAGEMENT_CLIENT_SECRET", ""),
        api_key=os.getenv("API_ASSET_MANAGEMENT_API_KEY", ""),
    )


def register(mcp: FastMCP, client: AribaClient) -> None:

    _auth = _make_auth()

    @mcp.tool(
        name="ariba_asset_list_requisitions",
        description=(
            "List SAP Ariba purchase requisitions that contain asset line items. "
            "Supports optional date filters and pagination."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def asset_list_requisitions(
        time_created: str | None = None,
        time_updated: str | None = None,
        top: int = 50,
        skip: int = 0,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            params: dict = {"realm": client.realm, "$top": top, "$skip": skip}
            filters: list[str] = []
            if time_created:
                filters.append(f"TimeCreated eq {time_created}")
            if time_updated:
                filters.append(f"TimeUpdated eq {time_updated}")
            if filters:
                params["$filter"] = ";".join(filters)
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{BASE_URL}/requisitions", headers=headers, params=params, timeout=60)
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_asset_count_requisitions",
        description=(
            "Return the total count of requisitions with asset line items. "
            "Useful before paginating through large result sets."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def asset_count_requisitions(
        time_created: str | None = None,
        time_updated: str | None = None,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            params: dict = {"realm": client.realm}
            filters: list[str] = []
            if time_created:
                filters.append(f"TimeCreated eq {time_created}")
            if time_updated:
                filters.append(f"TimeUpdated eq {time_updated}")
            if filters:
                params["$filter"] = ";".join(filters)
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{BASE_URL}/requisitions/$count", headers=headers, params=params, timeout=60)
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_asset_update_line_items",
        description=(
            "Assign or update asset numbers on asset line items inside a requisition. "
            "Use after a requisition is approved to tag physical assets to line items."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def asset_update_line_items(
        requisition_id: str,
        number_in_collection: int,
        asset_number: str,
        serial_number: str | None = None,
        description: str | None = None,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            headers["Content-Type"] = "application/json"
            asset: dict = {"AssetNumber": asset_number}
            if serial_number:
                asset["SerialNumber"] = serial_number
            if description:
                asset["Description"] = description
            payload = {
                "requisitions": [{
                    "RequisitionId": requisition_id,
                    "AssetLineItems": [{
                        "NumberInCollection": number_in_collection,
                        "Action": "update",
                        "Assets": [asset],
                    }],
                }]
            }
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    f"{BASE_URL}/requisitions/batch/assets",
                    headers=headers,
                    params={"realm": client.realm},
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)
