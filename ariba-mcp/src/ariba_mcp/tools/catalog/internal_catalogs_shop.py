"""Internal Catalogs Shop API.

Owner: Anil
Prod URL: https://openapi.ariba.com/api/ariba-catalog-provider/v1/prod
Docs: https://help.sap.com/doc/9d210ac7774a48c7a50e064db3c932e9/cloud/en-US/index.html

Key endpoints:
  GET /Shops({shopID})            — Retrieve all items and facets from catalogs in a shop.
  GET /Shops({shopID})/Items      — Return all items and their details from catalogs in a shop.
  GET /Shops({shopID})/AutoComplete — Typeahead search: retrieve matching search suggestions.

Prerequisites:
  - Access to SAP Ariba Developer Portal to create an application and request API access.
  - All queries must be authenticated using OAuth authentication.
  - Search 3.0 must be enabled for your site via a Designated Support Contact case.

Authentication: OAuth 2.0 Bearer token + apiKey header
Response format: JSON
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

API_PATH = "ariba-catalog-provider/v1/prod"


def _base_url(client: AribaClient) -> str:
    """Resolve the configured production URL for this API."""
    return client._settings.resolve_api_url(
        client._settings.internal_catalogs_shop_production_url, API_PATH
    )


def _resolve_shop_id(client: AribaClient, shop_id: str | None) -> str | None:
    """Return the provided shop_id or fall back to the configured one."""
    return shop_id or client._settings.internal_catalogs_shop_id


MISSING_SHOP_ID_MSG = (
    "Missing required `shop_id`. Set `INTERNAL_CATALOGS_SHOP_ID` in `.env` "
    "or pass `shop_id` directly. SAP Ariba requires calls like "
    "`GET /Shops({shopID})` — calling `/Shops` alone is not supported."
)


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register Internal Catalogs Shop API tools."""

    @mcp.tool(
        name="ariba_catalog_internal_get_shop",
        description=(
            "Retrieve all items and facets from the catalogs in a specific internal shop. "
            "Calls GET /Shops({shopID}). "
            "If `shop_id` is not passed, falls back to `INTERNAL_CATALOGS_SHOP_ID` in `.env`."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def catalog_internal_get_shop(shop_id: str | None = None) -> str:
        try:
            shop_id = _resolve_shop_id(client, shop_id)
            if not shop_id:
                return MISSING_SHOP_ID_MSG
            url = f"{_base_url(client)}/Shops({shop_id})"
            result = await client.get(url, params={"realm": client.realm}, api_path=API_PATH)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_catalog_internal_get_items",
        description=(
            "Return all items and their details from the catalogs in a specific internal shop. "
            "Calls GET /Shops({shopID})/Items. "
            "Optionally filter by `search_term` and limit fields with `select_fields`. "
            "If `shop_id` is not passed, falls back to `INTERNAL_CATALOGS_SHOP_ID` in `.env`."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def catalog_internal_get_items(
        shop_id: str | None = None,
        search_term: str | None = None,
        select_fields: str | None = None,
    ) -> str:
        try:
            shop_id = _resolve_shop_id(client, shop_id)
            if not shop_id:
                return MISSING_SHOP_ID_MSG
            params: dict = {"realm": client.realm}
            if search_term:
                params["$search"] = search_term
            if select_fields:
                params["$select"] = select_fields
            url = f"{_base_url(client)}/Shops({shop_id})/Items"
            result = await client.get(url, params=params, api_path=API_PATH)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_catalog_internal_autocomplete",
        description=(
            "Typeahead search: retrieve matching search suggestions from an internal catalog shop. "
            "Calls GET /Shops({shopID})/AutoComplete. "
            "Requires Search 3.0 to be enabled on your site. "
            "If `shop_id` is not passed, falls back to `INTERNAL_CATALOGS_SHOP_ID` in `.env`."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def catalog_internal_autocomplete(
        search_term: str,
        shop_id: str | None = None,
    ) -> str:
        try:
            shop_id = _resolve_shop_id(client, shop_id)
            if not shop_id:
                return MISSING_SHOP_ID_MSG
            params: dict = {"realm": client.realm, "$search": search_term}
            url = f"{_base_url(client)}/Shops({shop_id})/AutoComplete"
            result = await client.get(url, params=params, api_path=API_PATH)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)#internal