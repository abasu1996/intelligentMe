
"""Ship Notice API for Suppliers.,

Owner: Shabreen
Docs: https://help.sap.com/doc/5e5dbefd5c02432298d2a5ec45805b95/cloud/en-US/index.html

This API is for SAP Business Network suppliers with SAP Business Network for
Supply Chain enablement who want to retrieve detailed ship-notice information
directly from SAP Business Network.

Endpoints implemented:
  GET /shipNotices  – retrieve ship notice headers for the supplier
  GET /items        – retrieve line items for a specific ship notice

Prerequisites:
  - SAP Ariba Developer Portal access
  - SAP Business Network for Supply Chain enablement
  - OAuth authentication configured
  - API Management enabled in Seller account settings
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# TODO: Confirm exact API path slug from the Developer Portal
# "Environment Details" table on the discovery page for this API.
SHIP_NOTICE_SUPPLIER_API = "ship-notice-supplier/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_supplier_ship_notices_list",
        description=(
            "List ship notices for a supplier from Ariba Network. "
            "Retrieves ship notice header information associated with a supplier's "
            "Business Network user ID (ANID). Supports date-range filtering and pagination."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def supplier_ship_notices_list(
        anid: str,
        filter: str | None = None,
        top: int | None = None,
        page_token: str | None = None,
    ) -> str:
        """
        Args:
            anid:       Required. The supplier's Ariba Network ID (ANID).
            filter:     Optional. OData $filter expression. Supported fields:
                          startDate, endDate, documentNumber, buyerANID,
                          supplierANID, purchaseOrderId
                        Example: "startDate eq '2024-01-01T00:00:00' and
                                  endDate eq '2024-03-31T00:00:00'"
            top:        Optional. Number of records to return per page (e.g. 15).
            page_token: Optional. Pagination token from a previous response.
        """
        try:
            params: dict = {"anid": anid}
            if filter:
                params["$filter"] = filter
            if top:
                params["$top"] = top
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(
                SHIP_NOTICE_SUPPLIER_API, "shipNotices", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_supplier_ship_notice_items_list",
        description=(
            "List line items for a specific ship notice from Ariba Network. "
            "Retrieves item-level details for a ship notice associated with a "
            "supplier's Business Network user ID (ANID)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def supplier_ship_notice_items_list(
        anid: str,
        filter: str | None = None,
        top: int | None = None,
        page_token: str | None = None,
    ) -> str:
        """
        Args:
            anid:       Required. The supplier's Ariba Network ID (ANID).
            filter:     Optional. OData $filter expression. Supported fields:
                          documentNumber, startDate, endDate, buyerANID,
                          supplierANID, purchaseOrderId, lineNumber
                        Example: "documentNumber eq '123456789' and
                                  startDate eq '2024-01-01T00:00:00' and
                                  endDate eq '2024-03-31T00:00:00'"
            top:        Optional. Number of records to return per page (e.g. 10).
            page_token: Optional. Pagination token from a previous response.
        """
        try:
            params: dict = {"anid": anid}
            if filter:
                params["$filter"] = filter
            if top:
                params["$top"] = top
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(
                SHIP_NOTICE_SUPPLIER_API, "items", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)