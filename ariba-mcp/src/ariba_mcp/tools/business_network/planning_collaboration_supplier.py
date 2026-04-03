"""Planning Collaboration Supplier API.,

Owner: Shabreen
Docs: https://help.sap.com/doc/debc8145f35b4db686e9748d29a9a453/cloud/en-US/index.html

This API is for SAP Business Network suppliers with SAP Business Network for
Supply Chain enablement who want to retrieve planning information directly
from SAP Business Network.

Endpoints implemented:
  GET /forecast                – retrieve forecast records from buyers
  GET /supplierInventory       – retrieve supplier inventory records
  GET /supplierManagedInventory – retrieve supplier-managed inventory (SMI) records

Prerequisites:
  - SAP Ariba Developer Portal access (Enterprise accounts only)
  - SAP Business Network for Supply Chain enablement
  - Forecast collaboration component enabled
  - Transaction rule "Allow suppliers to view forecast data" enabled
  - OAuth authentication configured
  - API Management enabled in Seller account settings
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# TODO: Confirm exact API path slug from the Developer Portal
# "Environment Details" table on the discovery page for this API.
PLANNING_COLLABORATION_SUPPLIER_API = "planning-collaboration-supplier/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_supplier_forecast_list",
        description=(
            "Retrieve forecast records from buyers via the Planning Collaboration Supplier API. "
            "Returns demand forecast information for a supplier's Business Network user ID (ANID). "
            "Supports filtering by buyer, plant, part numbers, and date range."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def supplier_forecast_list(
        anid: str,
        filter: str | None = None,
        orderby: str | None = None,
        top: int | None = None,
        skip: int | None = None,
        count: bool | None = None,
    ) -> str:
        """
        Args:
            anid:    Required. The supplier's Ariba Network ID (ANID),
                     passed as X-ARIBA-NETWORK-ID header via the client.
            filter:  Optional. OData $filter expression. Supported fields:
                       buyerANID, plantId, vendorId,
                       buyerPartNumber, supplierPartNumber,
                       startDate (YYYY-MM-DDThh:mm), endDate (YYYY-MM-DDThh:mm)
                     Note: startDate/endDate period is limited to 1 year.
                     Example:
                       "buyerPartNumber eq Part007 and buyerANID eq AN01000000001
                        and startDate eq 2024-01-01T00:00 and endDate eq 2024-12-31T00:00"
            orderby: Optional. Sort expression, e.g. "buyerPartNumber asc".
            top:     Optional. Max records per page (default 10, max 100).
            skip:    Optional. Number of records to skip (offset, default 0).
            count:   Optional. If True, includes total record count in response.
        """
        try:
            params: dict = {"anid": anid}
            if filter:
                params["$filter"] = filter
            if orderby:
                params["$orderby"] = orderby
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            if count is not None:
                params["$count"] = str(count).lower()
            result = await client.fetch_resource(
                PLANNING_COLLABORATION_SUPPLIER_API, "forecast", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_supplier_inventory_list",
        description=(
            "Retrieve supplier inventory records from the Planning Collaboration Supplier API. "
            "Returns inventory information for a supplier's Business Network user ID (ANID). "
            "Supports filtering by buyer, plant, and part numbers."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def supplier_inventory_list(
        anid: str,
        filter: str | None = None,
        orderby: str | None = None,
        top: int | None = None,
        skip: int | None = None,
        count: bool | None = None,
    ) -> str:
        """
        Args:
            anid:    Required. The supplier's Ariba Network ID (ANID).
            filter:  Optional. OData $filter expression. Supported fields:
                       buyerANID, plantId, vendorId,
                       buyerPartNumber, supplierPartNumber
                     Note: startDate/endDate are NOT supported for this endpoint.
                     Example:
                       "buyerPartNumber eq siPart108 and buyerANID eq AN01000000001"
            orderby: Optional. Sort expression, e.g. "supplierPartNumber desc".
            top:     Optional. Max records per page (default 10, max 100).
            skip:    Optional. Number of records to skip (offset, default 0).
            count:   Optional. If True, includes total record count in response.
        """
        try:
            params: dict = {"anid": anid}
            if filter:
                params["$filter"] = filter
            if orderby:
                params["$orderby"] = orderby
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            if count is not None:
                params["$count"] = str(count).lower()
            result = await client.fetch_resource(
                PLANNING_COLLABORATION_SUPPLIER_API, "supplierInventory", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_supplier_managed_inventory_list",
        description=(
            "Retrieve supplier-managed inventory (SMI) records from the Planning Collaboration Supplier API. "
            "Returns SMI information for a supplier's Business Network user ID (ANID). "
            "Supports filtering by buyer, plant, part numbers, and date range."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def supplier_managed_inventory_list(
        anid: str,
        filter: str | None = None,
        orderby: str | None = None,
        top: int | None = None,
        skip: int | None = None,
        count: bool | None = None,
    ) -> str:
        """
        Args:
            anid:    Required. The supplier's Ariba Network ID (ANID).
            filter:  Optional. OData $filter expression. Supported fields:
                       buyerANID, plantId, vendorId,
                       buyerPartNumber, supplierPartNumber,
                       startDate (YYYY-MM-DDThh:mm), endDate (YYYY-MM-DDThh:mm)
                     Note: startDate/endDate period is limited to 1 year.
                     Example:
                       "buyerPartNumber eq smiPart007 and buyerANID eq AN01000000001
                        and startDate eq 2024-01-01T00:00 and endDate eq 2024-12-31T00:00"
            orderby: Optional. Sort expression, e.g. "buyerPartNumber asc".
            top:     Optional. Max records per page (default 10, max 100).
            skip:    Optional. Number of records to skip (offset, default 0).
            count:   Optional. If True, includes total record count in response.
        """
        try:
            params: dict = {"anid": anid}
            if filter:
                params["$filter"] = filter
            if orderby:
                params["$orderby"] = orderby
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            if count is not None:
                params["$count"] = str(count).lower()
            result = await client.fetch_resource(
                PLANNING_COLLABORATION_SUPPLIER_API, "supplierManagedInventory", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)