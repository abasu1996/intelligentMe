"""Order Change Requests API for Suppliers.,

Owner: Shabreen
Docs: https://help.sap.com/doc/a18df06872cc41dab5617273313a23d4/cloud/en-US/index.html

This API is for SAP Business Network suppliers with SAP Business Network for
Supply Chain enablement who want to extract change request information.

Endpoints implemented:
  GET  /changeRequests             – retrieve change requests received by the supplier
  GET  /changeRequestResponses     – retrieve responses sent by the supplier
  GET  /changeRequestConfirmations – retrieve confirmations sent to the supplier
  POST /changeRequests             – send change request information to the supplier
  POST /changeRequestConfirmations – send change request confirmation information

Prerequisites:
  - SAP Ariba Developer Portal access
  - SAP Business Network for Supply Chain enablement
  - OAuth authentication configured (handled by AribaClient)
  - API Management enabled in Seller account settings
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# TODO: Confirm exact API path slug from the Developer Portal
ORDER_CHANGE_REQUESTS_SUPPLIER_API = "order-change-requests-supplier/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_supplier_change_requests_list",
        description=(
            "List order change requests received by a supplier from Ariba Network. "
            "Get change request information associated with a supplier's "
            "Business Network user ID (ANID)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def supplier_change_requests_list(
        anid: str,
        filter: str | None = None,
        page_token: str | None = None,
    ) -> str:
        """
        Args:
            anid:       Required. The supplier's Ariba Network ID (ANID).
            filter:     Optional. OData $filter expression. Supported fields:
                          createdDateFrom, createdDateTo, purchaseOrderId,
                          purchaseOrderLineNumber, purchaseOrderScheduleLineNumber,
                          buyerPlantCode, buyerItemId, supplierOrgId
            page_token: Optional. Pagination token from a previous response.
        """
        try:
            params: dict = {"anid": anid}
            if filter:
                params["$filter"] = filter
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(
                ORDER_CHANGE_REQUESTS_SUPPLIER_API, "changeRequests", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_supplier_change_request_responses_list",
        description=(
            "List responses to order change requests sent by a supplier on Ariba Network. "
            "Get change request response information associated with a supplier's "
            "Business Network user ID (ANID)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def supplier_change_request_responses_list(
        anid: str,
        filter: str | None = None,
        page_token: str | None = None,
    ) -> str:
        """
        Args:
            anid:       Required. The supplier's Ariba Network ID (ANID).
            filter:     Optional. OData $filter expression to narrow results.
            page_token: Optional. Pagination token from a previous response.
        """
        try:
            params: dict = {"anid": anid}
            if filter:
                params["$filter"] = filter
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(
                ORDER_CHANGE_REQUESTS_SUPPLIER_API, "changeRequestResponses", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_supplier_change_request_confirmations_list",
        description=(
            "List order change request confirmations sent to a supplier on Ariba Network. "
            "Get confirmation information for change requests associated with a "
            "supplier's Business Network user ID (ANID)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def supplier_change_request_confirmations_list(
        anid: str,
        filter: str | None = None,
        page_token: str | None = None,
    ) -> str:
        """
        Args:
            anid:       Required. The supplier's Ariba Network ID (ANID).
            filter:     Optional. OData $filter expression to narrow results.
            page_token: Optional. Pagination token from a previous response.
        """
        try:
            params: dict = {"anid": anid}
            if filter:
                params["$filter"] = filter
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(
                ORDER_CHANGE_REQUESTS_SUPPLIER_API, "changeRequestConfirmations", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    # ── POST tools ────────────────────────────────────────────────────────────

    @mcp.tool(
        name="ariba_supplier_change_request_create",
        description=(
            "Send change request information to a supplier on Ariba Network. "
            "Use this to push change request data for a specified supplier's ANID."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def supplier_change_request_create(
        anid: str,
        body: dict,
    ) -> str:
        """
        Args:
            anid: Required. The supplier's Ariba Network ID (ANID).
            body: Required. JSON payload containing the change request details
                  (purchaseOrderId, lineItems, requested changes, etc.).
        """
        try:
            result = await client.post_resource(
                ORDER_CHANGE_REQUESTS_SUPPLIER_API,
                "changeRequests",
                {"anid": anid},
                body,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_supplier_change_request_confirmation_create",
        description=(
            "Send change request confirmation information to a supplier on Ariba Network. "
            "Use this to push confirmation data for change requests sent to a specified supplier's ANID."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def supplier_change_request_confirmation_create(
        anid: str,
        body: dict,
    ) -> str:
        """
        Args:
            anid: Required. The supplier's Ariba Network ID (ANID).
            body: Required. JSON payload containing the confirmation details
                  (changeRequestId, confirmationStatus, line items, etc.).
        """
        try:
            result = await client.post_resource(
                ORDER_CHANGE_REQUESTS_SUPPLIER_API,
                "changeRequestConfirmations",
                {"anid": anid},
                body,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
