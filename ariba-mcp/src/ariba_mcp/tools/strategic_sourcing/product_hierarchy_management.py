"""Product Hierarchy Management API.

Owner: Shabreen
Docs: https://help.sap.com/doc/27b5932567614e0eb559242fe24f57c4/cloud/en-US/index.html

This API retrieves product questionnaires responded to by a supplier,
including details on associated sourcing events and line items.

Endpoints implemented:
  GET /productQuestionnaires              – list all product questionnaires
  GET /productQuestionnaires/{id}         – get a specific questionnaire by ID
  GET /productQuestionnaires/{id}/lineItems – get line items for a questionnaire

Prerequisites:
  - SAP Ariba Developer Portal access
  - OAuth authentication configured (handled by AribaClient)
  - SAP Ariba Sourcing solution enabled
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# Note the v2 version in the path — confirmed from API docs
PRODUCT_HIERARCHY_API = "product-hierarchy/v2/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_product_questionnaires_list",
        description=(
            "Retrieve a list of product questionnaires responded to by suppliers in SAP Ariba. "
            "Includes data about associated sourcing events. Supports filtering and pagination."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def product_questionnaires_list(
        realm: str,
        filter: str | None = None,
        top: int | None = None,
        skip: int | None = None,
        count: bool | None = None,
    ) -> str:
        """
        Args:
            realm:  Required. The target SAP Ariba realm (e.g. 'MyCompanyS4').
            filter: Optional. OData $filter expression. Supported fields:
                      supplierId, eventId, status, createdDate, updatedDate
                    Example: "supplierId eq 'SUP-001' and status eq 'Completed'"
            top:    Optional. Max records per page (default 10).
            skip:   Optional. Number of records to skip (offset, default 0).
            count:  Optional. If True, returns total count of matching records.
        """
        try:
            params: dict = {"realm": realm}
            if filter:
                params["$filter"] = filter
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            if count is not None:
                params["$count"] = str(count).lower()
            result = await client.fetch_resource(
                PRODUCT_HIERARCHY_API, "productQuestionnaires", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_product_questionnaire_get",
        description=(
            "Retrieve a specific product questionnaire by its ID from SAP Ariba. "
            "Returns full details of the questionnaire including supplier responses."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def product_questionnaire_get(
        realm: str,
        questionnaire_id: str,
    ) -> str:
        """
        Args:
            realm:             Required. The target SAP Ariba realm.
            questionnaire_id:  Required. The unique ID of the product questionnaire.
        """
        try:
            params: dict = {"realm": realm}
            result = await client.fetch_resource(
                PRODUCT_HIERARCHY_API,
                f"productQuestionnaires/{questionnaire_id}",
                params,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_product_questionnaire_line_items_list",
        description=(
            "Retrieve line items for a specific product questionnaire from SAP Ariba. "
            "Returns item-level details including commodity codes, descriptions, "
            "and sourcing event line item associations."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def product_questionnaire_line_items_list(
        realm: str,
        questionnaire_id: str,
        top: int | None = None,
        skip: int | None = None,
    ) -> str:
        """
        Args:
            realm:            Required. The target SAP Ariba realm.
            questionnaire_id: Required. The unique ID of the product questionnaire.
            top:              Optional. Max records per page (default 10).
            skip:             Optional. Number of records to skip (offset, default 0).
        """
        try:
            params: dict = {"realm": realm}
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            result = await client.fetch_resource(
                PRODUCT_HIERARCHY_API,
                f"productQuestionnaires/{questionnaire_id}/lineItems",
                params,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)