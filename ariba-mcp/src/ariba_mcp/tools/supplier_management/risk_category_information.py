"""Risk Category Information API.,

Owner: Shabreen
Docs: https://help.sap.com/doc/3a9efde5fd074398b9ae284a478f98e4/cloud/en-US/index.html

This API allows developers to build applications that add and manage
supplier-level risk data within SAP Ariba Supplier Risk profiles.

Endpoints implemented:
  GET  /riskCategories    – retrieve list of defined risk categories in the realm
  POST /riskCategoryData  – add or update risk data for specific suppliers

Prerequisites:
  - SAP Ariba Developer Portal access
  - SAP Ariba Supplier Risk enabled
  - Risk exposure configured by a customer administrator
  - OAuth authentication configured (handled by AribaClient)
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# TODO: Confirm exact API path slug from the Developer Portal
RISK_CATEGORY_API = "https://openapi.ariba.com/api/risk-category-information/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_risk_categories_list",
        description=(
            "Retrieve a list of available risk categories defined in the SAP Ariba realm. "
            "Risk categories are used to classify supplier risk data (e.g., financial, "
            "compliance, sustainability). Use the returned category IDs when pushing "
            "risk data via ariba_risk_category_data_update."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def risk_categories_list(
        realm: str,
        top: int | None = None,
        skip: int | None = None,
    ) -> str:
        """
        Args:
            realm: Required. The target SAP Ariba realm (e.g. 'MyCompanyS4').
            top:   Optional. Max records per page (default 10).
            skip:  Optional. Number of records to skip (offset, default 0).
        """
        try:
            params: dict = {"realm": realm}
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            result = await client.fetch_resource(
                RISK_CATEGORY_API, "riskCategories", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_risk_category_data_update",
        description=(
            "Add or update supplier-level risk category data in SAP Ariba Supplier Risk. "
            "Pushes external risk scores or compliance data points into a supplier's "
            "risk profile under a specific risk category. Use ariba_risk_categories_list "
            "to find valid category IDs before calling this tool."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def risk_category_data_update(
        realm: str,
        body: dict,
    ) -> str:
        """
        Args:
            realm: Required. The target SAP Ariba realm.
            body:  Required. JSON payload containing risk data. Expected fields:
                     supplierId    – SM vendor ID or ERP vendor ID of the supplier
                     categoryId    – risk category ID (from ariba_risk_categories_list)
                     score         – numeric risk score
                     value         – string value or label for the risk field
                     effectiveDate – date the risk data takes effect (ISO 8601)
                   Example:
                     {
                       "supplierId": "SUP-001",
                       "categoryId": "financial-risk",
                       "score": 72,
                       "effectiveDate": "2024-01-01T00:00:00Z"
                     }
        """
        try:
            result = await client.post_resource(
                RISK_CATEGORY_API,
                "riskCategoryData",
                {"realm": realm},
                body,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)			
			