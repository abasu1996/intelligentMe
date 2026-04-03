"""Contract Terms Management API.,

Owner: Shabreen
Docs: https://help.sap.com/doc/3030e0a0f768498f91043a8abbc75ff1/cloud/en-US/index.html

This API allows developers to create and manage contract terms and contract
requests within SAP Ariba Contracts and SAP Ariba strategic sourcing solutions.

Endpoints implemented:
  GET  /contractWorkspaces/{contractId}/contractTerms  - retrieve contract terms for a workspace
  POST /contractWorkspaces/{contractId}/contractTerms  - create contract terms in a workspace
  GET  /contractRequests                               - retrieve contract requests (list)
  POST /contractRequests                               - create a new contract request

Prerequisites:
  - SAP Ariba Developer Portal access
  - SAP Ariba Contracts or strategic sourcing solution enabled
  - Contract workspace must exist before creating contract terms
  - OAuth authentication configured
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# TODO: Confirm exact API path slug from the Developer Portal
# "Environment Details" table on the discovery page for this API.
CONTRACT_TERMS_API = "https://openapi.ariba.com/api/contract-terms-management/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_contract_terms_get",
        description=(
            "Retrieve contract terms for a specific contract workspace from SAP Ariba. "
            "Returns the contract terms document associated with the given contract workspace ID."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def contract_terms_get(
        realm: str,
        contract_id: str,
        top: int | None = None,
        skip: int | None = None,
    ) -> str:
        """
        Args:
            realm:       Required. The target SAP Ariba realm (e.g. 'MyCompanyS4').
            contract_id: Required. The contract workspace ID to retrieve terms for.
            top:         Optional. Max records to return (default 10).
            skip:        Optional. Number of records to skip for pagination.
        """
        try:
            params: dict = {"realm": realm}
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            result = await client.fetch_resource(
                CONTRACT_TERMS_API,
                f"contractWorkspaces/{contract_id}/contractTerms",
                params,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_contract_terms_create",
        description=(
            "Create a new contract terms document in a specific contract workspace in SAP Ariba. "
            "Also creates a corresponding contract request in SAP Ariba Procurement solutions. "
            "The contract workspace must already exist before calling this tool."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def contract_terms_create(
        realm: str,
        contract_id: str,
        body: dict,
    ) -> str:
        """
        Args:
            realm:       Required. The target SAP Ariba realm.
            contract_id: Required. The contract workspace ID to create terms in.
            body:        Required. JSON payload with contract terms details.
                         On success, response contains contractTermsDocumentId
                         and contractRequestId.
        """
        try:
            result = await client.post_resource(
                CONTRACT_TERMS_API,
                f"contractWorkspaces/{contract_id}/contractTerms",
                {"realm": realm},
                body,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_contract_requests_list",
        description=(
            "Retrieve a list of contract requests from SAP Ariba. "
            "Supports filtering by requestId, status, or other OData filter fields."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def contract_requests_list(
        realm: str,
        filter: str | None = None,
        top: int | None = None,
        skip: int | None = None,
    ) -> str:
        """
        Args:
            realm:  Required. The target SAP Ariba realm.
            filter: Optional. OData $filter expression,
                    e.g. "status eq 'Approved'" or "requestId eq 'CR-1234'".
            top:    Optional. Max records to return (default 10).
            skip:   Optional. Number of records to skip for pagination.
        """
        try:
            params: dict = {"realm": realm}
            if filter:
                params["$filter"] = filter
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            result = await client.fetch_resource(
                CONTRACT_TERMS_API, "contractRequests", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_contract_request_create",
        description=(
            "Create a new contract request in SAP Ariba Procurement solutions. "
            "Use this to import contract request header details directly into the procurement solution."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def contract_request_create(
        realm: str,
        body: dict,
    ) -> str:
        """
        Args:
            realm: Required. The target SAP Ariba realm.
            body:  Required. JSON payload with contract request header details.
                   On success, response contains the new contractRequestId.
        """
        try:
            result = await client.post_resource(
                CONTRACT_TERMS_API,
                "contractRequests",
                {"realm": realm},
                body,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)			
