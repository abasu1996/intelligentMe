"""Contract Compliance API.

Owner: Vanshika
Prod URL: https://openapi.ariba.com/api/contract-compliance/v1/prod
Docs: https://help.sap.com/doc/3ef7b70c268149288e4d59f43a94b68b/cloud/en-US/index.html

Summary:
    For customer developers who want to create and manage contract data using
    the Contract Compliance API. Enables import, extraction, and accumulator
    management for compliance contracts and contract requests in SAP Ariba
    Procurement solutions.

Endpoints covered:
  ── Contract Requests ──────────────────────────────────────────────────────
  POST   /contractRequests                                 Create a contract request
  GET    /contractRequests                                 List / search contract requests
  GET    /contractRequests/{contractRequestId}             Get single contract request header
  PATCH  /contractRequests/{contractRequestId}             Update contract request header

  POST   /contractRequests/{contractRequestId}/lineitems  Add line items to a contract request
  GET    /contractRequests/{contractRequestId}/lineitems  Get line items of a contract request
  PATCH  /contractRequests/{contractRequestId}/lineitems  Update line items of a contract request
  DELETE /contractRequests/{contractRequestId}/lineitems  Delete line items from a contract request

  POST   /contractRequests/{contractRequestId}/lineitems/{numberInCollection}/pricingTerms
                                                          Add pricing terms to a line item
  GET    /contractRequests/{contractRequestId}/lineitems/{numberInCollection}/pricingTerms
                                                          Get pricing terms for a line item
  PATCH  /contractRequests/{contractRequestId}/lineitems/{numberInCollection}/pricingTerms
                                                          Update pricing terms for a line item
  DELETE /contractRequests/{contractRequestId}/lineitems/{numberInCollection}/pricingTerms
                                                          Delete pricing terms for a line item

  ── Contracts (read-only extraction + accumulator update) ──────────────────
  GET    /contracts                                        List / search contracts
  GET    /contracts/{contractId}                           Get single contract header
  GET    /contracts/{contractId}/lineitems                 Get line items of a contract
  PATCH  /contracts/{contractId}/accumulators              Update contract accumulators

Authentication: OAuth 2.0 Bearer token + apiKey header
Response format: JSON

Rate limits: Per minute: 10 requests · Per hour: 100 requests
             (this is a LOW rate-limit API — space your calls accordingly)

OData filter support:
  All list endpoints accept $filter with operators: eq, ne, gt, ge, lt, le, and, or
  Example: $filter=contractStatus eq 'Published' and supplierId eq 'SUP-001'

Key workflows (from SAP documentation):
  1. CREATE CONTRACTS  : POST /contractRequests → POST lineitems → POST pricingTerms
  2. EXTRACT CONTRACTS : GET /contracts ($filter) → GET /contracts/{id}/lineitems
  3. UPDATE ACCUMULATORS: GET /contracts ($filter) → PATCH /contracts/{id}/accumulators
  4. CHANGE CONTRACTS  : GET /contracts ($filter) → GET /contractRequests/{id}/lineitems
                        → PATCH lineitems / pricingTerms
"""

import json

import httpx
from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_PATH = "contract-compliance/v1/prod"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register all Contract Compliance API tools with the MCP server."""

    # ======================================================================
    # CONTRACT REQUESTS — Header
    # ======================================================================

    @mcp.tool(
        name="ariba_contract_compliance_create_request",
        description=(
            "Create a new contract request in SAP Ariba Procurement with header-level details. "
            "This is step 1 of the contract creation workflow. After creating the request, "
            "add line items using ariba_contract_compliance_add_lineitems, then optionally "
            "add pricing terms per line item. "
            "Required fields in header_data: title, currency, supplierId. "
            "Optional: effectiveDate, expirationDate, description, contractType, ownerId, etc. "
            "Returns the new contractRequestId needed for subsequent calls. "
            "NOTE: Submitting for approval must be done from the SAP Ariba UI."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def create_contract_request(header_data: dict) -> str:
        """
        POST /contractRequests
        Body: contract request header fields as a JSON object.
        Returns the created contractRequestId on success.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contractRequests"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=header_data,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_list_requests",
        description=(
            "List and search contract requests using OData-style filters. "
            "Supports $filter with operators: eq, ne, gt, ge, lt, le, and, or. "
            "Example filters:\n"
            "  - \"contractStatus eq 'Draft'\"\n"
            "  - \"supplierId eq 'SUP-001'\"\n"
            "  - \"effectiveDate ge '2024-01-01T00:00:00Z'\"\n"
            "  - \"expirationDate lt '2025-12-31T00:00:00Z'\"\n"
            "Use top and skip for pagination. "
            "Returns header-level fields including contractRequestId."       
        ),    
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    
    async def list_contract_requests(
        filter: str | None = None,
        top: int = 20,
        skip: int = 0,
    ) -> str:
        """
        GET /contractRequests
        Returns all contract requests matching filter criteria.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contractRequests"
            headers = await client.auth.get_headers()

            params: dict = {
                "realm": client.realm,
                "$top": top,
                "$skip": skip,
            }
            if filter:
                params["$filter"] = filter

            async with httpx.AsyncClient() as http:
                resp = await http.get(url, params=params, headers=headers, timeout=60)
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_get_request",
        description=(
            "Get header details of a specific contract request by its contractRequestId. "
            "Returns fields such as title, status, supplierId, effectiveDate, expirationDate, "
            "currency, ownerId, and the linked contractId (if contract has been approved). "
            "Use ariba_contract_compliance_list_requests to find contractRequestId first."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_contract_request(contract_request_id: str) -> str:
        """
        GET /contractRequests/{contractRequestId}
        Returns full header details for one contract request.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contractRequests/{contract_request_id}"
            headers = await client.auth.get_headers()

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_update_request",
        description=(
            "Update the header details of an existing contract request. "
            "Pass the contractRequestId and a dict of fields to update. "
            "Updatable fields include: title, description, effectiveDate, expirationDate, "
            "currency, supplierId, ownerId, and custom header fields. "
            "Only include fields you want to change — unspecified fields are not modified. "
            "NOTE: Only contract requests in 'Draft' status can be updated."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def update_contract_request(
        contract_request_id: str,
        update_data: dict,
    ) -> str:
        """
        PATCH /contractRequests/{contractRequestId}
        Body: partial contract request header fields to update.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contractRequests/{contract_request_id}"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=update_data,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json() if resp.content else {"status": "updated"}, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ======================================================================
    # CONTRACT REQUESTS — Line Items
    # ======================================================================

    @mcp.tool(
        name="ariba_contract_compliance_add_lineitems",
        description=(
            "Add line items to an existing contract request. "
            "This is step 2 of the contract creation workflow (after creating the request header). "
            "Pass contractRequestId and a list of line item dicts. "
            "Each line item should include: numberInCollection, description, quantity, unitPrice, "
            "unitOfMeasure, commodityCode, and optionally supplierId and partId. "
            "Returns the created line item details including numberInCollection for pricing terms."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def add_lineitems(
        contract_request_id: str,
        line_items: list[dict],
    ) -> str:
        """
        POST /contractRequests/{contractRequestId}/lineitems
        Body: list of line item objects.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contractRequests/{contract_request_id}/lineitems"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=line_items,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_get_lineitems",
        description=(
            "Get all line items of a contract request by its contractRequestId. "
            "Returns line item details including numberInCollection (needed for pricing terms), "
            "description, quantity, unit price, UOM, commodity code, and line item status. "
            "Use ariba_contract_compliance_get_request to find the contractRequestId first."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_request_lineitems(contract_request_id: str) -> str:
        """
        GET /contractRequests/{contractRequestId}/lineitems
        Returns all line items for the specified contract request.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contractRequests/{contract_request_id}/lineitems"
            headers = await client.auth.get_headers()

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_update_lineitems",
        description=(
            "Update one or more line items of a contract request. "
            "Pass contractRequestId and a list of line item dicts with updated fields. "
            "Each item MUST include numberInCollection to identify which line to update. "
            "Updatable fields: description, quantity, unitPrice, unitOfMeasure, commodityCode. "
            "Only include fields you want to change."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def update_lineitems(
        contract_request_id: str,
        line_items: list[dict],
    ) -> str:
        """
        PATCH /contractRequests/{contractRequestId}/lineitems
        Body: list of partial line item objects — must include numberInCollection.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contractRequests/{contract_request_id}/lineitems"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=line_items,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json() if resp.content else {"status": "updated"}, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_delete_lineitems",
        description=(
            "Delete line items from a contract request. "
            "Pass the contractRequestId and the numberInCollection values of lines to delete. "
            "If number_in_collection_list is empty or not provided, all line items are deleted. "
            "WARNING: This is destructive and cannot be undone. "
            "NOTE: Only contract requests in 'Draft' status support line item deletion."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def delete_lineitems(
        contract_request_id: str,
        number_in_collection_list: list[int] | None = None,
    ) -> str:
        """
        DELETE /contractRequests/{contractRequestId}/lineitems
        Deletes specified line items (or all if none specified).
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contractRequests/{contract_request_id}/lineitems"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            params: dict = {"realm": client.realm}
            if number_in_collection_list:
                # pass as comma-separated query param per SAP Ariba spec
                params["numberInCollection"] = ",".join(str(n) for n in number_in_collection_list)

            async with httpx.AsyncClient() as http:
                resp = await http.delete(
                    url,
                    params=params,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps({"status": "deleted", "contractRequestId": contract_request_id}, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ======================================================================
    # CONTRACT REQUESTS — Pricing Terms
    # ======================================================================

    @mcp.tool(
        name="ariba_contract_compliance_add_pricing_terms",
        description=(
            "Add pricing terms to a specific line item of a contract request. "
            "This is step 3 (optional) in the contract creation workflow. "
            "Pass contractRequestId, numberInCollection (line item number), "
            "and a list of pricing term dicts. "
            "Pricing term fields typically include: startDate, endDate, price, minQuantity, "
            "maxQuantity, currency, and discountPercentage."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def add_pricing_terms(
        contract_request_id: str,
        number_in_collection: int,
        pricing_terms: list[dict],
    ) -> str:
        """
        POST /contractRequests/{contractRequestId}/lineitems/{numberInCollection}/pricingTerms
        Body: list of pricing term objects.
        """
        try:
            url = (
                f"{client.base_url}/{API_PATH}/contractRequests"
                f"/{contract_request_id}/lineitems/{number_in_collection}/pricingTerms"
            )
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=pricing_terms,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_get_pricing_terms",
        description=(
            "Get pricing terms for a specific line item of a contract request. "
            "Pass contractRequestId and the numberInCollection of the target line item. "
            "Returns all pricing term records associated with that line item, "
            "including dates, price tiers, quantity ranges, and discount info."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_pricing_terms(
        contract_request_id: str,
        number_in_collection: int,
    ) -> str:
        """
        GET /contractRequests/{contractRequestId}/lineitems/{numberInCollection}/pricingTerms
        Returns all pricing terms for the specified line item.
        """
        try:
            url = (
                f"{client.base_url}/{API_PATH}/contractRequests"
                f"/{contract_request_id}/lineitems/{number_in_collection}/pricingTerms"
            )
            headers = await client.auth.get_headers()

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_update_pricing_terms",
        description=(
            "Update pricing terms for a specific line item of a contract request. "
            "Pass contractRequestId, numberInCollection, and a list of pricing term updates. "
            "Each pricing term update should identify the term to modify and include updated fields. "
            "Only fields provided will be updated."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def update_pricing_terms(
        contract_request_id: str,
        number_in_collection: int,
        pricing_terms: list[dict],
    ) -> str:
        """
        PATCH /contractRequests/{contractRequestId}/lineitems/{numberInCollection}/pricingTerms
        Body: list of partial pricing term updates.
        """
        try:
            url = (
                f"{client.base_url}/{API_PATH}/contractRequests"
                f"/{contract_request_id}/lineitems/{number_in_collection}/pricingTerms"
            )
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=pricing_terms,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json() if resp.content else {"status": "updated"}, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_delete_pricing_terms",
        description=(
            "Delete pricing terms from a specific line item of a contract request. "
            "Pass contractRequestId and numberInCollection of the target line item. "
            "WARNING: Deletes ALL pricing terms for that line item. This cannot be undone. "
            "Only contract requests in 'Draft' status support this operation."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def delete_pricing_terms(
        contract_request_id: str,
        number_in_collection: int,
    ) -> str:
        """
        DELETE /contractRequests/{contractRequestId}/lineitems/{numberInCollection}/pricingTerms
        Deletes all pricing terms for the specified line item.
        """
        try:
            url = (
                f"{client.base_url}/{API_PATH}/contractRequests"
                f"/{contract_request_id}/lineitems/{number_in_collection}/pricingTerms"
            )
            headers = await client.auth.get_headers()

            async with httpx.AsyncClient() as http:
                resp = await http.delete(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps({
                "status": "deleted",
                "contractRequestId": contract_request_id,
                "numberInCollection": number_in_collection,
            }, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ======================================================================
    # CONTRACTS — Extraction (read-only)
    # ======================================================================

    @mcp.tool(
        name="ariba_contract_compliance_list_contracts",
        description=(
            "List and search approved compliance contracts using OData-style $filter. "
            "Note: Contracts are created after a contract request completes its approval workflow. "
            "Supports $filter with: eq, ne, gt, ge, lt, le, and, or. "
            "Useful filter fields:\n"
            "  - contractStatus  : 'Published' | 'Expired' | 'Terminated' | 'Draft'\n"
            "  - supplierId      : supplier's ID\n"
            "  - effectiveDate   : ISO-8601 date\n"
            "  - expirationDate  : ISO-8601 date\n"
            "  - contractId      : specific contract ID\n"
            "Returns header fields including contractId, contractRequestId, status, and dates. "
            "IMPORTANT: Rate limit is 10 req/min and 100 req/hr — use filters to narrow results."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def list_contracts(
        filter: str | None = None,
        top: int = 20,
        skip: int = 0,
    ) -> str:
        """
        GET /contracts
        Returns contracts matching the filter. Always use $filter to avoid timeouts.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contracts"
            headers = await client.auth.get_headers()

            params: dict = {
                "realm": client.realm,
                "$top": top,
                "$skip": skip,
            }
            if filter:
                params["$filter"] = filter

            async with httpx.AsyncClient() as http:
                resp = await http.get(url, params=params, headers=headers, timeout=60)
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_get_contract",
        description=(
            "Get the header details of a specific compliance contract by its contractId. "
            "Returns: contractId, contractRequestId, title, status, supplierId, "
            "effectiveDate, expirationDate, currency, ownerId, and accumulated spend data. "
            "Use ariba_contract_compliance_list_contracts to find the contractId first."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_contract(contract_id: str) -> str:
        """
        GET /contracts/{contractId}
        Returns header-level details of one compliance contract.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contracts/{contract_id}"
            headers = await client.auth.get_headers()

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------

    @mcp.tool(
        name="ariba_contract_compliance_get_contract_lineitems",
        description=(
            "Get all line items of a specific compliance contract by contractId. "
            "Returns line item details such as description, quantity, unitPrice, "
            "unitOfMeasure, commodityCode, accumulatedAmount, accumulatedQuantity, and status. "
            "Useful for checking spend against contract line item limits (accumulator validation)."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_contract_lineitems(contract_id: str) -> str:
        """
        GET /contracts/{contractId}/lineitems
        Returns all line items for the specified compliance contract.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contracts/{contract_id}/lineitems"
            headers = await client.auth.get_headers()

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ======================================================================
    # CONTRACTS — Accumulator Update
    # ======================================================================

    @mcp.tool(
        name="ariba_contract_compliance_update_accumulators",
        description=(
            "Update the spend accumulators for a compliance contract. "
            "Accumulators track consumed amounts and quantities against contract limits. "
            "Pass the contractId and a list of accumulator update objects. "
            "Each accumulator update needs: numberInCollection (line item number), "
            "and one or more of: incrementalAmount, incrementalQuantity. "
            "IMPORTANT: You can pass NEGATIVE values to reduce accumulators "
            "(e.g. for returns or reversals). "
            "Use ariba_contract_compliance_get_contract_lineitems first to see current "
            "accumulatedAmount and accumulatedQuantity values."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def update_accumulators(
        contract_id: str,
        accumulator_updates: list[dict],
    ) -> str:
        """
        PATCH /contracts/{contractId}/accumulators
        Body: list of accumulator update objects with numberInCollection and delta values.

        Each accumulator update object:
        {
            "numberInCollection": 1,
            "incrementalAmount": 500.00,   # positive to add, negative to subtract
            "incrementalQuantity": 10      # optional, units consumed
        }
        """
        try:
            url = f"{client.base_url}/{API_PATH}/contracts/{contract_id}/accumulators"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=accumulator_updates,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(resp.json() if resp.content else {"status": "accumulators updated", "contractId": contract_id}, default=str)

        except Exception as e:
            return handle_ariba_error(e)