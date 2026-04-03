"""Create Procurement Workspace API.

Owner: Vanshika
Prod URL: https://openapi.ariba.com/api/procurement-workspace/v1/prod
Docs: https://help.sap.com/doc/bffc0be7c97c48c2b0f9a95fe215b3b7/cloud/en-US/index.html

Summary:
    For buying organizations with SAP Ariba Procurement solutions that want to use
    an external application to create procurement workspace projects in SAP Ariba Sourcing.
    Procurement workspaces act as a project-level repository to manage complex
    procurement activities (requisitions, invoices, compliance docs) as a structured
    project spanning multiple stakeholders.

Prerequisites (must be configured in SAP Ariba Sourcing BEFORE using this API):
    1. Site must have "Ariba Procure To Pay" entitlement.
    2. Procurement workspace project templates must exist in Ariba Sourcing.
    3. Header forms & process names must be configured (Forms Builder + process mapping).
    4. Calling user must be in "Procurement Project Creator" user group.
    5. Parameters Application.Procurement.enableProcurementWorkspace = Yes.

Endpoints covered:
  ── Workspace Management ───────────────────────────────────────────────────
  POST  /workspaces                             Create a new procurement workspace
  GET   /workspaces                             List / search procurement workspaces
  GET   /workspaces/{workspaceId}               Get details of a specific workspace
  PATCH /workspaces/{workspaceId}               Update header fields of a workspace
  PATCH /workspaces/{workspaceId}/state         Change the state of a workspace

  ── Template Discovery ─────────────────────────────────────────────────────
  GET   /templates                              List available workspace templates

  ── Workspace Documents (linked procurement docs) ──────────────────────────
  GET   /workspaces/{workspaceId}/documents     List documents linked to a workspace
  POST  /workspaces/{workspaceId}/documents     Link a document to a workspace

Authentication: OAuth 2.0 Bearer token + apiKey header
Response format: JSON

Workspace IDs: SAP Ariba auto-generates IDs prefixed with "WS" (e.g. WS1234567).

Key workflow:
    1. GET /templates                  → discover available templateId values
    2. POST /workspaces                → create workspace (returns workspaceId)
    3. GET  /workspaces/{id}           → verify workspace was created correctly
    4. PATCH /workspaces/{id}          → optionally update header fields
    5. GET  /workspaces/{id}/documents → track linked procurement documents
    6. PATCH /workspaces/{id}/state    → advance workspace state when ready

Workspace states:
    Draft → In Review → Approved → Active → Closed
    (Available transitions depend on your realm's process configuration)

IMPORTANT: Custom header fields must be configured in the realm before being
passed via the API. Passing unknown custom fields returns HTTP 400.
"""

import json
import os

import httpx
from fastmcp import FastMCP

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://openapi.ariba.com/api/procurement-workspace/v1/prod"


def _make_auth() -> DirectAuthClient:
    return DirectAuthClient(
        client_id=os.getenv("PROCUREMENT_WORKSPACE_CLIENT_ID", ""),
        client_secret=os.getenv("PROCUREMENT_WORKSPACE_CLIENT_SECRET", ""),
        api_key=os.getenv("PROCUREMENT_WORKSPACE_API_KEY", ""),
    )

# Valid workspace states for the state-change endpoint
WORKSPACE_STATES = [
    "Draft",
    "Submitted",
    "InReview",
    "Approved",
    "Active",
    "Closed",
    "Terminated",
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register all Create Procurement Workspace API tools with the MCP server."""

    _auth = _make_auth()

    # ======================================================================
    # 1. List Available Templates
    # ======================================================================

    @mcp.tool(
        name="ariba_procure_workspace_list_templates",
        description=(
            "List all available procurement workspace templates configured in the realm. "
            "Returns template IDs and names. "
            "ALWAYS call this first before creating a workspace to discover valid templateId values. "
            "Templates are created and maintained in SAP Ariba Sourcing by a Procurement Admin. "
            "Template IDs follow the pattern: SYS1234 (system-generated)."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def list_templates() -> str:
        """
        GET /templates
        Returns all workspace templates available for the calling user's realm.
        """
        try:
            url = f"{BASE_URL}/templates"
            headers = await _auth.get_headers()

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
    # 2. Create Procurement Workspace
    # ======================================================================

    @mcp.tool(
        name="ariba_procure_workspace_create",
        description=(
            "Create a new procurement workspace project in SAP Ariba Sourcing. "
            "This is the core action of this API. "
            "\n\nRequired fields:"
            "\n  - title        : workspace name (no special chars: \\ / : ? * | < > \")"
            "\n  - template_id  : templateId from ariba_procure_workspace_list_templates"
            "\n  - process_name : the process name mapped to the header form in Ariba Admin"
            "\n  - owner_id     : Ariba username of the workspace owner"
            "\n\nOptional fields:"
            "\n  - description       : free-text description"
            "\n  - supplier_id       : supplier system ID (for procurement workspaces)"
            "\n  - department        : department name"
            "\n  - commodity_codes   : list of UNSPSC commodity code strings"
            "\n  - regions           : list of region code strings"
            "\n  - custom_fields     : dict of custom header field name → value pairs "
            "(must be pre-configured in realm)"
            "\n\nReturns the created workspaceId (prefixed WS, e.g. 'WS1234567')."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def create_workspace(
        title: str,
        template_id: str,
        process_name: str,
        owner_id: str,
        description: str | None = None,
        supplier_id: str | None = None,
        department: str | None = None,
        commodity_codes: list[str] | None = None,
        regions: list[str] | None = None,
        custom_fields: dict | None = None,
    ) -> str:
        """
        POST /workspaces
        Creates a new procurement workspace project using the given template.
        Returns workspaceId on success.
        """
        try:
            url = f"{BASE_URL}/workspaces"
            headers = await _auth.get_headers()
            headers["Content-Type"] = "application/json"

            # Build request body — only include fields that were provided
            body: dict = {
                "title": title,
                "templateId": template_id,
                "processName": process_name,
                "owner": {"userId": owner_id},
            }

            if description:
                body["description"] = description

            if supplier_id:
                body["supplier"] = {"systemId": supplier_id}

            if department:
                body["department"] = department

            if commodity_codes:
                body["commodityCodes"] = [
                    {"domain": "unspsc", "uniqueName": code}
                    for code in commodity_codes
                ]

            if regions:
                body["regions"] = [{"uniqueName": r} for r in regions]

            if custom_fields:
                # Custom fields are passed as a flat dict: fieldName → value
                body["customFields"] = custom_fields

            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=body,
                    timeout=60,
                )
                resp.raise_for_status()

            response_data = resp.json()

            result = {
                "status": "created",
                "workspaceId": response_data.get("workspaceId") or response_data.get("id"),
                "title": title,
                "templateId": template_id,
                "processName": process_name,
                "response": response_data,
            }
            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ======================================================================
    # 3. List / Search Workspaces
    # ======================================================================

    @mcp.tool(
        name="ariba_procure_workspace_list",
        description=(
            "List and search procurement workspaces with optional OData-style $filter. "
            "Filter examples:"
            "\n  - \"status eq 'Active'\""
            "\n  - \"owner eq 'jsmith@company.com'\""
            "\n  - \"title eq 'IT Procurement Q1 2025'\""
            "\n  - \"createdDate ge '2024-01-01T00:00:00Z'\""
            "\nWorkspace status values: Draft | Submitted | InReview | Approved | Active | Closed | Terminated"
            "\nUse top and skip for pagination."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def list_workspaces(
        filter: str | None = None,
        top: int = 20,
        skip: int = 0,
    ) -> str:
        """
        GET /workspaces
        Returns all procurement workspaces matching filter criteria.
        """
        try:
            url = f"{BASE_URL}/workspaces"
            headers = await _auth.get_headers()

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

    # ======================================================================
    # 4. Get Workspace by ID
    # ======================================================================

    @mcp.tool(
        name="ariba_procure_workspace_get",
        description=(
            "Get full details of a specific procurement workspace by its workspaceId. "
            "Returns: title, status, owner, templateId, processName, description, "
            "supplier, department, commodityCodes, regions, custom header fields, "
            "created/modified timestamps, and linked document counts. "
            "Workspace IDs are prefixed with 'WS' (e.g. 'WS1234567'). "
            "Use ariba_procure_workspace_list to find a workspaceId."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_workspace(workspace_id: str) -> str:
        """
        GET /workspaces/{workspaceId}
        Returns full header details for a single procurement workspace.
        """
        try:
            url = f"{BASE_URL}/workspaces/{workspace_id}"
            headers = await _auth.get_headers()

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
    # 5. Update Workspace Header Fields
    # ======================================================================

    @mcp.tool(
        name="ariba_procure_workspace_update",
        description=(
            "Update the header fields of an existing procurement workspace. "
            "Pass workspaceId and a dict of fields to change. "
            "Updatable standard fields: title, description, owner (userId), "
            "department, supplierId, commodityCodes, regions. "
            "Custom fields can also be updated via the custom_fields parameter. "
            "Only fields you provide will be changed — all others remain untouched. "
            "NOTE: Workspaces in 'Closed' or 'Terminated' state cannot be updated."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def update_workspace(
        workspace_id: str,
        title: str | None = None,
        description: str | None = None,
        owner_id: str | None = None,
        department: str | None = None,
        supplier_id: str | None = None,
        commodity_codes: list[str] | None = None,
        regions: list[str] | None = None,
        custom_fields: dict | None = None,
    ) -> str:
        """
        PATCH /workspaces/{workspaceId}
        Partial update — only supplied fields are changed.
        """
        try:
            url = f"{BASE_URL}/workspaces/{workspace_id}"
            headers = await _auth.get_headers()
            headers["Content-Type"] = "application/json"

            body: dict = {}

            if title:
                body["title"] = title
            if description is not None:
                body["description"] = description
            if owner_id:
                body["owner"] = {"userId": owner_id}
            if department:
                body["department"] = department
            if supplier_id:
                body["supplier"] = {"systemId": supplier_id}
            if commodity_codes:
                body["commodityCodes"] = [
                    {"domain": "unspsc", "uniqueName": code}
                    for code in commodity_codes
                ]
            if regions:
                body["regions"] = [{"uniqueName": r} for r in regions]
            if custom_fields:
                body["customFields"] = custom_fields

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=body,
                    timeout=60,
                )
                resp.raise_for_status()

            return json.dumps(
                resp.json() if resp.content else {"status": "updated", "workspaceId": workspace_id},
                default=str,
            )

        except Exception as e:
            return handle_ariba_error(e)

    # ======================================================================
    # 6. Change Workspace State
    # ======================================================================

    @mcp.tool(
        name="ariba_procure_workspace_change_state",
        description=(
            "Change the state (status) of a procurement workspace. "
            "Valid target states: Draft | Submitted | InReview | Approved | Active | Closed | Terminated. "
            "Common transitions:\n"
            "  Draft      → Submitted  (send for review)\n"
            "  Submitted  → Approved   (approve the workspace)\n"
            "  Approved   → Active     (activate for procurement use)\n"
            "  Active     → Closed     (close completed workspace)\n"
            "  Any state  → Terminated (cancel the workspace)\n"
            "Available transitions depend on your realm's process configuration. "
            "An optional comment can be included with the state change."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def change_workspace_state(
        workspace_id: str,
        new_state: str,
        comment: str = "",
    ) -> str:
        """
        PATCH /workspaces/{workspaceId}/state
        Body: { "state": "<newState>", "comment": "<optional>" }
        Transitions the workspace to the specified state.
        """
        try:
            url = f"{BASE_URL}/workspaces/{workspace_id}/state"
            headers = await _auth.get_headers()
            headers["Content-Type"] = "application/json"

            body: dict = {"state": new_state}
            if comment:
                body["comment"] = comment

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=body,
                    timeout=60,
                )
                resp.raise_for_status()

            result = {
                "status": "state_changed",
                "workspaceId": workspace_id,
                "newState": new_state,
                "response": resp.json() if resp.content else {},
            }
            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ======================================================================
    # 7. List Documents Linked to Workspace
    # ======================================================================

    @mcp.tool(
        name="ariba_procure_workspace_list_documents",
        description=(
            "List all procurement documents linked to a specific workspace. "
            "Returns document IDs, types, titles, and statuses for documents "
            "such as: requisitions, purchase orders, invoices, and compliance docs "
            "that have been associated with this workspace. "
            "Use this to track all procurement activity grouped under the workspace project."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def list_workspace_documents(
        workspace_id: str,
        top: int = 50,
        skip: int = 0,
    ) -> str:
        """
        GET /workspaces/{workspaceId}/documents
        Returns all documents associated with the workspace.
        """
        try:
            url = f"{BASE_URL}/workspaces/{workspace_id}/documents"
            headers = await _auth.get_headers()

            params: dict = {
                "realm": client.realm,
                "$top": top,
                "$skip": skip,
            }

            async with httpx.AsyncClient() as http:
                resp = await http.get(url, params=params, headers=headers, timeout=60)
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ======================================================================
    # 8. Link a Document to a Workspace
    # ======================================================================

    @mcp.tool(
        name="ariba_procure_workspace_link_document",
        description=(
            "Link an existing procurement document to a workspace. "
            "Supported document types: Requisition | PurchaseOrder | Invoice | Receipt. "
            "Pass the workspaceId and the document_id + document_type of the document to link. "
            "This enables the workspace to serve as the central project hub "
            "for all related procurement activity. "
            "NOTE: The document must already exist in SAP Ariba before linking."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def link_document_to_workspace(
        workspace_id: str,
        document_id: str,
        document_type: str,
    ) -> str:
        """
        POST /workspaces/{workspaceId}/documents
        Body: { "documentId": "...", "documentType": "..." }
        Links an existing document to the workspace.
        """
        try:
            url = f"{BASE_URL}/workspaces/{workspace_id}/documents"
            headers = await _auth.get_headers()
            headers["Content-Type"] = "application/json"

            body: dict = {
                "documentId": document_id,
                "documentType": document_type,
            }

            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=body,
                    timeout=60,
                )
                resp.raise_for_status()

            result = {
                "status": "linked",
                "workspaceId": workspace_id,
                "documentId": document_id,
                "documentType": document_type,
                "response": resp.json() if resp.content else {},
            }
            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)