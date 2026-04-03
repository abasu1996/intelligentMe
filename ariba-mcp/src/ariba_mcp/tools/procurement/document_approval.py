"""Document Approval API.

Owner: Anim
Prod URL: https://openapi.ariba.com/api/approval/v2/prod
Docs: https://help.sap.com/doc/f9cd5fe02da34e5a9c0ddd8161ee04d1/cloud/en-US/index.html

Key endpoints:
  GET  /approvables          — List all pending approvable documents (requisitions, invoices)
  GET  /approvables/{id}     — Get details for a specific approvable document
  GET  /changes              — Get list of document IDs whose approval nodes changed state
  PATCH /approvables/{id}    — Approve or Deny a document (action in request body)
  GET  /groups               — List approval groups
  GET  /groups/{groupId}     — Get details of a specific approval group

Authentication: OAuth 2.0 Bearer token + apiKey header
Response format: JSON

SAP Ariba Document Approval API allows a client application to:
  - List and retrieve requisitions, invoices, or user profiles pending approval
  - Approve or deny those documents on behalf of an authorized approver
  - Monitor changes in approval node states using the /changes polling endpoint
  - Look up approval groups and their members (including delegate support)

Rate limits: 300 req/sec · 800 req/min · 35,000 req/hr
"""

import json

import httpx
from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_PATH = "approval/v2/prod"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register all Document Approval API tools with the MCP server."""

    # ------------------------------------------------------------------
    # 1. List Pending Approvables
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_list_approvables",
        description=(
            "List all documents currently pending approval (requisitions, invoices, or user profiles). "
            "Supports OData-style filters: pass $filter to narrow results, e.g. "
            "\"ApprovableType eq 'Requisition'\". "
            "Optional: approver_id to filter by assigned approver, "
            "$top to limit results (default: server max), "
            "$skip for pagination offset, "
            "$count=true to include total count in response."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def list_approvables(
        filter: str | None = None,
        approver_id: str | None = None,
        top: int | None = None,
        skip: int | None = None,
        count: bool = False,
    ) -> str:
        """
        GET /approvables
        Returns all approvable documents waiting for approval action.

        ApprovableType values: Requisition, Invoice, UserProfile
        """
        try:
            url = f"{client.base_url}/{API_PATH}/approvables"
            headers = await client.auth.get_headers()

            params: dict = {"realm": client.realm}
            if filter:
                params["$filter"] = filter
            if approver_id:
                params["approverId"] = approver_id
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            if count:
                params["$count"] = "true"

            async with httpx.AsyncClient() as http:
                resp = await http.get(url, params=params, headers=headers, timeout=60)
                resp.raise_for_status()

            data = resp.json()
            return json.dumps(data, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------
    # 2. Get Approvable by ID
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_get_approvable",
        description=(
            "Get full details of a specific approvable document by its ID. "
            "Returns document metadata, approval graph nodes, approvers, "
            "line items, and current approval state. "
            "Pass the approvableId returned from ariba_approval_list_approvables or ariba_approval_get_changes."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_approvable(approvable_id: str) -> str:
        """
        GET /approvables/{approvableId}
        Returns detailed document info including approval graph structure.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/approvables/{approvable_id}"
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
    # 3. Get Approval Node Changes  (polling / change-tracking endpoint)
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_get_changes",
        description=(
            "Poll for approval node changes — returns IDs of documents whose approval "
            "nodes changed state (activated, approved, denied, withdrawn, etc.) since a given changeSequenceId. "
            "Start with changeSequenceId=0 for the first pull, then pass the highest "
            "changeSequenceId from the previous response to get only new changes. "
            "Optionally filter by approvable_type: 'Requisition' | 'Invoice' | 'UserProfile'. "
            "Changes are retained for 180 days."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def get_changes(
        change_sequence_id: int = 0,
        approvable_type: str | None = None,
        top: int | None = None,
    ) -> str:
        """
        GET /changes
        Tracks changes to approval nodes. Supports incremental polling via changeSequenceId.
        Use $filter: changeSequenceId gt <last_id> to only retrieve new changes.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/changes"
            headers = await client.auth.get_headers()

            # Build OData filter
            filter_parts = [f"changeSequenceId gt {change_sequence_id}"]
            if approvable_type:
                filter_parts.append(f"ApprovableType eq '{approvable_type}'")
            odata_filter = " and ".join(filter_parts)

            params: dict = {
                "realm": client.realm,
                "$filter": odata_filter,
            }
            if top is not None:
                params["$top"] = top

            async with httpx.AsyncClient() as http:
                resp = await http.get(url, params=params, headers=headers, timeout=60)
                resp.raise_for_status()

            data = resp.json()
            return json.dumps(data, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------
    # 4. Approve a Document
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_approve_document",
        description=(
            "Approve a pending requisition or invoice by its approvableId. "
            "The approver_id must be a valid SAP Ariba user who is currently an active approver "
            "on the document's approval node. An optional comment can be added. "
            "Returns the updated approval state. "
            "NOTE: Only call this when the user explicitly confirms they want to approve."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def approve_document(
        approvable_id: str,
        approver_id: str,
        comment: str = "",
    ) -> str:
        """
        PATCH /approvables/{approvableId}
        Body: { "action": "approve", "approverId": "...", "comment": "..." }
        """
        try:
            url = f"{client.base_url}/{API_PATH}/approvables/{approvable_id}"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            payload: dict = {
                "action": "approve",
                "approverId": approver_id,
            }
            if comment:
                payload["comment"] = comment

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()

            result = {
                "status": "approved",
                "approvableId": approvable_id,
                "approverId": approver_id,
                "response": resp.json() if resp.content else {},
            }
            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------
    # 5. Deny a Document
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_deny_document",
        description=(
            "Deny a pending requisition or invoice by its approvableId. "
            "The approver_id must be the active approver on the document. "
            "A denial reason/comment is strongly recommended. "
            "Returns the updated approval state. "
            "NOTE: Only call this when the user explicitly confirms they want to deny."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def deny_document(
        approvable_id: str,
        approver_id: str,
        comment: str = "",
    ) -> str:
        """
        PATCH /approvables/{approvableId}
        Body: { "action": "deny", "approverId": "...", "comment": "..." }
        """
        try:
            url = f"{client.base_url}/{API_PATH}/approvables/{approvable_id}"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            payload: dict = {
                "action": "deny",
                "approverId": approver_id,
            }
            if comment:
                payload["comment"] = comment

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()

            result = {
                "status": "denied",
                "approvableId": approvable_id,
                "approverId": approver_id,
                "response": resp.json() if resp.content else {},
            }
            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------
    # 6. List Approval Groups
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_list_groups",
        description=(
            "List all approval groups defined in the realm. "
            "Useful to find a group's groupId before using delegate approval endpoints. "
            "Returns group name, groupId, and member list."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def list_groups(
        top: int | None = None,
        skip: int | None = None,
    ) -> str:
        """
        GET /groups
        Returns all approval groups configured in the realm.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/groups"
            headers = await client.auth.get_headers()

            params: dict = {"realm": client.realm}
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip

            async with httpx.AsyncClient() as http:
                resp = await http.get(url, params=params, headers=headers, timeout=60)
                resp.raise_for_status()

            return json.dumps(resp.json(), default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------
    # 7. Get Approval Group by ID
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_get_group",
        description=(
            "Get details of a specific approval group by its groupId. "
            "Returns group name, members, and delegate configuration. "
            "Use ariba_approval_list_groups first to find the groupId."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_group(group_id: str) -> str:
        """
        GET /groups/{groupId}
        Returns details for a single approval group.
        """
        try:
            url = f"{client.base_url}/{API_PATH}/groups/{group_id}"
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
    # 8. Approve as Delegate
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_approve_as_delegate",
        description=(
            "Approve a document on behalf of the original approver, acting as their delegate. "
            "The delegate_id is the user performing the approval action. "
            "The original_approver_id is the user who delegated their authority. "
            "Both must be valid SAP Ariba users. "
            "NOTE: Delegate must already be configured in the SAP Ariba GUI before using this."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def approve_as_delegate(
        approvable_id: str,
        delegate_id: str,
        original_approver_id: str,
        comment: str = "",
    ) -> str:
        """
        PATCH /approvables/{approvableId}
        Body: { "action": "approve", "approverId": "<delegate>",
                "delegateForApproverId": "<original_approver>", "comment": "..." }
        """
        try:
            url = f"{client.base_url}/{API_PATH}/approvables/{approvable_id}"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            payload: dict = {
                "action": "approve",
                "approverId": delegate_id,
                "delegateForApproverId": original_approver_id,
            }
            if comment:
                payload["comment"] = comment

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()

            result = {
                "status": "approved_as_delegate",
                "approvableId": approvable_id,
                "delegateId": delegate_id,
                "originalApproverId": original_approver_id,
                "response": resp.json() if resp.content else {},
            }
            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)

    # ------------------------------------------------------------------
    # 9. Deny as Delegate
    # ------------------------------------------------------------------
    @mcp.tool(
        name="ariba_approval_deny_as_delegate",
        description=(
            "Deny a document on behalf of the original approver, acting as their delegate. "
            "The delegate_id is the user performing the denial. "
            "The original_approver_id is the user who delegated their authority. "
            "NOTE: Delegate must already be configured in the SAP Ariba GUI before using this."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def deny_as_delegate(
        approvable_id: str,
        delegate_id: str,
        original_approver_id: str,
        comment: str = "",
    ) -> str:
        """
        PATCH /approvables/{approvableId}
        Body: { "action": "deny", "approverId": "<delegate>",
                "delegateForApproverId": "<original_approver>", "comment": "..." }
        """
        try:
            url = f"{client.base_url}/{API_PATH}/approvables/{approvable_id}"
            headers = await client.auth.get_headers()
            headers["Content-Type"] = "application/json"

            payload: dict = {
                "action": "deny",
                "approverId": delegate_id,
                "delegateForApproverId": original_approver_id,
            }
            if comment:
                payload["comment"] = comment

            async with httpx.AsyncClient() as http:
                resp = await http.patch(
                    url,
                    params={"realm": client.realm},
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()

            result = {
                "status": "denied_as_delegate",
                "approvableId": approvable_id,
                "delegateId": delegate_id,
                "originalApproverId": original_approver_id,
                "response": resp.json() if resp.content else {},
            }
            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)
        

        