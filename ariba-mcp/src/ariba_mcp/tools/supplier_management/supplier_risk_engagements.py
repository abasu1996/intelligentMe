"""Supplier Risk Engagements API.

Owner: Nitish SM
Prod URL: https://openapi.ariba.com/api/risk-engagement/v2/prod
Swagger: swagger (11).json
Docs: https://help.sap.com/doc/892703b130354abbb5013c3587a08cb2/cloud/en-US/270c23ef0b584430aaf0d932755d3089.html

Endpoints (all GET, all require realm):
  /engagements          — list (requires $filter with updatedDateTo)
  /engagements/{wsId}   — single engagement by workspace ID
  /issues               — list (requires $filter with updatedDateTo)
  /issues/{wsId}        — single issue by workspace ID
  /questionnaires       — list (requires $filter with updatedDateTo)
  /questionnaires/{wsId}— single questionnaire by workspace ID

$filter format: updatedDateFrom ge <date>Z and updatedDateTo le <date>Z
  updatedDateTo is MANDATORY. Date format: yyyy-MM-ddTHH:mm:ssZ (GMT)

Pagination: $top (max 100), $skip (offset or pageToken), $count=true

Authentication: OAuth 2.0 Bearer token + apiKey header (own credentials)
"""

import json
from datetime import datetime, timezone

import httpx
from fastmcp import FastMCP

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.config import get_settings
from ariba_mcp.errors import handle_ariba_error

BASE_URL = "https://openapi.ariba.com/api/risk-engagement/v2/prod"


def _make_auth() -> DirectAuthClient:
    s = get_settings()
    return DirectAuthClient(
        client_id=s.ariba_risk_client_id,
        client_secret=s.ariba_risk_client_secret,
        api_key=s.ariba_risk_api_key,
        timeout=s.request_timeout,
    )


def _build_filter(updated_since: str | None, updated_until: str | None) -> str:
    """Build $filter string. Both dates are required by the API."""
    if not updated_since:
        updated_since = "2020-01-01T00:00:00Z"
    if not updated_until:
        updated_until = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"updatedDateFrom ge {updated_since} and updatedDateTo le {updated_until}"


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register Supplier Risk Engagements API tools."""

    _auth = _make_auth()

    # ── Engagements ──

    @mcp.tool(
        name="ariba_risk_engagement_list",
        description=(
            "List risk engagement projects from the realm. "
            "Returns workspaceId, title, status, smVendorId, and updatedDate for each. "
            "Optional filters: updated_since / updated_until (format: 2024-01-01T00:00:00Z). "
            "Supports pagination: top (page size, max 100) and skip (offset). "
            "Set count=True to include total record count in response."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_engagements(
        updated_since: str | None = None,
        updated_until: str | None = None,
        top: int = 100,
        skip: int = 0,
        count: bool = True,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            params: dict = {
                "realm": client.realm,
                "$filter": _build_filter(updated_since, updated_until),
                "$top": str(top),
                "$skip": str(skip),
                "$count": str(count).lower(),
            }
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{BASE_URL}/engagements", headers=headers, params=params, timeout=60)
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_risk_engagement_get",
        description=(
            "Get full details of a single risk engagement project by workspace ID "
            "(e.g. 'WS4889266012'). Returns title, status, inherent/residual risk scores, "
            "supplier info, commodities, tasks, controls, assessments, issues, and "
            "domain-based risk scores."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_engagement(workspace_id: str) -> str:
        try:
            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/engagements/{workspace_id}",
                    headers=headers,
                    params={"realm": client.realm},
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    # ── Issues ──

    @mcp.tool(
        name="ariba_risk_issue_list",
        description=(
            "List risk issues across all engagements. "
            "Returns workspaceId, parentType, parentId, title, status, updatedDate. "
            "Optional filters: updated_since / updated_until (format: 2024-01-01T00:00:00Z). "
            "Supports pagination: top (max 100) and skip (offset)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_issues(
        updated_since: str | None = None,
        updated_until: str | None = None,
        top: int = 100,
        skip: int = 0,
        count: bool = True,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            params: dict = {
                "realm": client.realm,
                "$filter": _build_filter(updated_since, updated_until),
                "$top": str(top),
                "$skip": str(skip),
                "$count": str(count).lower(),
            }
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{BASE_URL}/issues", headers=headers, params=params, timeout=60)
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_risk_issue_get",
        description=(
            "Get full details of a single risk issue by workspace ID. "
            "Returns title, description, status, assignee, severity, classification, "
            "risk domain/rating, resolution details, supplier info, and tasks."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_issue(workspace_id: str) -> str:
        try:
            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/issues/{workspace_id}",
                    headers=headers,
                    params={"realm": client.realm},
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    # ── Questionnaires ──

    @mcp.tool(
        name="ariba_risk_questionnaire_list",
        description=(
            "List risk questionnaires across all engagements. "
            "Returns workspaceId, title, status, sentDate, smVendorId, updatedDate. "
            "Optional filters: updated_since / updated_until (format: 2024-01-01T00:00:00Z). "
            "Supports pagination: top (max 100) and skip (offset)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_questionnaires(
        updated_since: str | None = None,
        updated_until: str | None = None,
        top: int = 100,
        skip: int = 0,
        count: bool = True,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            params: dict = {
                "realm": client.realm,
                "$filter": _build_filter(updated_since, updated_until),
                "$top": str(top),
                "$skip": str(skip),
                "$count": str(count).lower(),
            }
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{BASE_URL}/questionnaires", headers=headers, params=params, timeout=60)
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_risk_questionnaire_get",
        description=(
            "Get full details of a single risk questionnaire by workspace ID. "
            "Returns title, status, sent/due/expiration dates, supplier info, "
            "and all questions with answers (section, question, answerType, answer)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_questionnaire(workspace_id: str) -> str:
        try:
            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/questionnaires/{workspace_id}",
                    headers=headers,
                    params={"realm": client.realm},
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)
