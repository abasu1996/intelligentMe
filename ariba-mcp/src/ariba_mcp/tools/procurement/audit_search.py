"""Audit Search API.

Owner: Vanshika
Prod URL: https://openapi.ariba.com/api/audit-search/v2/prod
Docs: https://help.sap.com/doc/e42379dea91647fb8fcec25f8fdbddd3/cloud/en-US/index.html

Correct endpoint (discovered via testing):
  GET /audits?tenantId={realm}&auditType={type}&searchStartTime={date}

Required params:
  tenantId        — realm name (e.g. BrainBoxDSAPP-T)  [NOT "realm"]
  auditType       — one of: GenericAction, DataModification, Integration,
                    ConfigurationModification, Security, DataAccess
  searchStartTime — date in yyyy-MM-dd'T'HH:mm:ss+0000 format
                    must be within ~90 days of now

Authentication: OAuth 2.0 Bearer token + apiKey header (Vanshika audit creds)
"""

import json
import os
from datetime import datetime, timedelta, timezone

import httpx
from fastmcp import FastMCP

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

BASE_URL = "https://openapi.ariba.com/api/audit-search/v2/prod"

VALID_AUDIT_TYPES = [
    "GenericAction",
    "DataModification",
    "Integration",
    "ConfigurationModification",
    "Security",
    "DataAccess",
]


def _make_auth() -> DirectAuthClient:
    return DirectAuthClient(
        client_id=os.getenv("AUDIT_SEARCH_CLIENT_ID", ""),
        client_secret=os.getenv("AUDIT_SEARCH_CLIENT_SECRET", ""),
        api_key=os.getenv("AUDIT_SEARCH_API_KEY", ""),
    )


def _format_time(iso_str: str | None = None, hours_back: int | None = None) -> str:
    """Return date in yyyy-MM-dd'T'HH:mm:ss+0000 format required by the API."""
    if iso_str:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    elif hours_back:
        dt = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    else:
        dt = datetime.now(timezone.utc) - timedelta(days=7)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+0000")


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register Audit Search API tools."""

    _auth = _make_auth()

    @mcp.tool(
        name="ariba_audit_search",
        description=(
            "Search SAP Ariba audit records by type and time range. "
            "audit_type must be one of: GenericAction, DataModification, Integration, "
            "ConfigurationModification, Security, DataAccess. "
            "search_start_time is ISO-8601 (e.g. '2026-03-01T00:00:00Z') — must be within ~90 days. "
            "Defaults to last 7 days if not provided."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def search_audits(
        audit_type: str = "GenericAction",
        search_start_time: str | None = None,
        top: int = 50,
        skip: int = 0,
    ) -> str:
        try:
            if audit_type not in VALID_AUDIT_TYPES:
                return json.dumps({
                    "error": True,
                    "message": f"Invalid audit_type '{audit_type}'. Must be one of: {VALID_AUDIT_TYPES}",
                })
            headers = await _auth.get_headers()
            params: dict = {
                "tenantId": client.realm,
                "auditType": audit_type,
                "searchStartTime": _format_time(search_start_time),
                "$top": top,
                "$skip": skip,
            }
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{BASE_URL}/audits", headers=headers, params=params, timeout=60)
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_audit_search_recent",
        description=(
            "Search audit records from the last N hours (default 24). "
            "Quick way to check recent activity without specifying exact timestamps. "
            "audit_type: GenericAction | DataModification | Integration | "
            "ConfigurationModification | Security | DataAccess."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def search_recent(
        hours_back: int = 24,
        audit_type: str = "GenericAction",
        top: int = 50,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            params: dict = {
                "tenantId": client.realm,
                "auditType": audit_type,
                "searchStartTime": _format_time(hours_back=hours_back),
                "$top": top,
            }
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{BASE_URL}/audits", headers=headers, params=params, timeout=60)
                resp.raise_for_status()
            data = resp.json()
            audits = data.get("audits", [])
            return json.dumps({
                "hours_back": hours_back,
                "audit_type": audit_type,
                "count": len(audits),
                "audits": audits,
            }, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_audit_search_all_types",
        description=(
            "Search all 6 audit types at once for a given time range. "
            "Returns a summary with record counts per type and combined results. "
            "Useful for a quick compliance check across all audit categories."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def search_all_types(
        search_start_time: str | None = None,
        top_per_type: int = 20,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            start = _format_time(search_start_time)
            summary: dict = {}
            all_audits: list = []
            for atype in VALID_AUDIT_TYPES:
                params: dict = {
                    "tenantId": client.realm,
                    "auditType": atype,
                    "searchStartTime": start,
                    "$top": top_per_type,
                }
                async with httpx.AsyncClient() as http:
                    resp = await http.get(f"{BASE_URL}/audits", headers=headers, params=params, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    audits = data.get("audits", [])
                    summary[atype] = len(audits)
                    all_audits.extend(audits)
                else:
                    summary[atype] = f"error {resp.status_code}"
            return json.dumps({
                "searchStartTime": start,
                "summary": summary,
                "total": len(all_audits),
                "audits": all_audits,
            }, default=str)
        except Exception as e:
            return handle_ariba_error(e)
