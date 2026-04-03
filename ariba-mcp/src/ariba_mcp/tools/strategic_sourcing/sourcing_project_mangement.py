"""Sourcing Project Management API.

Owner: Pranathi
Prod URL: https://openapi.ariba.com/api/sourcing-project-management/v2/prod

List, get, and create sourcing projects (RFQs, RFPs, events).

Authentication: OAuth 2.0 Bearer token + apiKey header (Pranathi credentials)
Note: This API also requires user + passwordAdapter query params for user context.
"""

import json
import os

import httpx
from fastmcp import FastMCP

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

BASE_URL = "https://openapi.ariba.com/api/sourcing-project-management/v2/prod"


def _make_auth() -> DirectAuthClient:
    return DirectAuthClient(
        client_id=os.getenv("PRANATHI_CLIENT_ID", ""),
        client_secret=os.getenv("PRANATHI_CLIENT_SECRET", ""),
        api_key=os.getenv("PRANATHI_API_KEY", ""),
    )


def register(mcp: FastMCP, client: AribaClient) -> None:

    _auth = _make_auth()

    @mcp.tool(
        name="ariba_list_sourcing_projects",
        description=(
            "List sourcing projects from Ariba (RFQs, RFPs, events). "
            "Requires user and password_adapter for user-context auth. "
            "Also requires a filter_expr (OData $filter) — e.g. "
            "\"status eq 'Open'\" or \"projectType eq 'RFQ'\". "
            "Returns project IDs, titles, statuses, and owners."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_sourcing_projects(
        user: str,
        password_adapter: str,
        filter_expr: str,
        page_token: str | None = None,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            params: dict = {
                "realm": client.realm,
                "user": user,
                "passwordAdapter": password_adapter,
                "$filter": filter_expr,
            }
            if page_token:
                params["pageToken"] = page_token
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/projects",
                    headers=headers,
                    params=params,
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_get_sourcing_project",
        description=(
            "Get details of a specific sourcing project by project ID. "
            "Requires user and password_adapter for user-context auth. "
            "Returns full project details including events, participants, and timeline."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_sourcing_project(
        project_id: str,
        user: str,
        password_adapter: str,
    ) -> str:
        try:
            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL}/projects/{project_id}",
                    headers=headers,
                    params={
                        "realm": client.realm,
                        "user": user,
                        "passwordAdapter": password_adapter,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_create_sourcing_project",
        description=(
            "Create a new sourcing project in Ariba. "
            "Requires user and password_adapter for user-context auth. "
            "Pass project_data as a JSON string with project details "
            "(title, projectType, description, owner, etc.)."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
    async def create_sourcing_project(
        project_data: str,
        user: str,
        password_adapter: str,
    ) -> str:
        try:
            payload = json.loads(project_data)
            headers = await _auth.get_headers()
            headers["Content-Type"] = "application/json"
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    f"{BASE_URL}/projects",
                    headers=headers,
                    params={
                        "realm": client.realm,
                        "user": user,
                        "passwordAdapter": password_adapter,
                    },
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
            return json.dumps(resp.json(), default=str)
        except Exception as e:
            return handle_ariba_error(e)
