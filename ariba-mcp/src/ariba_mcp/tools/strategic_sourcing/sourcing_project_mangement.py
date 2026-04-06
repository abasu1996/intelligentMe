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
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from ariba_mcp.auth import DirectAuthClient
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

BASE_URL = "https://openapi.ariba.com/api/sourcing-project-management/v2/prod"
BASE_URL_MS = "https://openapi.in.cloud.ariba.com/api/sourcing-project-management/v2/prod"


class GetSourcingProjectInput(BaseModel):
    project_id: str = Field(description="SAP Ariba sourcing project ID to retrieve")
    user: str = Field(description="SAP Ariba user for user-context authentication")
    password_adapter: str = Field(description="SAP Ariba password adapter for user-context authentication")


class ListSourcingProjectsInput(BaseModel):
    user: str = Field(description="SAP Ariba user for user-context authentication")
    password_adapter: str = Field(description="SAP Ariba password adapter for user-context authentication")
    created_date_from: str = Field(description="Start of the project created date range in ISO 8601 format")
    created_date_to: str = Field(description="End of the project created date range in ISO 8601 format")


def _make_auth() -> DirectAuthClient:
    return DirectAuthClient(
        client_id=os.getenv("PRANATHI_CLIENT_ID", ""),
        client_secret=os.getenv("PRANATHI_CLIENT_SECRET", ""),
        api_key=os.getenv("PRANATHI_API_KEY", ""),
    )

def _make_auth_ms() -> DirectAuthClient:
    return DirectAuthClient(
        client_id=os.getenv("MS_CLIENT_ID", ""),
        client_secret=os.getenv("MS_CLIENT_SECRET", ""),
        api_key=os.getenv("MS_API_KEY", ""),
    )


def register(mcp: FastMCP, client: AribaClient) -> None:

    _auth = _make_auth_ms()

    @mcp.tool(
        name="ariba_list_sourcing_projects",
        description=(
            "List sourcing projects from Ariba (RFQs, RFPs, events). "
            "Requires user and password_adapter for user-context auth. "
            "Also requires a filter_expr (OData $filter) — e.g. "
            "\"status eq 'Open'\" or \"projectType eq 'RFQ'\". "
            "Returns project IDs, titles, statuses, and owners."
            "Prompt the user for any missing parameters (user, password_adapter, filter_expr with date range) if not provided. "
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_sourcing_projects(
        user: str | None = None,
        password_adapter: str | None = None,
        filter_expr: str | None = None,
        page_token: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        try:
            if not user or not password_adapter or not filter_expr:
                if ctx is None:
                    return json.dumps(
                        {
                            "error": True,
                            "message": "user, password_adapter, and filter_expr are required.",
                        }
                    )

                elicitation = await ctx.elicit(
                    "Please provide the sourcing project credentials and date range to build the project filter.",
                    response_type=ListSourcingProjectsInput,
                )

                if elicitation.action == "decline":
                    return json.dumps(
                        {
                            "error": True,
                            "message": "User declined to provide the sourcing project list inputs.",
                        }
                    )
                if elicitation.action == "cancel":
                    return json.dumps(
                        {
                            "error": True,
                            "message": "User cancelled the sourcing project list input request.",
                        }
                    )

                user = user or elicitation.data.user
                password_adapter = password_adapter or elicitation.data.password_adapter
                if not filter_expr:
                    filter_expr = (
                        f"createdDate ge '{elicitation.data.created_date_from}' and "
                        f"createdDate le '{elicitation.data.created_date_to}'"
                    )

            headers = await _auth.get_headers()
            params: dict = {
                "realm": client.realm_ms,
                "user": user,
                "passwordAdapter": password_adapter,
                "$filter": filter_expr,
            }
            if page_token:
                params["pageToken"] = page_token
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL_MS}/projects",
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
            "Prompt the user for a specific project ID or multiple project IDs if not provided. "

        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_sourcing_project(
        project_id: str | None = None,
        user: str | None = None,
        password_adapter: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        try:
            if not project_id or not user or not password_adapter:
                if ctx is None:
                    return json.dumps(
                        {
                            "error": True,
                            "message": "project_id, user, and password_adapter are required.",
                        }
                    )

                elicitation = await ctx.elicit(
                    "Please provide the sourcing project details needed to fetch the project.",
                    response_type=GetSourcingProjectInput,
                )

                if elicitation.action == "decline":
                    return json.dumps(
                        {
                            "error": True,
                            "message": "User declined to provide the sourcing project inputs.",
                        }
                    )
                if elicitation.action == "cancel":
                    return json.dumps(
                        {
                            "error": True,
                            "message": "User cancelled the sourcing project input request.",
                        }
                    )

                project_id = project_id or elicitation.data.project_id
                user = user or elicitation.data.user
                password_adapter = password_adapter or elicitation.data.password_adapter

            headers = await _auth.get_headers()
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{BASE_URL_MS}/projects/{project_id}",
                    headers=headers,
                    params={
                        "realm": client.realm_ms,
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
                    f"{BASE_URL_MS}/projects",
                    headers=headers,
                    params={
                        "realm": client.realm_ms,
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
