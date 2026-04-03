"""User Qualification API.

Owner: Anil
Prod URL: https://openapi.ariba.com/api/user-qualification/v1/prod
Docs: https://help.sap.com/doc/1d24538317664af48135fbf225ee924e/cloud/en-US/index.html

Key endpoints:
  POST /qualifications — Create user qualification records.
  PUT  /qualifications — Replace existing user qualification records.

  
Prerequisites:
  - User Qualification feature must be enabled for your site.
  - Import User Qualification Fields and Field Mappings must be configured by a Customer Administrator.
  - All queries must be authenticated using OAuth authentication.

Authentication: OAuth 2.0 Bearer token + apiKey header
Response format: JSON

Applies to: SAP Ariba Strategic Sourcing Suite; SAP Ariba Sourcing
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

API_PATH = "user-qualification/v1/prod"


def _base_url(client: AribaClient) -> str:
    """Resolve the configured production URL for this API."""
    return client._settings.resolve_api_url(
        client._settings.user_qualification_production_url, API_PATH
    )


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register User Qualification API tools."""

    @mcp.tool(
        name="ariba_user_qualification_create",
        description=(
            "Create user qualification records in SAP Ariba. "
            "Pass the full request body as a JSON string. "
            "Requires User Qualification feature enabled on your site. "
            "Applies to SAP Ariba Strategic Sourcing Suite and SAP Ariba Sourcing."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def user_qualification_create(payload_json: str) -> str:
        try:
            payload = json.loads(payload_json)
            url = f"{_base_url(client)}/qualifications"
            result = await client.post(url, json_body=payload, api_path=API_PATH)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_user_qualification_replace",
        description=(
            "Replace existing user qualification records in SAP Ariba. "
            "Pass the full request body as a JSON string. "
            "Requires User Qualification feature enabled on your site. "
            "Applies to SAP Ariba Strategic Sourcing Suite and SAP Ariba Sourcing."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def user_qualification_replace(payload_json: str) -> str:
        try:
            payload = json.loads(payload_json)
            url = f"{_base_url(client)}/qualifications"
            result = await client.put(url, json_body=payload, api_path=API_PATH)
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)