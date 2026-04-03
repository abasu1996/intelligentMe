import json

from fastmcp import FastMCP
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

PRICING_API = "https://openapi.ariba.com/api/pricing/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_get_pricing_details",
        description=(
            "Retrieve pricing details for items in Ariba. "
            "Used to fetch price based on supplier, item, or contract context. "
            # "Example filters: '{\"itemId\":\"123\",\"supplier\":\"ABC\"}'."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_pricing_details(
        filters: str | None = None,
        page_token: str | None = None,
    ) -> str:
        try:
            filter_dict = json.loads(filters) if filters else None
            
            # filter_dict = json.loads(filters)   # parse to dict
            # json.dumps(filter_dict)             # re-serialize to string

            # Generic fetch call (since no fetch_view here)
            result = await client.fetch(  
                PRICING_API,
                params={
                    "filters": json.dumps(filter_dict) if filter_dict else None,
                    "pageToken": page_token
                    ##handle if u get error
                }
            )

            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)