import json

from fastmcp import FastMCP
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

SOURCING_MDS_API = "https://openapi.ariba.com/api/sourcing-mds-search/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_master_data_retrieval",
        description=(
            "Retrieve master data from Ariba Strategic Sourcing. "
            "Fetch commodities, regions, departments,company code using search term."
        ),
        # 
        # annotations={
        #     "readonlyHint": True,
        #     "destructiveHint": False,
        #     "idempotentHint": True,
        #     "openWorldHint": True,
        #     ,
        # }
    )
    
    async def master_data_retrieval(
        search_term: str,
        page_token: str | None = None,
    ) -> str:
        try:
            filters = {"searchTerm": search_term}

            result = await client.fetch(
                SOURCING_MDS_API,
                params={
                    "filters": json.dumps(filters),
                    "pageToken": page_token
                }
            )

            return json.dumps(result, default=str)

        except Exception as e:
            return handle_ariba_error(e)