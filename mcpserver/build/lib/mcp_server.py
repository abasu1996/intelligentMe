from mcp.server.fastmcp import FastMCP
#import requests
import json


# Create MCP server
mcp = FastMCP("S4-API-Agent")


LAST_API_RESPONSE = {}

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def call_api(endpoint: str, params: dict | None = None) -> dict:
    """
    Call an API endpoint and store the response in MCP context so Claude
    can learn from it and answer follow-up questions using the data.
    """
    try:
        response = .get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        # Save in global memory for Claude to access via resources
        LAST_API_RESPONSE["data"] = data
        LAST_API_RESPONSE["endpoint"] = endpoint

        return {
            "status": "success",
            "endpoint": endpoint,
            "records": len(data) if isinstance(data, list) else 1
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.resource("lastapi://response")
def get_last_api_response() -> str:
    """
    Return the most recently fetched API response
    so Claude can treat it as context.
    """
    if not LAST_API_RESPONSE:
        return "No API response available yet."

    return json.dumps(LAST_API_RESPONSE, indent=2)


@mcp.tool()
def fetch_product_data() -> dict:
    """Fetch product data from a mock S/4HANA service"""
    # data = {
    #     "product_id": 123,
    #     "name": "Sample Product",
    #     "price": 19.99
    # }

    # LAST_API_RESPONSE["data"] = data
    # LAST_API_RESPONSE["endpoint"] = "s4hana/mock"

    # return data


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."


if __name__ == "__main__":
    mcp.run()
