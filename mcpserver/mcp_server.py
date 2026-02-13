from mcp.server.fastmcp import FastMCP
import requests  # <-- now works once installed
import json

# Create MCP server
mcp = FastMCP("S4-API-Agent")

LAST_API_RESPONSE = {}


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def fetch_customer_data(endpoint: str, params: dict | None = None) -> dict:
    """
    you are an agent that fetches customer data from this https://services.odata.org/northwind/northwind.svc/Customers?$format=application/json API.
    user will ask you to give certain number of records and additional information. Without giving any additional dialogues,
    Store the most recent API response in LAST_API_RESPONSE for context retrieval. give proper response without additional dialogues. Based on the question it asks for, fetch the data from the endpoint and return a summary including the status of the request and number of records retrieved.
    """
    try:
        response = requests.get("https://services.odata.org/northwind/northwind.svc/Customers?$format=application/json", params=params)  # <-- FIXED
        response.raise_for_status()
        data = response.json()

        LAST_API_RESPONSE["data"] = data
        LAST_API_RESPONSE["endpoint"] = "https://services.odata.org/northwind/northwind.svc/Customers?$format=application/json"

        return {
            "status": "success",
            "endpoint": "https://services.odata.org/northwind/northwind.svc/Customers?$format=application/json",
            "records": len(data) if isinstance(data, list) else 1,
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
    """Example S/4 mock endpoint"""
    data = {
        "product_id": 123,
        "name": "Sample Product",
        "price": 19.99
    }

    LAST_API_RESPONSE["data"] = data
    LAST_API_RESPONSE["endpoint"] = "s4hana/mock"

    return data


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Personalized greeting"""
    return f"Hello, {name}!"


@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a professional, formal greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."


if __name__ == "__main__":
    mcp.run()
