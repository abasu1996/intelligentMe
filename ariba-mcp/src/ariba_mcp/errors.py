"""Centralized error handling for Ariba API responses."""

import json

import httpx

_STATUS_MESSAGES = {
    400: "Bad request. Check date format (YYYY-MM-DDTHH:MM:SS) and ensure date range <= 31 days.",
    401: "Authentication failed. Check ARIBA_CLIENT_ID and ARIBA_CLIENT_SECRET.",
    403: "Access denied. Verify ARIBA_REALM and API permissions for this endpoint.",
    404: "Resource not found. Verify the document ID exists in this realm.",
    429: "Rate limit exceeded. Wait a moment and retry.",
    500: "Ariba server error. Retry later or check SAP status page.",
    503: "Ariba service unavailable. Retry later.",
}


def handle_ariba_error(e: Exception) -> str:
    """Convert an exception into a JSON error string with actionable guidance."""
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        message = _STATUS_MESSAGES.get(status, f"Ariba API returned HTTP {status}.")
        return json.dumps({"error": True, "status": status, "message": message})

    if isinstance(e, httpx.TimeoutException):
        return json.dumps({"error": True, "message": "Request timed out. Try a smaller date range or retry."})

    return json.dumps({"error": True, "message": f"Unexpected error: {type(e).__name__}: {e}"})
