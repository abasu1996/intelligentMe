from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP


mcp = FastMCP(
    name="sap-ariba-sourcing",
    instructions=(
        "Use this server to call SAP Ariba Sourcing APIs. "
        "Configure runtime endpoints first with configure_ariba_runtime, then call the named "
        "project and event tools or the generic call_ariba_api tool."
    ),
)


_ENV_FILE = Path(__file__).with_name(".env")
_SENSITIVE_MARKERS = ("secret", "token", "password", "authorization", "api_key", "apikey")


def _strip_optional_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    env: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        env[key] = _strip_optional_quotes(value)
    return env


_ENV = _load_env_file(_ENV_FILE)


def _env_first(*keys: str) -> str:
    for key in keys:
        value = _ENV.get(key, "").strip()
        if value:
            return value
    return ""


def _env_bool(key: str, default: bool) -> bool:
    raw = _ENV.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(key: str, default: float) -> float:
    raw = _ENV.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_json_object(key: str) -> dict[str, Any]:
    raw = _ENV.get(key, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{key} must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{key} must decode to a JSON object.")
    return value


def _build_runtime_credentials_from_env() -> dict[str, str]:
    return {
        "realm": _env_first("ARIBA_REALM"),
        "api_key": _env_first("ARIBA_API_KEY"),
        "oauth_url": _env_first("ARIBA_OAUTH_URL"),
        "client_id": _env_first("ARIBA_CLIENT_ID"),
        "client_secret": _env_first("ARIBA_CLIENT_SECRET"),
    }


def _build_runtime_config_from_env() -> dict[str, Any]:
    return {
        "project_api_base_url": _env_first("ARIBA_PROJECT_API_BASE_URL", "ARIBA_API_URL").rstrip("/"),
        "event_api_base_url": _env_first("ARIBA_EVENT_API_BASE_URL", "ARIBA_API_URL").rstrip("/"),
        "default_query": _env_json_object("ARIBA_DEFAULT_QUERY_JSON"),
        "default_headers": _env_json_object("ARIBA_DEFAULT_HEADERS_JSON"),
        "timeout_seconds": _env_float("ARIBA_TIMEOUT_SECONDS", 30.0),
        "verify_ssl": _env_bool("ARIBA_VERIFY_SSL", True),
    }


_state: dict[str, Any] = {
    "config": _build_runtime_config_from_env(),
    "credentials": _build_runtime_credentials_from_env(),
    "last_response": None,
    "request_history": [],
}


def _parse_json_object(raw: str, field_name: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")
    return value


def _parse_json_payload(raw: str) -> Any:
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Allow raw strings for APIs that expect literal payload content.
        return raw


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None and v != ""})
    return merged


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(marker in normalized for marker in _SENSITIVE_MARKERS)


def _mask_value(value: Any) -> str:
    text = str(value)
    if len(text) <= 6:
        return "***"
    return f"{text[:3]}...{text[-2:]}"


def _sanitize_headers(headers: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in headers.items():
        sanitized[key] = _mask_value(value) if _is_sensitive_key(key) else value
    return sanitized


def _build_env_request_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    credentials = _state["credentials"]
    if credentials["realm"]:
        headers["realm"] = credentials["realm"]
    if credentials["api_key"]:
        headers["apiKey"] = credentials["api_key"]
    return headers


def _get_service_base_url(service: str) -> str:
    config = _state["config"]
    if service == "project_management":
        base_url = config["project_api_base_url"]
    elif service == "event_management":
        base_url = config["event_api_base_url"]
    else:
        raise ValueError(
            "service must be one of: project_management, event_management"
        )
    if not base_url:
        raise ValueError(
            f"No base URL configured for {service}. Call configure_ariba_runtime first."
        )
    return base_url


def _build_url(service: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base_url = _get_service_base_url(service)
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{_normalize_base_url(base_url)}{normalized_path}"


def _store_response(response_payload: dict[str, Any]) -> None:
    _state["last_response"] = response_payload
    history = _state["request_history"]
    history.append(
        {
            "method": response_payload["request"]["method"],
            "url": response_payload["request"]["url"],
            "status_code": response_payload["response"]["status_code"],
        }
    )
    if len(history) > 20:
        del history[0]


def _request(
    *,
    service: str,
    method: str,
    path: str,
    query_json: str = "{}",
    payload_json: str = "",
    headers_json: str = "{}",
) -> str:
    config = _state["config"]
    query = _merge_dicts(config["default_query"], _parse_json_object(query_json, "query_json"))
    headers = _merge_dicts(_build_env_request_headers(), config["default_headers"])
    headers = _merge_dicts(headers, _parse_json_object(headers_json, "headers_json"))
    payload = _parse_json_payload(payload_json)
    url = _build_url(service, path)

    request_kwargs: dict[str, Any] = {
        "method": method.upper(),
        "url": url,
        "params": query,
        "headers": headers,
    }
    if payload is not None and method.upper() not in {"GET", "DELETE"}:
        if isinstance(payload, (dict, list)):
            request_kwargs["json"] = payload
        else:
            request_kwargs["content"] = str(payload)

    try:
        with httpx.Client(
            verify=config["verify_ssl"],
            timeout=config["timeout_seconds"],
            follow_redirects=True,
        ) as client:
            response = client.request(**request_kwargs)
    except httpx.HTTPError as exc:
        error_payload = {
            "request": {
                "service": service,
                "method": method.upper(),
                "url": url,
                "headers": _sanitize_headers(headers),
                "query": query,
            },
            "response": {
                "status_code": None,
                "reason_phrase": "HTTP client error",
                "headers": {},
                "body": str(exc),
            },
        }
        _store_response(error_payload)
        return json.dumps(error_payload, indent=2, default=str)

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            response_body: Any = response.json()
        except ValueError:
            response_body = response.text
    else:
        response_body = response.text

    response_payload = {
        "request": {
            "service": service,
            "method": method.upper(),
            "url": str(response.request.url),
            "headers": _sanitize_headers(dict(response.request.headers)),
        },
        "response": {
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "headers": dict(response.headers),
            "body": response_body,
        },
    }
    _store_response(response_payload)
    return json.dumps(response_payload, indent=2, default=str)


@mcp.tool()
def configure_ariba_runtime(
    project_api_base_url: str = "",
    event_api_base_url: str = "",
    default_query_json: str = "",
    default_headers_json: str = "",
    timeout_seconds: float | None = None,
    verify_ssl: bool | None = None,
) -> str:
    """
    Configure SAP Ariba runtime details used by the server.

    Args:
        project_api_base_url: Base URL for Sourcing Project Management API requests.
        event_api_base_url: Base URL for Event Management API requests.
        default_query_json: JSON object with query parameters included on every request.
        default_headers_json: JSON object with headers included on every request.
        timeout_seconds: Request timeout in seconds.
        verify_ssl: Whether to verify TLS certificates.
    """
    current_config = _state["config"]
    _state["config"] = {
        "project_api_base_url": (
            _normalize_base_url(project_api_base_url)
            if project_api_base_url
            else current_config["project_api_base_url"]
        ),
        "event_api_base_url": (
            _normalize_base_url(event_api_base_url)
            if event_api_base_url
            else current_config["event_api_base_url"]
        ),
        "default_query": (
            _parse_json_object(default_query_json, "default_query_json")
            if default_query_json.strip()
            else current_config["default_query"]
        ),
        "default_headers": (
            _parse_json_object(default_headers_json, "default_headers_json")
            if default_headers_json.strip()
            else current_config["default_headers"]
        ),
        "timeout_seconds": (
            timeout_seconds
            if timeout_seconds is not None
            else current_config["timeout_seconds"]
        ),
        "verify_ssl": (
            verify_ssl if verify_ssl is not None else current_config["verify_ssl"]
        ),
    }
    return json.dumps({"status": "ok", "config": _state["config"]}, indent=2)


@mcp.tool()
def get_runtime_config() -> str:
    """Return the active runtime configuration."""
    return json.dumps(_state["config"], indent=2)


@mcp.tool()
def call_ariba_api(
    service: str,
    method: str,
    path: str,
    query_json: str = "{}",
    payload_json: str = "",
    headers_json: str = "{}",
) -> str:
    """
    Make a generic request to an SAP Ariba Sourcing API.

    Args:
        service: project_management or event_management.
        method: HTTP method such as GET, POST, PUT, PATCH, or DELETE.
        path: Relative API path such as /projects or /events/Doc123.
        query_json: JSON object of query parameters.
        payload_json: JSON object, JSON array, or raw string body.
        headers_json: JSON object of request headers.
    """
    return _request(
        service=service,
        method=method,
        path=path,
        query_json=query_json,
        payload_json=payload_json,
        headers_json=headers_json,
    )


@mcp.tool()
def list_projects(query_json: str = "{}") -> str:
    """Call GET /projects on the Sourcing Project Management API."""
    return _request(service="project_management", method="GET", path="/projects", query_json=query_json)


@mcp.tool()
def get_project(project_id: str, query_json: str = "{}") -> str:
    """Call GET /projects/{projectId}."""
    return _request(service="project_management", method="GET", path=f"/projects/{project_id}", query_json=query_json)


@mcp.tool()
def update_project(project_id: str, payload_json: str, query_json: str = "{}") -> str:
    """Call PUT /projects/{projectId}."""
    return _request(
        service="project_management",
        method="PUT",
        path=f"/projects/{project_id}",
        query_json=query_json,
        payload_json=payload_json,
    )


@mcp.tool()
def list_project_documents(project_id: str, query_json: str = "{}") -> str:
    """Call GET /projects/{projectId}/documents."""
    return _request(
        service="project_management",
        method="GET",
        path=f"/projects/{project_id}/documents",
        query_json=query_json,
    )


@mcp.tool()
def list_project_tasks(project_id: str, query_json: str = "{}") -> str:
    """Call GET /projects/{projectId}/tasks."""
    return _request(
        service="project_management",
        method="GET",
        path=f"/projects/{project_id}/tasks",
        query_json=query_json,
    )


@mcp.tool()
def get_project_task(project_id: str, task_id: str, query_json: str = "{}") -> str:
    """Call GET /projects/{projectId}/tasks/{taskId}."""
    return _request(
        service="project_management",
        method="GET",
        path=f"/projects/{project_id}/tasks/{task_id}",
        query_json=query_json,
    )


@mcp.tool()
def list_events(query_json: str = "{}") -> str:
    """Call GET /events on the Event Management API."""
    return _request(service="event_management", method="GET", path="/events", query_json=query_json)


@mcp.tool()
def get_event(event_id: str, query_json: str = "{}") -> str:
    """Call GET /events/{eventId}."""
    return _request(service="event_management", method="GET", path=f"/events/{event_id}", query_json=query_json)


@mcp.tool()
def create_event(payload_json: str, query_json: str = "{}") -> str:
    """Call POST /events."""
    return _request(
        service="event_management",
        method="POST",
        path="/events",
        query_json=query_json,
        payload_json=payload_json,
    )


@mcp.tool()
def update_event(event_id: str, payload_json: str, query_json: str = "{}") -> str:
    """Call PUT /events/{eventId}."""
    return _request(
        service="event_management",
        method="PUT",
        path=f"/events/{event_id}",
        query_json=query_json,
        payload_json=payload_json,
    )


@mcp.tool()
def get_event_items(event_id: str, query_json: str = "{}") -> str:
    """Call GET /events/{eventId}/items."""
    return _request(
        service="event_management",
        method="GET",
        path=f"/events/{event_id}/items",
        query_json=query_json,
    )


@mcp.tool()
def get_supplier_invitations(event_id: str, query_json: str = "{}") -> str:
    """Call GET /events/{eventId}/supplierInvitations."""
    return _request(
        service="event_management",
        method="GET",
        path=f"/events/{event_id}/supplierInvitations",
        query_json=query_json,
    )


@mcp.tool()
def get_bid_summary(event_id: str, query_json: str = "{}") -> str:
    """Call GET /events/{eventId}/bidSummary."""
    return _request(
        service="event_management",
        method="GET",
        path=f"/events/{event_id}/bidSummary",
        query_json=query_json,
    )


@mcp.tool()
def get_request_history() -> str:
    """Return recent request metadata captured by this server."""
    return json.dumps(_state["request_history"], indent=2)


@mcp.resource("config://runtime")
def runtime_config_resource() -> str:
    """Expose the active runtime config as an MCP resource."""
    return json.dumps(_state["config"], indent=2)


@mcp.resource("response://last")
def last_response_resource() -> str:
    """Expose the most recent API response as an MCP resource."""
    if _state["last_response"] is None:
        return json.dumps({"status": "empty", "message": "No request has been made yet."}, indent=2)
    return json.dumps(_state["last_response"], indent=2, default=str)
