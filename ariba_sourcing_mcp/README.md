# Ariba Sourcing MCP

This project provides a small MCP server for SAP Ariba Sourcing APIs.

It is intentionally built without authorization logic for now, per your request. The server only manages:

- runtime endpoint configuration
- request construction
- named tools for common Sourcing Project Management and Event Management endpoints
- a generic passthrough tool for other Ariba Sourcing endpoints

## Reference

This scaffold is based on the SAP Help documentation for:

- Sourcing Project Management API v2.0
- Event Management API

Verified endpoint families used in this project include:

- `GET /projects`
- `GET /projects/{projectId}`
- `PUT /projects/{projectId}`
- `GET /projects/{projectId}/documents`
- `GET /projects/{projectId}/tasks`
- `GET /projects/{projectId}/tasks/{taskId}`
- `GET /events`
- `POST /events`
- `GET /events/{eventId}`
- `PUT /events/{eventId}`
- `GET /events/{eventId}/items`
- `GET /events/{eventId}/supplierInvitations`
- `GET /events/{eventId}/bidSummary`

## Run

From this directory:

```bash
uv run mcp run main.py
```

Or, if you already have the dependencies installed:

```bash
mcp run main.py
```

## Typical flow

1. Call `configure_ariba_runtime`
2. Call one of the named tools like `list_projects` or `get_event`
3. Use `call_ariba_api` for endpoints that are not wrapped yet

## Runtime configuration

You can configure separate base URLs for the two SAP Ariba API families:

- `project_api_base_url`
- `event_api_base_url`

Example values depend on your tenant and SAP runtime URL. This project does not assume a tenant-specific hostname.

You can also set:

- default query parameters as JSON
- default headers as JSON
- timeout
- SSL verification

## Notes

- Auth is deliberately omitted.
- If your tenant requires headers, cookies, or OAuth later, add that in `_request`.
- The server stores the last response in memory and exposes it as an MCP resource.
