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

For an HTTP MCP endpoint locally:

```bash
python3 app.py
```

The server listens on `http://127.0.0.1:8000/mcp` by default. Override the path with `MCP_HTTP_PATH` and the port with `PORT`.

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
- project-specific default query parameters as JSON
- event-specific default query parameters as JSON
- default headers as JSON
- timeout
- SSL verification

For Sourcing Project Management endpoints, some tenants require these query parameters on every call:

- `realm`
- `user`
- `passwordAdapter`

This server now supports them through environment variables or runtime config:

- `ARIBA_PROJECT_REALM` or fallback `ARIBA_REALM`
- `ARIBA_PROJECT_USER` or fallback `ARIBA_USER`
- `ARIBA_PROJECT_PASSWORD_ADAPTER` or fallback `ARIBA_PASSWORD_ADAPTER`

You can also set them at runtime with `project_default_query_json` in `configure_ariba_runtime`.

## Notes

- Auth is deliberately omitted.
- If your tenant requires headers, cookies, or OAuth later, add that in `_request`.
- The server stores the last response in memory and exposes it as an MCP resource.

## SAP BTP Cloud Foundry deployment

This folder is now prepared for direct deployment to SAP BTP Cloud Foundry with:

- `manifest.yml` for Cloud Foundry app settings
- `Procfile` to start the HTTP server
- `runtime.txt` to request Python 3.12
- `requirements.txt` for buildpack dependency installation
- `.cfignore` to keep the local `.env` file out of the upload bundle

### Deploy

From this directory:

```bash
cf login -a <api-endpoint> -o <org> -s <space>
cf push
```

The manifest uses `random-route: true`, so Cloud Foundry will assign a route automatically on first push. The MCP endpoint will be available at:

```text
https://<generated-route>/mcp
```

### Set configuration and secrets

The app now reads real environment variables first and falls back to the local `.env` file only for local development. For SAP BTP, set values with `cf set-env` instead of shipping `.env`:

```bash
cf set-env ariba-sourcing-mcp ARIBA_PROJECT_API_BASE_URL https://<project-api-host>
cf set-env ariba-sourcing-mcp ARIBA_EVENT_API_BASE_URL https://<event-api-host>
cf set-env ariba-sourcing-mcp ARIBA_REALM <realm>
cf set-env ariba-sourcing-mcp ARIBA_API_KEY <api-key>
cf set-env ariba-sourcing-mcp ARIBA_OAUTH_URL https://<oauth-host>
cf set-env ariba-sourcing-mcp ARIBA_CLIENT_ID <client-id>
cf set-env ariba-sourcing-mcp ARIBA_CLIENT_SECRET <client-secret>
cf restage ariba-sourcing-mcp
```

Optional runtime variables:

- `ARIBA_DEFAULT_QUERY_JSON`
- `ARIBA_PROJECT_USER`
- `ARIBA_PROJECT_PASSWORD_ADAPTER`
- `ARIBA_PROJECT_REALM`
- `ARIBA_USER`
- `ARIBA_PASSWORD_ADAPTER`
- `ARIBA_DEFAULT_HEADERS_JSON`
- `ARIBA_TIMEOUT_SECONDS`
- `ARIBA_VERIFY_SSL`
- `MCP_HTTP_PATH`

### Scaling note

Runtime configuration, request history, and the last response are stored only in process memory. Keep the deployment at one instance unless you externalize that state.
