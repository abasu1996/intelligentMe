# mcpserver

This project exposes the `invoice-po-tax-analyzer` FastMCP server.

## Local run

For stdio MCP usage:

```bash
mcp run mcpserver/mcp_server.py
```

For HTTP testing:

```bash
sh mcpserver/startup.sh
```

The MCP HTTP endpoint is served at `/mcp` and the health endpoint is `/health`.

## Azure App Service

Azure deployment artifacts are generated in the repository root:

- `azure.yaml`
- `infra/`
- `mcpserver/requirements.txt`
- `mcpserver/startup.sh`

The App Service startup command runs:

```bash
sh startup.sh
```
