# Contributing to ariba-mcp

8 team members building an MCP server for 48 SAP Ariba APIs.

**API Docs:** https://help.sap.com/docs/ariba-apis
**Developer Portal:** https://developer.ariba.com

---

## Getting Started

```bash
git clone https://github.com/nitishsm2002/ariba-mcp.git
cd ariba-mcp
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Get the `.env` file from the credentials owner and place it in the project root.

---

## How It Works

- **Shared infrastructure is done** — `config.py`, `auth.py`, `client.py`, `errors.py`, `server.py` are complete
- **6 domain folders** under `tools/` — each has an `_example.py` to copy from
- **You create a new .py file** in the right folder for your API
- **Register it** by adding an import in the folder's `__init__.py`

---

## Step-by-Step: Implement Your API

### 1. Find your folder

| Folder | APIs |
|--------|------|
| `tools/business_network/` | POs, invoices, ship notices, planning, certifications |
| `tools/catalog/` | Internal/public/network catalogs, content, connectivity |
| `tools/general/` | Approvals, audit, monitoring, config, forms, assets |
| `tools/procurement/` | Operational + analytical reporting, contracts |
| `tools/strategic_sourcing/` | Sourcing projects, events, approvals, master data |
| `tools/supplier_management/` | Supplier data, profiles, risk, invite |

### 2. Look at the `_example.py` in your folder

Each folder has a working example tool. Read it to understand the pattern.

### 3. Create your file

Create a new `.py` file named after your API:
```
tools/supplier_management/supplier_data_extraction.py
```

### 4. Write your register function

```python
import json
from fastmcp import FastMCP
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

# Get this from the Developer Portal Swagger spec
MY_API_PATH = "my-api/v1/prod"

def register(mcp: FastMCP, client: AribaClient) -> None:

    @mcp.tool(
        name="ariba_<meaningful_name>",
        description="Clear description of what this does.",
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def my_tool(param: str, page_token: str | None = None) -> str:
        try:
            result = await client.fetch_resource(MY_API_PATH, f"resource/{param}")
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
```

### 5. Register in `__init__.py`

Open your folder's `__init__.py` and add:
```python
from ariba_mcp.tools.supplier_management import supplier_data_extraction
supplier_data_extraction.register(mcp, client)
```

### 6. Test

```bash
python -m ariba_mcp.server
npx @modelcontextprotocol/inspector python -m ariba_mcp.server
```

---

## Client Methods Reference

The `AribaClient` (in `client.py`) provides these methods:

```python
# Reporting view queries (Operational/Analytical Reporting)
await client.fetch_view(api_path, view_name, filters_dict, page_token)
await client.fetch_view_count(api_path, view_name, filters_dict)

# Async jobs (large datasets)
await client.submit_job(job_api_path, view_name, filters_dict)
await client.get_job_status(job_api_path, job_id)
await client.get_job_results(result_api_path, job_id, page_token)

# REST resource APIs (Supplier Data, Contracts, etc.)
await client.fetch_resource(api_path, "resource/path", extra_params)

# Generic (if nothing else fits)
await client.get(full_url, params)
await client.post(full_url, json_body, params)
```

`client.realm` and `client.base_url` are available as properties.

---

## Branch Naming

```
feature/<api-name>         # e.g. feature/supplier-data-api
fix/<api-name>-<desc>      # e.g. fix/risk-exposure-pagination
chore/<desc>               # e.g. chore/update-deps
```

---

## PR Process

1. Branch from `main`
2. Implement your tool(s)
3. Run: `ruff check src/ && ruff format src/`
4. Push and open a PR
5. 1 reviewer approves → squash merge

---

## Rules

- **One file per API** — each person creates their own `.py` file in the right folder
- **Always register** — add import + `.register()` call in the folder's `__init__.py`
- **Async everywhere** — all tool functions are `async`
- **Always catch exceptions** — use `handle_ariba_error(e)`
- **Return JSON strings** — `json.dumps(result, default=str)`
- **Support pagination** — accept `page_token` where applicable
- **No secrets in code** — credentials from `.env` only
- **Tool names** — prefix with `ariba_`, use `snake_case`
- **Don't modify shared infra** without coordinating with the team
