# SAP Ariba MCP Server

A **Model Context Protocol (MCP) server** that exposes **48 SAP Ariba APIs** as Claude-compatible tools. Built with Python and [FastMCP](https://github.com/jlowin/fastmcp).

This enables Claude (or any MCP-compatible AI) to interact with SAP Ariba procurement, sourcing, supplier management, contracts, catalogs, and supply chain data via natural language.

**API Reference:** https://help.sap.com/docs/ariba-apis
**Developer Portal:** https://developer.ariba.com

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Consumer Layer                                  │
│  Claude Desktop / Claude Code / Custom App       │
└────────────────────┬────────────────────────────┘
                     │ MCP (stdio)
┌────────────────────▼────────────────────────────┐
│  ariba-mcp Server (this project)                 │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ 6 Domain │  │ Auth     │  │ Client   │       │
│  │ Folders  │──│ OAuth2.0 │──│ httpx    │       │
│  │ 48 APIs  │  │ cached   │  │ async    │       │
│  └──────────┘  └──────────┘  └────┬─────┘       │
└────────────────────────────────────┼─────────────┘
                                     │ HTTPS
┌────────────────────────────────────▼─────────────┐
│  SAP Ariba APIs                                   │
│  OAuth: api.ariba.com                             │
│  Data:  openapi.ariba.com/api                     │
└──────────────────────────────────────────────────┘
```

---

## All 48 APIs — 6 Domain Folders

### 1. Business Network — [tools/business_network/](src/ariba_mcp/tools/business_network/)

| API | Owner |
|-----|-------|
| Ariba Network Purchase Orders API | Anil |
| Purchase Orders Supplier API | Nitish SM |
| Order Change Requests API for Buyers | Anim |
| Order Change Requests API for Suppliers | Shabreen |
| Ariba Network Invoice Header Data Extraction API | Ayub |
| Ship Notice API for Buyers | Ayub |
| Ship Notice API for Suppliers | Shabreen |
| Planning Collaboration Buyer API | Anim |
| Planning Collaboration Supplier API | Shabreen |
| Trading Partner Profile Certification API | Ayub |
| Supplier Information API | Shabreen |
| Proof of Service API for Buyers | (unassigned) |
| Data Replication Status for Multi-ERP | Ayub |
| Transaction Monitoring API | Anim |

### 2. Catalog — [tools/catalog/](src/ariba_mcp/tools/catalog/)

| API | Owner |
|-----|-------|
| Internal Catalogs Shop API | Anil |
| Public Catalogs Shop API | Anil |
| Network Catalog Management API | Anil |
| SAP Ariba Catalog Content API | Ayub |
| Catalog Connectivity Service API | Ayub |
| Content Lookup API | Anil |
| Materials and BOM Tag Management API | Anil |

### 3. General — [tools/general/](src/ariba_mcp/tools/general/)

| API | Owner |
|-----|-------|
| Document Approval API | Anim |
| Audit Search API | Vanshika |
| Integration Monitoring API for Procurement | Vanshika |
| Integration Monitoring API for Strategic Sourcing | Pranathi |
| Master Data Integration Job Status API | Anim |
| Configuration Parameter Review API | Vanshika |
| SAP Ariba Custom Forms API | Vanshika |
| Asset Management API | Rohit Naik |
| Master Data Retrieval API for Procurement | Rohit Naik |
| Guided Buying Functional Documents API | Anim |
| Create Procurement Workspace API | Vanshika |
| User Qualification API | Anil |
| Public Procurement Notices Export API | Rohit Naik |
| NDA Data Export API | Rohit Naik |

### 4. Procurement — [tools/procurement/](src/ariba_mcp/tools/procurement/)

| API | Owner |
|-----|-------|
| Operational Reporting API for Procurement | Vanshika |
| Analytical Reporting API (Strategic & Operational) | Anim |
| Contract Compliance API | Vanshika |
| Contract Workspace Retrieval API | Anim |
| Contract Workspace Management APIs | Rohit Naik |
| Contract Terms Management API | Shabreen |
| Cost Breakdown Data Extraction API | Vanshika |

### 5. Strategic Sourcing — [tools/strategic_sourcing/](src/ariba_mcp/tools/strategic_sourcing/)

| API | Owner |
|-----|-------|
| Operational Reporting API for Strategic Sourcing | Pranathi |
| Sourcing Project Management API | Pranathi |
| Event Management API | Anim |
| External Approval API for Sourcing & Supplier Mgmt | Pranathi |
| Master Data Retrieval API for Sourcing | Pranathi |
| Pricing API for Product Sourcing | Pranathi |
| Surrogate Bid API | Rohit Naik |
| Product Hierarchy Management API | Shabreen |
| Bill of Materials Import API | Anim |

### 6. Supplier Management — [tools/supplier_management/](src/ariba_mcp/tools/supplier_management/)

| API | Owner |
|-----|-------|
| Supplier Data API with Pagination | Nitish SM |
| Supplier Data API | Nitish SM |
| Supplier Data Extraction API | Nitish SM |
| Ariba Network Supplier Profile API | Nitish SM |
| Supplier Invite API | Nitish SM |
| Supplier Risk Engagements API | Nitish SM |
| Risk Exposure API | Anim |
| Risk Category Information API | Shabreen |

---

## Prerequisites

- **Python 3.11+**
- **uv** (recommended) or pip
- **SAP Ariba API credentials** — one person provisions them for the team

---

## Credential Setup (One-time)

One team member registers on the SAP Ariba Developer Portal and shares credentials:

1. Go to https://developer.ariba.com
2. Register a new application
3. Enable the APIs needed
4. Note: **OAuth Client ID**, **Client Secret**, **API Key**, **Realm** name
5. Create `.env` and share securely:

```bash
cp .env.example .env
# Fill in credentials
```

> **Never commit `.env` to git.**

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/nitishsm2002/ariba-mcp.git
cd ariba-mcp
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Add .env file (get from credentials owner)

# Run the server
python -m ariba_mcp.server

# Test with MCP Inspector
npx @modelcontextprotocol/inspector python -m ariba_mcp.server
```

### Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ariba": {
      "command": "python",
      "args": ["-m", "ariba_mcp.server"],
      "cwd": "/path/to/ariba-mcp"
    }
  }
}
```

---

## Project Structure

```
ariba-mcp/
├── src/ariba_mcp/
│   ├── server.py                       # FastMCP entrypoint
│   ├── config.py                       # Settings (reads .env)
│   ├── auth.py                         # OAuth 2.0 client credentials
│   ├── client.py                       # Async HTTP client (views, jobs, resources)
│   ├── errors.py                       # Error handling
│   ├── models/common.py               # Shared Pydantic models
│   └── tools/
│       ├── __init__.py                 # Registers all 6 domain folders
│       ├── business_network/           # 14 APIs — POs, invoices, ship notices, planning
│       │   ├── __init__.py
│       │   ├── _example.py
│       │   └── <your_api>.py           # ← team members add files here
│       ├── catalog/                    # 7 APIs — catalogs, content, connectivity
│       │   ├── __init__.py
│       │   ├── _example.py
│       │   └── <your_api>.py
│       ├── general/                    # 14 APIs — approvals, audit, monitoring, config
│       │   ├── __init__.py
│       │   ├── _example.py
│       │   └── <your_api>.py
│       ├── procurement/                # 7 APIs — reporting, contracts, compliance
│       │   ├── __init__.py
│       │   ├── _example.py
│       │   └── <your_api>.py
│       ├── strategic_sourcing/         # 9 APIs — sourcing projects, events, bids
│       │   ├── __init__.py
│       │   ├── _example.py
│       │   └── <your_api>.py
│       └── supplier_management/        # 8 APIs — supplier data, risk, profiles
│           ├── __init__.py
│           ├── _example.py
│           └── <your_api>.py
├── tests/
├── .agents/skills/mcp-builder/        # MCP builder reference docs
├── pyproject.toml
├── .env.example
├── CONTRIBUTING.md
└── .gitignore
```

---

## How to Implement Your API

Each folder has a `_example.py` with a working tool to copy from. To add your API:

1. Find the right folder in `src/ariba_mcp/tools/` (see table above)
2. Look at `_example.py` in that folder for the pattern
3. Create a **new .py file** named after your API (e.g. `supplier_data_extraction.py`)
4. Define a `register(mcp, client)` function with your `@mcp.tool` inside it
5. Open the folder's `__init__.py` and add your import + `register()` call
6. Look up the exact endpoint path on the [Developer Portal](https://developer.ariba.com)
7. Test with MCP Inspector: `npx @modelcontextprotocol/inspector python -m ariba_mcp.server`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed coding patterns.

---

## Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ARIBA_REALM` | Yes | — | Ariba realm (e.g. `mycompany-T`) |
| `ARIBA_CLIENT_ID` | Yes | — | OAuth 2.0 client ID |
| `ARIBA_CLIENT_SECRET` | Yes | — | OAuth 2.0 client secret |
| `ARIBA_API_KEY` | Yes | — | Application key |
| `ARIBA_OAUTH_URL` | No | `https://api.ariba.com` | OAuth endpoint |
| `ARIBA_API_URL` | No | `https://openapi.ariba.com/api` | API base URL |

---

## License

MIT
