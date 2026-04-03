"""Catalog APIs.

APIs in this folder:
  - Internal Catalogs Shop API (Anil)
  - Public Catalogs Shop API (Anil)
  - Network Catalog Management API (Anil)
  - SAP Ariba Catalog Content API (Ayub)
  - Catalog Connectivity Service API (Ayub)
  - Content Lookup API (Anil)
  - Materials and BOM Tag Management API (Anil)

  
Each person creates their own .py file in this folder.
"""

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register all Catalog tools from submodules."""
    from ariba_mcp.tools.catalog import _example

    _example.register(mcp, client)
