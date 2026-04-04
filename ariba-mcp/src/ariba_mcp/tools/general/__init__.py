"""General / Administration APIs.

APIs in this folder:
  - Document Approval API (Anim)
  - Audit Search API (Vanshika)
  - Integration Monitoring API for Procurement (Vanshika)
  - Integration Monitoring API for Strategic Sourcing (Pranathi)
  - Master Data Integration Job Status API (Anim)
  - Configuration Parameter Review API (Vanshika)
  - SAP Ariba Custom Forms API (Vanshika)
  - Asset Management API (Rohit Naik)
  - Master Data Retrieval API for Procurement (Rohit Naik)
  - Guided Buying Functional Documents API (Anim)
  - Create Procurement Workspace API (Vanshika)
  - User Qualification API (Anil)
  - Public Procurement Notices Export API (Rohit Naik)
  - NDA Data Export API (Rohit Naik)

Each person creates their own .py file in this folder.
"""

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient

def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register all General/Admin tools from submodules."""
    from ariba_mcp.tools.general import Integration_monitoring_API, asset_management, user_qualification

    Integration_monitoring_API.register(mcp, client)
    asset_management.register(mcp, client)
    user_qualification.register(mcp, client)
