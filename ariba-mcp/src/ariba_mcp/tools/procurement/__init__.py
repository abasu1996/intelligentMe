"""Procurement APIs.

APIs in this folder:
  - Operational Reporting API for Procurement (Vanshika)
  - Analytical Reporting API for Strategic & Operational Procurement (Anim)
  - Contract Compliance API (Vanshika)
  - Contract Workspace Retrieval API (Anim)
  - Contract Workspace Management APIs (Rohit Naik)
  - Contract Terms Management API (Shabreen)
  - Cost Breakdown Data Extraction API (Vanshika)

Each person creates their own .py file in this folder.
"""

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient

def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register all Procurement tools from submodules."""
    from ariba_mcp.tools.procurement import _example
    from ariba_mcp.tools.procurement import document_approval
    from ariba_mcp.tools.procurement import audit_search
    from ariba_mcp.tools.procurement import contract_compliance
    from ariba_mcp.tools.procurement import integration_monitoring_for_procurement
    from ariba_mcp.tools.procurement import procurement_workspace
    from ariba_mcp.tools.procurement import operational_reporting_for_procurement

    _example.register(mcp, client)
    document_approval.register(mcp, client) 
    audit_search.register(mcp, client)
    contract_compliance.register(mcp, client)
    integration_monitoring_for_procurement.register(mcp, client)
    procurement_workspace.register(mcp, client)
    operational_reporting_for_procurement.register(mcp, client)

    # As team members add files, import and register them here:
    # from ariba_mcp.tools.procurement import analytical_reporting
    # analytical_reporting.register(mcp, client)
