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
    from ariba_mcp.tools.procurement import (
        audit_search,
        contract_compliance,
        contract_terms_management,
        document_approval,
        integration_monitoring_for_procurement,
        operational_reporting_for_procurement,
        procurement_workspace,
    )

    document_approval.register(mcp, client)
    audit_search.register(mcp, client)
    contract_compliance.register(mcp, client)
    contract_terms_management.register(mcp, client)
    integration_monitoring_for_procurement.register(mcp, client)
    procurement_workspace.register(mcp, client)
    operational_reporting_for_procurement.register(mcp, client)
