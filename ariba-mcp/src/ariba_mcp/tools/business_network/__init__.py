"""Business Network APIs.

APIs in this folder:
  - Ariba Network Purchase Orders API (Anil)
  - Purchase Orders Supplier API (Nitish SM)
  - Order Change Requests API for Buyers (Anim)
  - Order Change Requests API for Suppliers (Shabreen)
  - Ariba Network Invoice Header Data Extraction API (Ayub)
  - Ship Notice API for Buyers (Ayub)
  - Ship Notice API for Suppliers (Shabreen)
  - Planning Collaboration Buyer API (Anim)
  - Planning Collaboration Supplier API (Shabreen)
  - Trading Partner Profile Certification API (Ayub)
  - Supplier Information API (Shabreen)
  - Proof of Service API for Buyers (unassigned)
  - Data Replication Status for Multi-ERP (Ayub)
  - Transaction Monitoring API (Anim)

Each person creates their own .py file in this folder.
"""

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register all Business Network tools from submodules."""
    from ariba_mcp.tools.business_network import _example, data_replication_status

    _example.register(mcp, client)
    data_replication_status.register(mcp, client)

    # invoice_header_data_extraction — file not yet created by Ayub
    # risk_category_information — file not yet created by Shabreen



