"""Tool registration for all SAP Ariba API domains.

6 domain folders covering 48 SAP Ariba APIs:
  - business_network/   — PO, invoices, ship notices, planning, certifications
  - catalog/            — Internal/public/network catalogs, content, connectivity
  - general/            — Approvals, audit, monitoring, config, forms, assets
  - procurement/        — Operational + analytical reporting, contracts
  - strategic_sourcing/ — Sourcing projects, events, approvals, master data
  - supplier_management/ — Supplier data, profiles, risk, invite

https://help.sap.com/docs/ariba-apis
"""

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.tools import (
    business_network,
    catalog,
    general,
    procurement,
    strategic_sourcing,
    supplier_management,
)

_DOMAINS = [
    business_network,
    catalog,
    general,
    procurement,
    strategic_sourcing,
    supplier_management,
]


def register_all_tools(mcp: FastMCP, client: AribaClient) -> None:
    """Register tools from every domain folder."""
    for domain in _DOMAINS:
        domain.register(mcp, client)
