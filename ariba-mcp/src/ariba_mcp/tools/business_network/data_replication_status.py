"""Data Replication Status for Multi-ERP Configurations API.

Owner: AYUB SHAIK
Prod URL: https://openapi.ariba.com/api/replication/v1/prod
Docs Index: https://help.sap.com/doc/6350dec774d9447b8ce1823a91ac698d/cloud/en-US/index.html?forSiteMap=true
Docs Topic: https://help.sap.com/doc/6350dec774d9447b8ce1823a91ac698d/cloud/en-US/f2ae044c66e84ffd80584f2013954155.html?forSiteMap=true

Key endpoint:
  GET /replicationStatus — Returns the status of data replication across multi-ERP configurations
  Accepts query parameters to filter by realm, object type, status, target site, etc.

Authentication: OAuth 2.0 Bearer token + apiKey header
Response format: JSON
"""

import json

import httpx
from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

API_PATH = "replication/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:
    """Register Data Replication Status for Multi-ERP Configurations API tools."""

    @mcp.tool(
        name="ariba_replication_list_all",
        description=(
            "List all data replication status records for a given realm. "
            "Returns object type, replication status, source site (parent), "
            "target site (child), timestamp, and error code if applicable."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_all_replication_records(realm: str) -> str:
        try:
            url = f"{client.base_url}/{API_PATH}/statuses"
            headers = await client.auth.get_headers()    

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params={"realm": realm},
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            data = resp.json()
            records = data if isinstance(data, list) else data.get("replicationRecords", [])
            result = {
                "total_count": len(records),
                "replication_records": records,
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_replication_get_by_object_type",
        description=(
            "Get replication status for specific object types. "
            "Pass one or more object types (comma-separated, e.g. 'SharedUser,Catalog'). "
            "Returns replication records for each matched type."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_by_object_type(realm: str, object_types: str) -> str:
        try:
            types = [t.strip() for t in object_types.split(",")]
            url = f"{client.base_url}/{API_PATH}/statuses"
            headers = await client.auth.get_headers()

            all_records = []
            for object_type in types:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(
                        url,
                        params={"realm": realm, "objectType": object_type},
                        headers=headers,
                        timeout=60,
                    )
                    resp.raise_for_status()

                data = resp.json()
                records = data if isinstance(data, list) else data.get("replicationRecords", [])
                all_records.extend(records)

            result = {
                "total_count": len(all_records),
                "replication_records": all_records,
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_replication_filter_by_status",
        description=(
            "Filter replication records by status. "
            "Statuses: Completed, Failed, InProgress, Pending, PartiallyCompleted. "
            "Returns all records for the realm that match the given status."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def filter_by_status(
        realm: str,
        status: str | None = None,
    ) -> str:
        try:
            url = f"{client.base_url}/{API_PATH}/statuses"
            headers = await client.auth.get_headers()

            params: dict = {"realm": realm}
            if status:
                params["status"] = status

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            data = resp.json()
            records = data if isinstance(data, list) else data.get("replicationRecords", [])

            # Client-side guard in case the API does not filter server-side
            if status:
                records = [r for r in records if r.get("status") == status]

            result = {
                "filters": {"status": status},
                "total_count": len(records),
                "replication_records": records,
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_replication_filter_by_site",
        description=(
            "Filter replication records by child site (target site). "
            "Useful for monitoring replication health for a specific ERP child instance "
            "(e.g. AcmeDSAPP-1, AcmeDSAPP-2)."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def filter_by_site(
        realm: str,
        target_site: str | None = None,
    ) -> str:
        try:
            url = f"{client.base_url}/{API_PATH}/statuses"
            headers = await client.auth.get_headers()

            params: dict = {"realm": realm}
            if target_site:
                params["targetSite"] = target_site

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            data = resp.json()
            records = data if isinstance(data, list) else data.get("replicationRecords", [])

            # Client-side guard in case the API does not filter server-side
            if target_site:
                records = [r for r in records if r.get("targetSite") == target_site]

            result = {
                "filters": {"target_site": target_site},
                "total_count": len(records),
                "replication_records": records,
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    @mcp.tool(
        name="ariba_replication_summary",
        description=(
            "Get a summary of all replication records for the realm: "
            "total count, breakdown by status, breakdown by object type, "
            "and count of records with errors."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def replication_summary(realm: str) -> str:
        try:
            url = f"{client.base_url}/{API_PATH}/statuses"
            headers = await client.auth.get_headers()

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    params={"realm": realm},
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()

            data = resp.json()
            records = data if isinstance(data, list) else data.get("replicationRecords", [])

            # Deduplicate by objectType + targetSite composite key
            unique: dict = {}
            for r in records:
                key = f"{r.get('objectType', '')}::{r.get('targetSite', '')}"
                if key and key not in unique:
                    unique[key] = r

            by_status: dict[str, int] = {}
            by_object_type: dict[str, int] = {}
            records_with_errors = 0

            for r in unique.values():
                status = r.get("status", "Unknown")
                obj_type = r.get("objectType", "Unknown")
                error_code = r.get("errorCode")

                by_status[status] = by_status.get(status, 0) + 1
                by_object_type[obj_type] = by_object_type.get(obj_type, 0) + 1
                if error_code:
                    records_with_errors += 1

            result = {
                "total_unique_records": len(unique),
                "total_rows": len(records),
                "by_status": by_status,
                "by_object_type": by_object_type,
                "records_with_errors": records_with_errors,
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
