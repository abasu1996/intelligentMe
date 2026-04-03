"""Operational Reporting API for Procurement.
 
Owner: <assign owner>
Docs: https://help.sap.com/doc/42cb9e6fb65a4fa7b03f5e0ec7d406f9/cloud/en-US/index.html
 
This API allows customer developers to extract and report on transactional
procurement data from SAP Ariba to support operational decisions.
 
This module covers all three sub-APIs that together form the Operational
Reporting API for Procurement, plus the Synchronous (details) API:
 
  ┌─────────────────────────────────────────────────────────────────┐
  │  WORKFLOW: View Management → Job Submission → Job Result        │
  │                                                                 │
  │  1. Create/manage view templates (View Management API)          │
  │  2. Submit async reporting job   (Job Submission API)           │
  │  3. Poll job status + download   (Job Result API)               │
  │                                                                 │
  │  For small datasets, use Synchronous API (details) directly.    │
  └─────────────────────────────────────────────────────────────────┘
 
Sub-APIs and their confirmed production base URLs:
  View Management API : procurement-reporting-view/v2/prod
  Job Submission API  : procurement-reporting-job/v2/prod
  Job Result API      : procurement-reporting-jobresult/v2/prod
  Synchronous API     : procurement-reporting-details/v2/prod
 
Endpoints implemented:
 
  View Management (procurement-reporting-view/v2/prod):
    GET  /viewTemplates                          – list all view templates
    GET  /viewTemplates/{viewTemplateName}       – get a specific view template
    POST /viewTemplates/{viewTemplateName}       – create a new view template
    POST /viewTemplates/{viewTemplateName}/patch – update an existing view template
    GET  /metadata                               – get available fields for a document type
 
  Job Submission (procurement-reporting-job/v2/prod):
    POST /jobs                                   – submit an async reporting job
 
  Job Result (procurement-reporting-jobresult/v2/prod):
    GET  /jobs/{jobId}                           – check job status
    GET  /jobs/{jobId}/files/{fileId}            – download job result file
 
  Synchronous (procurement-reporting-details/v2/prod):
    GET  /views/{viewTemplateName}               – synchronous data retrieval
 
Rate Limits (all sub-APIs):
  Synchronous endpoint : 1/sec, 3/min, 50/hr, 300/day
  Async job POST       : 1/sec, 2/min, 8/hr, 40/day
  Async status/download: 2/sec, 20/min, 200/hr, 1000/day
  View template mgmt   : 1/sec, 10/min, 100/hr, 500/day
 
Prerequisites:
  - SAP Ariba Developer Portal access (Procurement tab)
  - OAuth authentication configured (handled by AribaClient)
  - View templates must be created before submitting jobs or querying views
  - For async jobs: poll status until 'completed', 'completedZeroRecords', or 'expired'
"""
 
import json
 
from fastmcp import FastMCP
 
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error
 
# Confirmed production base paths for each sub-API
VIEW_MANAGEMENT_API  = "procurement-reporting-view/v2/prod"
JOB_SUBMISSION_API   = "procurement-reporting-job/v2/prod"
JOB_RESULT_API       = "procurement-reporting-jobresult/v2/prod"
SYNCHRONOUS_API      = "procurement-reporting-details/v2/prod"
 
 
def register(mcp: FastMCP, client: AribaClient) -> None:
 
    # ── VIEW MANAGEMENT ───────────────────────────────────────────────────────
 
    @mcp.tool(
        name="ariba_procurement_view_templates_list",
        description=(
            "List all reporting view templates available for the realm in the "
            "Operational Reporting API for Procurement. Returns both system "
            "(built-in) and custom view templates. Use this to discover valid "
            "viewTemplateName values before submitting jobs or querying data."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def procurement_view_templates_list(
        realm: str,
        product: str = "buyer",
        include_inactive: bool = False,
    ) -> str:
        """
        Args:
            realm:            Required. The target SAP Ariba realm.
            product:          Optional. Product context. Use 'buyer' for
                              Buying and Invoicing (default).
            include_inactive: Optional. If True, includes deactivated templates.
        """
        try:
            params: dict = {
                "realm": realm,
                "product": product,
                "includeInactive": str(include_inactive).lower(),
            }
            result = await client.fetch_resource(
                VIEW_MANAGEMENT_API, "viewTemplates", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    @mcp.tool(
        name="ariba_procurement_view_template_get",
        description=(
            "Retrieve the definition of a specific reporting view template by name "
            "from the Operational Reporting API for Procurement. Returns the "
            "document type, select attributes, and filter criteria for the template."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def procurement_view_template_get(
        realm: str,
        view_template_name: str,
        product: str = "buyer",
    ) -> str:
        """
        Args:
            realm:              Required. The target SAP Ariba realm.
            view_template_name: Required. The name of the view template to retrieve,
                                e.g. 'InvoiceSystemView', 'RequisitionSystemView',
                                'PurchaseOrderSystemView', 'Invoice_CELONIS_VIEW_V2'.
            product:            Optional. Product context (default: 'buyer').
        """
        try:
            params: dict = {"realm": realm, "product": product}
            result = await client.fetch_resource(
                VIEW_MANAGEMENT_API,
                f"viewTemplates/{view_template_name}",
                params,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    @mcp.tool(
        name="ariba_procurement_view_template_create",
        description=(
            "Create a new custom reporting view template in the Operational Reporting "
            "API for Procurement. A view template defines the document type, the fields "
            "to select, and the filter criteria for data extraction. "
            "Templates must be 'published' before use in jobs or synchronous queries. "
            "Use ariba_procurement_metadata_get first to discover available fields."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def procurement_view_template_create(
        realm: str,
        view_template_name: str,
        body: dict,
    ) -> str:
        """
        Args:
            realm:              Required. The target SAP Ariba realm.
            view_template_name: Required. Name for the new view template.
                                Convention: use descriptive names ending in '_VIEW',
                                e.g. 'Invoice_MYAPP_VIEW'.
            body:               Required. JSON payload defining the view template:
                                {
                                  "documentType": "Invoice",
                                  "selectAttributes": ["InvoiceNumber", "Status",
                                                       "TotalAmount", "CreatedDate"],
                                  "filterExpressions": [
                                    {
                                      "name": "createdDateFrom",
                                      "value": "2024-01-01T00:00:00Z"
                                    }
                                  ]
                                }
        """
        try:
            result = await client.post_resource(
                VIEW_MANAGEMENT_API,
                f"viewTemplates/{view_template_name}",
                {"realm": realm},
                body,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    @mcp.tool(
        name="ariba_procurement_view_template_update",
        description=(
            "Update an existing custom reporting view template in the Operational "
            "Reporting API for Procurement. Use this to modify fields, filters, or "
            "status (publish/deactivate) of an existing template. "
            "Valid status values: 'published', 'deactivated'."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def procurement_view_template_update(
        realm: str,
        view_template_name: str,
        body: dict,
    ) -> str:
        """
        Args:
            realm:              Required. The target SAP Ariba realm.
            view_template_name: Required. Name of the existing view template to update.
            body:               Required. JSON payload with fields to update. Can include:
                                {
                                  "status": "published",       ← or "deactivated"
                                  "selectAttributes": [...],   ← updated field list
                                  "filterExpressions": [...]   ← updated filters
                                }
        """
        try:
            result = await client.post_resource(
                VIEW_MANAGEMENT_API,
                f"viewTemplates/{view_template_name}/patch",
                {"realm": realm},
                body,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    @mcp.tool(
        name="ariba_procurement_metadata_get",
        description=(
            "Retrieve metadata for a specific document type in the Operational Reporting "
            "API for Procurement. Returns all available select fields and filter fields "
            "that can be used when creating a custom view template. "
            "Always call this before creating a new view template for a document type."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def procurement_metadata_get(
        realm: str,
        document_type: str,
        product: str = "buyer",
        include_custom_fields: bool = False,
    ) -> str:
        """
        Args:
            realm:                 Required. The target SAP Ariba realm.
            document_type:         Required. The document type to get metadata for.
                                   Common values:
                                     'Invoice'          – invoices and credit memos
                                     'Requisition'      – purchase requisitions
                                     'PurchaseOrder'    – purchase orders
                                     'CopyOrder'        – copied purchase orders
                                     'Receipt'          – goods receipts
                                     'ContractOrder'    – contract-based orders
                                     'SettlementRequest'– settlement / ERS documents
            product:               Optional. Product context (default: 'buyer').
            include_custom_fields: Optional. If True, includes custom field metadata.
                                   Useful for realms with custom extensions.
        """
        try:
            params: dict = {
                "realm": realm,
                "documentType": document_type,
                "product": product,
                "includeCustomFields": str(include_custom_fields).lower(),
            }
            result = await client.fetch_resource(
                VIEW_MANAGEMENT_API, "metadata", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    # ── JOB SUBMISSION ────────────────────────────────────────────────────────
 
    @mcp.tool(
        name="ariba_procurement_reporting_job_submit",
        description=(
            "Submit an asynchronous reporting job to the Operational Reporting API "
            "for Procurement. The job extracts data according to the specified view "
            "template and date filters. After submission, poll "
            "ariba_procurement_reporting_job_status until status is 'completed', "
            "'completedZeroRecords', or 'expired', then download using "
            "ariba_procurement_reporting_job_file_download. "
            "Rate limit: max 8 jobs/hour, 40 jobs/day."
        ),
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def procurement_reporting_job_submit(
        realm: str,
        view_template_name: str,
        created_date_from: str,
        created_date_to: str,
        include_inactive: bool = False,
        page_token: str | None = None,
    ) -> str:
        """
        Args:
            realm:              Required. The target SAP Ariba realm.
            view_template_name: Required. Name of a published view template to use.
                                e.g. 'InvoiceSystemView', 'Invoice_MYAPP_VIEW'.
                                Use ariba_procurement_view_templates_list to find valid names.
            created_date_from:  Required. Start of the date range (ISO 8601 UTC).
                                e.g. '2024-01-01T00:00:00Z'. Max range: varies by document
                                type (typically 1-3 months per job for large datasets).
            created_date_to:    Required. End of the date range (ISO 8601 UTC).
                                e.g. '2024-03-31T23:59:59Z'.
            include_inactive:   Optional. If True, includes inactive/deleted documents.
            page_token:         Optional. Token to retrieve the next page of a large
                                result set from a previously completed job.
 
        Returns:
            JSON containing jobId — use this to poll status and download results.
        """
        try:
            params: dict = {
                "realm": realm,
                "includeInactive": str(include_inactive).lower(),
            }
            if page_token:
                params["pageToken"] = page_token
            body = {
                "viewTemplateName": view_template_name,
                "filters": {
                    "createdDateFrom": created_date_from,
                    "createdDateTo": created_date_to,
                },
            }
            result = await client.post_resource(
                JOB_SUBMISSION_API, "jobs", params, body
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    # ── JOB RESULT ────────────────────────────────────────────────────────────
 
    @mcp.tool(
        name="ariba_procurement_reporting_job_status",
        description=(
            "Check the status of a previously submitted async reporting job in the "
            "Operational Reporting API for Procurement. Poll this every 5 minutes "
            "after submitting a job via ariba_procurement_reporting_job_submit. "
            "Job statuses: 'pending' → 'processing' → 'completed' / "
            "'completedZeroRecords' / 'expired'. "
            "When 'completed', response includes file IDs to download."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def procurement_reporting_job_status(
        realm: str,
        job_id: str,
    ) -> str:
        """
        Args:
            realm:  Required. The target SAP Ariba realm.
            job_id: Required. The job ID returned by ariba_procurement_reporting_job_submit.
                    e.g. '88fe53b0-848e-48d8-99d4-31758a5e42b91574699039445'.
        """
        try:
            params: dict = {"realm": realm}
            result = await client.fetch_resource(
                JOB_RESULT_API, f"jobs/{job_id}", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    @mcp.tool(
        name="ariba_procurement_reporting_job_file_download",
        description=(
            "Download a result file from a completed async reporting job in the "
            "Operational Reporting API for Procurement. A completed job may contain "
            "multiple ZIP files — call this for each fileId returned in the job status "
            "response. Each ZIP contains the extracted data in JSON format. "
            "Only call this when job status is 'completed'."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def procurement_reporting_job_file_download(
        realm: str,
        job_id: str,
        file_id: str,
        include_inactive: bool = False,
    ) -> str:
        """
        Args:
            realm:            Required. The target SAP Ariba realm.
            job_id:           Required. The job ID from ariba_procurement_reporting_job_submit.
            file_id:          Required. The file ID from the job status response
                              (e.g. 'Fk3enaqz1.zip'). A single job may produce
                              multiple files — call this once per file ID.
            include_inactive: Optional. If True, includes inactive/deleted records
                              in the downloaded result.
        """
        try:
            params: dict = {
                "realm": realm,
                "includeInactive": str(include_inactive).lower(),
            }
            result = await client.fetch_resource(
                JOB_RESULT_API, f"jobs/{job_id}/files/{file_id}", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    # ── SYNCHRONOUS API ───────────────────────────────────────────────────────
 
    @mcp.tool(
        name="ariba_procurement_reporting_sync_get",
        description=(
            "Synchronously retrieve procurement reporting data using a published view "
            "template from the Operational Reporting API for Procurement. "
            "Returns results immediately with pagination via pageToken. "
            "Recommended only for small datasets — for large extractions use the async "
            "job workflow (ariba_procurement_reporting_job_submit). "
            "Rate limit: max 3 requests/min, 50/hr, 300/day."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def procurement_reporting_sync_get(
        realm: str,
        view_template_name: str,
        filters: str | None = None,
        page_token: str | None = None,
    ) -> str:
        """
        Args:
            realm:              Required. The target SAP Ariba realm.
            view_template_name: Required. Name of a published view template.
                                e.g. 'RequisitionSystemView', 'KC_GetRequisitions'.
            filters:            Optional. JSON-encoded filter string. Example:
                                  '{"createdDateFrom":"2024-01-01T00:00:00Z",
                                    "createdDateTo":"2024-01-31T23:59:59Z",
                                    "status":["Submitted","Approved"]}'
                                Note: Special characters must be URL-encoded when
                                passed as query params.
            page_token:         Optional. Pagination token from a previous response
                                to retrieve the next page of results.
        """
        try:
            params: dict = {"realm": realm}
            if filters:
                params["filters"] = filters
            if page_token:
                params["pageToken"] = page_token
            result = await client.fetch_resource(
                SYNCHRONOUS_API,
                f"views/{view_template_name}",
                params,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)