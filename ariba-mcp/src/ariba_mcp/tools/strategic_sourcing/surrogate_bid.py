"""
SAP Ariba Surrogate Bid API tools.

Workflow overview (all steps are async job-based):
  1. POST /jobs  (operation=export)  → jobId
  2. GET  /jobs/{jobId}              → poll until status == "Success", get fileId
  3. GET  /jobs/{jobId}/files/{fileId} → download the bid-response Excel sheet
  4. (Caller fills in bid values in the sheet, then re-uploads it as base64)
  5. POST /jobs  (operation=import)  → jobId
  6. GET  /jobs/{jobId}              → poll until status == "Success"
  7. POST /jobs  (operation=submit)  → jobId
  8. GET  /jobs/{jobId}              → poll until status == "Success"

Reference: SAP Ariba Surrogate Bid API – Document Version 2602 (2026-02)
Endpoint base: sourcing-externalsystem/v1/prod
"""

import json

from fastmcp import FastMCP

from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error

SURROGATE_BID_API = "sourcing-event-bid/v1/prod"


def register(mcp: FastMCP, client: AribaClient) -> None:

    # -------------------------------------------------------------------------
    # 1. Export – generate a bid-response sheet for a participant
    # -------------------------------------------------------------------------
    @mcp.tool(
        name="ariba_surrogate_bid_export",
        description=(
            "Step 1 of surrogate bidding: create an export job that generates a bid-response "
            "sheet (Excel) for a specific participant (supplier) in an SAP Ariba Sourcing event. "
            "Returns a jobId that must be polled with ariba_surrogate_bid_job_status until "
            "the status is 'Success', after which the sheet can be downloaded with "
            "ariba_surrogate_bid_download_file."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def surrogate_bid_export(
        realm: str,
        event_id: str,
        participant_id: str,
        user: str,
        password_adapter: str,
    ) -> str:
        """
        Args:
            realm:            Your SAP Ariba site name (e.g. 'MyCompany-T' for test).
            event_id:         ID of the sourcing event for which the bid will be submitted
                              (e.g. 'Doc3456789').
            participant_id:   User ID of the supplier/participant for whom the surrogate bid
                              will be submitted. The participant must be invited to the event.
            user:             User name of a buyer authorised to submit surrogate bids for
                              this event (as entered on the SAP Ariba sign-in page).
            password_adapter: Password adapter for the buyer user (e.g. 'PasswordAdapter1').
        """
        try:
            params = {
                "user": user,
                "passwordAdapter": password_adapter,
                "realm": realm,
            }
            payload = {
                "operation": "export",
                "eventId": event_id,
                "participantId": participant_id,
            }
            result = await client.post_resource(
                SURROGATE_BID_API, "jobs", payload, params=params, api_name="surrogate_bidding"
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    # -------------------------------------------------------------------------
    # 2. Job status – poll the status of any surrogate-bid job
    # -------------------------------------------------------------------------
    @mcp.tool(
        name="ariba_surrogate_bid_job_status",
        description=(
            "Poll the status of an SAP Ariba Surrogate Bid job (export, import, or submit). "
            "Call repeatedly until the returned status is 'Success' or 'Failed'. "
            "On success of an export job the response also contains the fileId needed to "
            "download the bid-response sheet."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def surrogate_bid_job_status(
        realm: str,
        job_id: str,
        user: str,
        password_adapter: str,
    ) -> str:
        """
        Args:
            realm:            Your SAP Ariba site name.
            job_id:           ID of the job returned by a previous export, import, or submit call.
            user:             Buyer user name used when the job was created.
            password_adapter: Password adapter for the buyer user.
        """
        try:
            params = {
                "user": user,
                "passwordAdapter": password_adapter,
                "realm": realm,
            }
            result = await client.fetch_resource(
                SURROGATE_BID_API, f"jobs/{job_id}", params, api_name="surrogate_bidding"
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    # -------------------------------------------------------------------------
    # 3. Download file – retrieve the generated bid-response Excel sheet
    # -------------------------------------------------------------------------
    @mcp.tool(
        name="ariba_surrogate_bid_download_file",
        description=(
            "Step 3 of surrogate bidding: download the bid-response Excel sheet that was "
            "generated by a successful export job. The fileId is obtained from the job-status "
            "response. The returned bytes should be saved as an .xlsx file, filled in with "
            "bid values, and then base64-encoded for the import step."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def surrogate_bid_download_file(
        realm: str,
        job_id: str,
        file_id: str,
        user: str,
        password_adapter: str,
    ) -> str:
        """
        Args:
            realm:            Your SAP Ariba site name.
            job_id:           ID of the completed export job.
            file_id:          File ID returned in the job-status response for the export job.
            user:             Buyer user name.
            password_adapter: Password adapter for the buyer user.
        """
        try:
            params = {
                "user": user,
                "passwordAdapter": password_adapter,
                "realm": realm,
            }
            result = await client.fetch_resource(
                SURROGATE_BID_API, f"jobs/{job_id}/files/{file_id}", params, api_name="surrogate_bidding"
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    # -------------------------------------------------------------------------
    # 4. Import – upload the filled bid-response sheet
    # -------------------------------------------------------------------------
    @mcp.tool(
        name="ariba_surrogate_bid_import",
        description=(
            "Step 4 of surrogate bidding: upload a completed bid-response Excel sheet "
            "(base64-encoded) back to SAP Ariba. This creates an import job whose status "
            "must be polled with ariba_surrogate_bid_job_status until 'Success'. "
            "After a successful import, call ariba_surrogate_bid_submit to finalise the bid."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def surrogate_bid_import(
        realm: str,
        event_id: str,
        participant_id: str,
        user: str,
        password_adapter: str,
        file_base64: str,
        file_name: str = "bid_response.xlsx",
    ) -> str:
        """
        Args:
            realm:            Your SAP Ariba site name.
            event_id:         ID of the sourcing event.
            participant_id:   User ID of the supplier/participant.
            user:             Buyer user name authorised to submit surrogate bids.
            password_adapter: Password adapter for the buyer user.
            file_base64:      Base64-encoded content of the completed bid-response Excel file.
            file_name:        File name to use for the uploaded sheet (default: 'bid_response.xlsx').
        """
        try:
            params = {
                "user": user,
                "passwordAdapter": password_adapter,
                "realm": realm,
            }
            payload = {
                "operation": "import",
                "eventId": event_id,
                "participantId": participant_id,
                "fileName": file_name,
                "fileContent": file_base64,
            }
            result = await client.post_resource(
                SURROGATE_BID_API, "jobs", payload, params=params, api_name="surrogate_bidding"
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)

    # -------------------------------------------------------------------------
    # 5. Submit – finalise and submit the surrogate bid
    # -------------------------------------------------------------------------
    @mcp.tool(
        name="ariba_surrogate_bid_submit",
        description=(
            "Step 5 of surrogate bidding: submit the imported bid on behalf of a participant "
            "in an SAP Ariba Sourcing event. Must be called after a successful import job. "
            "Creates a submit job whose status should be polled with "
            "ariba_surrogate_bid_job_status until 'Success'. "
            "WARNING: Do not use this tool and the SAP Ariba interactive UI to bid on the "
            "same event and participant at the same time – concurrent operations can cause "
            "bids to need resubmission."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def surrogate_bid_submit(
        realm: str,
        event_id: str,
        participant_id: str,
        user: str,
        password_adapter: str,
    ) -> str:
        """
        Args:
            realm:            Your SAP Ariba site name.
            event_id:         ID of the sourcing event.
            participant_id:   User ID of the supplier/participant for whom the bid is submitted.
            user:             User name of the buyer submitting the surrogate bid. The web service
                              records the bid as submitted by this user.
            password_adapter: Password adapter for the buyer user.
        """
        try:
            params = {
                "user": user,
                "passwordAdapter": password_adapter,
                "realm": realm,
            }
            payload = {
                "operation": "submit",
                "eventId": event_id,
                "participantId": participant_id,
            }
            result = await client.post_resource(
                SURROGATE_BID_API, "jobs", payload, params=params, api_name="surrogate_bidding"
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
