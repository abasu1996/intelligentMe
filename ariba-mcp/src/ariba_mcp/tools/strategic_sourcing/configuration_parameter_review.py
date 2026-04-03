"""Configuration Parameter Review API.
 
Owner: <assign owner>
Docs: https://help.sap.com/doc/c1aedced5c044a41b84be5312db93fc1/cloud/en-US/index.html
 
This API allows customer developers to review the current and default values
of SAP Ariba Sourcing and SAP Ariba Buying configuration parameters programmatically,
without needing to log into the Intelligent Configuration Manager (ICM) UI.
 
Use cases:
  - Audit current vs. default parameter values across realms
  - Automate parameter drift detection between test and production realms
  - Verify parameter state before or after a deployment
 
Endpoints implemented:
  GET /parameters                    – list all configuration parameters for a realm
  GET /parameters/{parameterName}    – retrieve a specific parameter by name
 
Prerequisites:
  - SAP Ariba Developer Portal access (Procurement or Sourcing tab)
  - Customer developer role — this API is for customer applications only,
    not third-party or partner integrations
  - OAuth authentication configured (handled by AribaClient)
  - The calling user must have Customer Administrator role in the realm
 
Notes:
  - Returns both currentValue and defaultValue for each parameter
  - Read-only API — parameter values cannot be modified via this API
  - To modify parameters, use the Intelligent Configuration Manager (ICM) UI
  - Covers both SAP Ariba Sourcing and SAP Ariba Buying parameters
"""
 
import json
 
from fastmcp import FastMCP
 
from ariba_mcp.client import AribaClient
from ariba_mcp.errors import handle_ariba_error
 
# Production URL confirmed from API docs:
# https://openapi.ariba.com/api/sourcing-config-parameter/v1/prod
CONFIGURATION_PARAMETER_REVIEW_API = "sourcing-config-parameter/v1/prod"
 
 
def register(mcp: FastMCP, client: AribaClient) -> None:
 
    @mcp.tool(
        name="ariba_config_parameters_list",
        description=(
            "Retrieve a list of all configuration parameters and their current and default values "
            "for a SAP Ariba Sourcing or Buying realm. "
            "Useful for auditing parameter settings, detecting drift from defaults, "
            "or comparing test vs. production realm configurations."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def config_parameters_list(
        realm: str,
        filter: str | None = None,
        top: int | None = None,
        skip: int | None = None,
    ) -> str:
        """
        Args:
            realm:  Required. The target SAP Ariba realm (e.g. 'MyCompanyS4').
                    Must be a realm the calling user has Customer Administrator
                    access to.
            filter: Optional. OData $filter expression. Supported fields:
                      parameterName  – name of the configuration parameter
                                       (e.g. 'Application.Sourcing.AllowableProjectType')
                      currentValue   – the value currently set in the realm
                      defaultValue   – the product default value
                      isModified     – true/false, whether the value differs from default
                    Example:
                      "isModified eq true"
                      "parameterName eq 'Application.Sourcing.EnablePrivateMessaging'"
            top:    Optional. Max records to return per page (default 10).
            skip:   Optional. Number of records to skip for pagination (default 0).
        """
        try:
            params: dict = {"realm": realm}
            if filter:
                params["$filter"] = filter
            if top is not None:
                params["$top"] = top
            if skip is not None:
                params["$skip"] = skip
            result = await client.fetch_resource(
                CONFIGURATION_PARAMETER_REVIEW_API, "parameters", params
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)
 
    @mcp.tool(
        name="ariba_config_parameter_get",
        description=(
            "Retrieve a specific SAP Ariba configuration parameter by its exact name. "
            "Returns the current value set in the realm, the product default value, "
            "and whether the parameter has been modified from its default. "
            "Use ariba_config_parameters_list to discover valid parameter names."
        ),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def config_parameter_get(
        realm: str,
        parameter_name: str,
    ) -> str:
        """
        Args:
            realm:          Required. The target SAP Ariba realm (e.g. 'MyCompanyS4').
            parameter_name: Required. The exact name of the configuration parameter
                            to retrieve. Parameter names follow dot-notation convention,
                            e.g.:
                              'Application.Sourcing.AllowableProjectType'
                              'Application.Sourcing.EnablePrivateMessaging'
                              'Application.Sourcing.EnableRFXCustomTemplates'
                              'Application.Sourcing.RestrictCopyProjectToOwner'
                              'Application.Base.EmailApprovalEnabled'
                            Use ariba_config_parameters_list to discover valid names.
        """
        try:
            params: dict = {"realm": realm}
            result = await client.fetch_resource(
                CONFIGURATION_PARAMETER_REVIEW_API,
                f"parameters/{parameter_name}",
                params,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return handle_ariba_error(e)