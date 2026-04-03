"""OAuth 2.0 client-credentials authentication for SAP Ariba APIs.

Endpoint: POST {oauth_url}/v2/oauth/token?grant_type=client_credentials
Auth:     HTTP Basic (client_id:client_secret)
Docs:     https://help.sap.com/docs/ariba-apis/help-for-sap-ariba-developer-portal/making-of-rest-api-calls-with-oauth-access-token-and-application-key
"""

import asyncio
import time

import httpx

from ariba_mcp.config import AribaSettings


class DirectAuthClient:
    """Auth client with explicit credentials — use when an API has its own client ID/secret/key."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        api_key: str,
        oauth_url: str = "https://api.ariba.com",
        timeout: int = 30,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._api_key = api_key
        self._oauth_url = oauth_url
        self._timeout = timeout
        self._token: str | None = None
        self._expires_at: float = 0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Return a valid access token, refreshing if expired."""
        async with self._lock:
            if self._token and time.time() < (self._expires_at - 60):
                return self._token

            async with httpx.AsyncClient() as http:
                response = await http.post(
                    f"{self._oauth_url}/v2/oauth/token",
                    data={"grant_type": "client_credentials"},
                    auth=(self._client_id, self._client_secret),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()

            self._token = data["access_token"]
            self._expires_at = time.time() + data.get("expires_in", 1440)
            return self._token

    async def get_headers(self) -> dict[str, str]:
        """Return header set for an authenticated Ariba API request."""
        token = await self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "apiKey": self._api_key,
            "Accept": "application/json",
        }


class AribaAuthClient:
    """Manages OAuth token acquisition and caching for Ariba APIs."""

    def __init__(self, settings: AribaSettings) -> None:
        self._settings = settings
        self._token: str | None = None
        self._expires_at: float = 0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Return a valid access token, refreshing if expired."""
        async with self._lock:
            if self._token and time.time() < (self._expires_at - 60):
                return self._token

            async with httpx.AsyncClient() as http:
                response = await http.post(
                    f"{self._settings.ariba_oauth_url}/v2/oauth/token",
                    data={"grant_type": "client_credentials"},
                    auth=(self._settings.ariba_client_id, self._settings.ariba_client_secret),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                data = response.json()

            self._token = data["access_token"]
            self._expires_at = time.time() + data.get("expires_in", 1440)
            return self._token

    async def get_headers(self) -> dict[str, str]:
        """Return header set for an authenticated Ariba API request."""
        token = await self.get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "apiKey": self._settings.ariba_api_key,
            "Accept": "application/json",
        }
        if self._settings.ariba_network_id:
            headers["X-ARIBA-NETWORK-ID"] = self._settings.ariba_network_id
        return headers
