"""Async iLink HTTP client."""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from .types import Credentials

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"


class ILinkClient:
    """Minimal async client for the iLink bot HTTP API."""

    def __init__(self, credentials: Credentials):
        self.credentials = credentials
        self.base_url = credentials.base_url or DEFAULT_BASE_URL
        self.bot_id = credentials.ilink_bot_id
        self.http = httpx.AsyncClient(timeout=40.0)
        self._wechat_uin = base64.b64encode(str(int.from_bytes(os.urandom(4), "little")).encode())
        self._wechat_uin = self._wechat_uin.decode()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def get_updates(self, cursor: str) -> dict[str, Any]:
        return await self._post(
            "/ilink/bot/getupdates",
            {
                "get_updates_buf": cursor,
                "base_info": {"channel_version": "1.0.0"},
            },
            timeout=40.0,
        )

    async def send_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ilink/bot/sendmessage", payload, timeout=15.0)

    async def get_config(self, user_id: str, context_token: str = "") -> dict[str, Any]:
        return await self._post(
            "/ilink/bot/getconfig",
            {
                "ilink_user_id": user_id,
                "context_token": context_token,
                "base_info": {},
            },
            timeout=10.0,
        )

    async def send_typing(self, user_id: str, typing_ticket: str, status: int) -> dict[str, Any]:
        return await self._post(
            "/ilink/bot/sendtyping",
            {
                "ilink_user_id": user_id,
                "typing_ticket": typing_ticket,
                "status": status,
                "base_info": {},
            },
            timeout=10.0,
        )

    async def get_upload_url(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ilink/bot/getuploadurl", payload, timeout=15.0)

    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "Authorization": f"Bearer {self.credentials.bot_token}",
            "X-WECHAT-UIN": self._wechat_uin,
        }

    async def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        response = await self.http.post(
            f"{self.base_url}{path}",
            headers=self.headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected iLink response type: {type(data).__name__}")
        return data
