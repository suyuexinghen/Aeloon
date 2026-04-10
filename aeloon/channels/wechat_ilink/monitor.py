"""Long-poll monitor for iLink updates."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from loguru import logger

from .client import ILinkClient
from .types import WeixinMessage


class ILinkMonitor:
    """Poll iLink updates and feed parsed messages to a handler."""

    def __init__(
        self,
        client: ILinkClient,
        handler: Callable[[WeixinMessage], Awaitable[None]],
        poll_interval_ms: int = 1000,
    ):
        self.client = client
        self.handler = handler
        self.poll_interval_ms = max(100, poll_interval_ms)
        self._cursor = ""

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run until *stop_event* is set."""
        while not stop_event.is_set():
            try:
                response = await self.client.get_updates(self._cursor)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("WeChat get_updates failed: {}", e)
                await asyncio.sleep(self.poll_interval_ms / 1000.0)
                continue

            if response.get("get_updates_buf"):
                self._cursor = str(response["get_updates_buf"])

            if response.get("ret", 0) != 0 and response.get("errcode", 0) != 0:
                logger.warning(
                    "WeChat server error ret={} errcode={} errmsg={}",
                    response.get("ret"),
                    response.get("errcode"),
                    response.get("errmsg"),
                )
                await asyncio.sleep(self.poll_interval_ms / 1000.0)
                continue

            for raw in response.get("msgs") or []:
                if not isinstance(raw, dict):
                    continue
                await self.handler(WeixinMessage.from_dict(raw))
