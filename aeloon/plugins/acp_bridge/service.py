"""ACP Connection Service — managed by ServiceSupervisor.

Owns the ``ACPClient`` lifecycle and exposes it for command handlers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aeloon.plugins._sdk.acp.client import ACPClient
from aeloon.plugins._sdk.acp.types import BackendProfile
from aeloon.plugins._sdk.base import PluginService

if TYPE_CHECKING:
    from aeloon.plugins._sdk.runtime import PluginRuntime

logger = logging.getLogger(__name__)


class ACPConnectionService(PluginService):
    """Supervised service that manages a single ACP backend connection."""

    def __init__(self) -> None:
        self._client: ACPClient = ACPClient()
        self._active_profile: BackendProfile | None = None
        self._runtime: PluginRuntime | None = None

    @property
    def client(self) -> ACPClient:
        return self._client

    @property
    def active_profile(self) -> BackendProfile | None:
        return self._active_profile

    async def start(self, runtime: PluginRuntime, config: Mapping[str, Any]) -> None:
        """Called by ServiceSupervisor during plugin activation."""
        self._runtime = runtime
        logger.info("ACP connection service started (not yet connected)")

    async def stop(self) -> None:
        """Disconnect and clean up."""
        if self._client.is_connected:
            await self._client.disconnect()
            logger.info("ACP connection service stopped")
        self._active_profile = None

    async def connect(self, profile: BackendProfile) -> None:
        """Connect to an ACP backend using the given profile."""
        if self._client.is_connected:
            await self._client.disconnect()

        self._active_profile = profile
        try:
            await self._client.connect(profile)
            logger.info("Connected to ACP backend '%s'", profile.name)
        except Exception:
            self._active_profile = None
            raise

    async def disconnect(self) -> None:
        """Disconnect from the current backend."""
        await self._client.disconnect()
        self._active_profile = None

    def health_check(self) -> dict[str, Any]:
        """Return connection health for status reporting."""
        health = self._client.health_check()
        if self._active_profile:
            health["profile"] = self._active_profile.name
        return health
