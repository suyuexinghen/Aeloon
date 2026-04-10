"""Plugin and service base classes.

Defines the abstract contracts that every Aeloon plugin and every
plugin-managed service must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aeloon.plugins._sdk.api import PluginAPI
    from aeloon.plugins._sdk.runtime import PluginRuntime


class ServiceStatus(str, Enum):
    """Lifecycle state of a :class:`PluginService`."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"


class Plugin(ABC):
    """Base class for all Aeloon plugins.

    Sub-classes **must** implement :meth:`register`.  Override
    :meth:`activate` and :meth:`deactivate` for async start/stop work.
    """

    @abstractmethod
    def register(self, api: PluginAPI) -> None:
        """Declare capabilities.

        Must be **synchronous**, **idempotent**, and **free of I/O**.
        """

    async def activate(self, api: PluginAPI) -> None:
        """Activation phase — I/O allowed, start services, warm caches."""

    async def deactivate(self) -> None:
        """Shutdown phase — release all resources."""

    def health_check(self) -> dict[str, Any]:
        """Return health status dict.  Default: ``{"status": "ok"}``."""
        return {"status": "ok"}


class PluginService(ABC):
    """Base class for long-running services managed by :class:`ServiceSupervisor`."""

    @abstractmethod
    async def start(self, runtime: PluginRuntime, config: Mapping[str, Any]) -> None:
        """Start the service.  Called by :class:`ServiceSupervisor`."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the service gracefully."""

    def health_check(self) -> dict[str, Any]:
        """Return service health status."""
        return {"status": "ok"}
