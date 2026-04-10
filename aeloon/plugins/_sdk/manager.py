"""Plugin manager and service supervisor.

:class:`PluginManager` orchestrates the full plugin lifecycle:
discover → validate → resolve → register → activate → shutdown.

:class:`ServiceSupervisor` manages :class:`PluginService` instances with
startup/shutdown timeouts, restart policies, and health checks.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from aeloon.plugins._sdk.api import PluginAPI
from aeloon.plugins._sdk.base import PluginService, ServiceStatus
from aeloon.plugins._sdk.discovery import PluginDiscovery
from aeloon.plugins._sdk.hooks import HookDispatcher, HookEvent
from aeloon.plugins._sdk.loader import CircularDependencyError, PluginLoader
from aeloon.plugins._sdk.registry import PluginRecord, PluginRegistry
from aeloon.plugins._sdk.runtime import PluginRuntime
from aeloon.plugins._sdk.state_store import PluginStateStore
from aeloon.plugins._sdk.types import ServiceRecord

if TYPE_CHECKING:
    from aeloon.plugins._sdk.discovery import PluginCandidate


# ---------------------------------------------------------------------------
# ServiceSupervisor
# ---------------------------------------------------------------------------


class ServiceSupervisor:
    """Manages lifecycle of :class:`PluginService` instances."""

    def __init__(self, hook_dispatcher: HookDispatcher | None = None) -> None:
        self._instances: dict[str, PluginService] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._records: dict[str, ServiceRecord] = {}
        self._hooks = hook_dispatcher

    async def start_service(
        self,
        record: ServiceRecord,
        runtime: PluginRuntime,
        config: Mapping[str, Any],
    ) -> None:
        """Start a service with timeout and status tracking."""
        full_id = record.full_id
        record.status = ServiceStatus.STARTING

        instance = record.service_cls()
        self._instances[full_id] = instance
        self._records[full_id] = record

        try:
            await asyncio.wait_for(
                instance.start(runtime, config),
                timeout=record.policy.startup_timeout_seconds,
            )
            record.status = ServiceStatus.RUNNING
            logger.info("Service '{}' started", full_id)
            # Dispatch SERVICE_STARTED hook (after successful start, outside try)
            if self._hooks:
                try:
                    await self._hooks.dispatch_notify(
                        HookEvent.SERVICE_STARTED,
                        service_id=full_id,
                        plugin_id=record.plugin_id,
                    )
                except Exception:
                    logger.opt(exception=True).debug("SERVICE_STARTED hook dispatch failed")
        except asyncio.TimeoutError:
            record.status = ServiceStatus.FAILED
            logger.error("Service '{}' startup timed out", full_id)
            raise
        except Exception:
            record.status = ServiceStatus.FAILED
            logger.exception("Service '{}' failed to start", full_id)
            raise

    async def stop_service(self, full_id: str) -> None:
        """Stop a service with shutdown timeout."""
        instance = self._instances.get(full_id)
        record = self._records.get(full_id)
        if not instance or not record:
            return

        try:
            await asyncio.wait_for(
                instance.stop(),
                timeout=record.policy.shutdown_timeout_seconds,
            )
            record.status = ServiceStatus.STOPPED
            logger.info("Service '{}' stopped", full_id)
        except asyncio.TimeoutError:
            record.status = ServiceStatus.STOPPED
            logger.warning("Service '{}' shutdown timed out, forced stop", full_id)
        except Exception:
            record.status = ServiceStatus.FAILED
            logger.exception("Service '{}' failed to stop cleanly", full_id)
        finally:
            self._instances.pop(full_id, None)
            task = self._tasks.pop(full_id, None)
            if task and not task.done():
                task.cancel()
            # Dispatch SERVICE_STOPPED hook
            if self._hooks:
                try:
                    await self._hooks.dispatch_notify(
                        HookEvent.SERVICE_STOPPED,
                        service_id=full_id,
                    )
                except Exception:
                    logger.opt(exception=True).debug("SERVICE_STOPPED hook dispatch failed")

    async def restart_service(
        self,
        full_id: str,
        runtime: PluginRuntime,
        config: Mapping[str, Any],
    ) -> None:
        """Stop then start a service, respecting restart policy."""
        record = self._records.get(full_id)
        if not record:
            return

        if record.policy.restart_policy == "never":
            logger.info("Service '{}' restart_policy=never, not restarting", full_id)
            return

        if record.restart_count >= record.policy.max_restarts:
            logger.error(
                "Service '{}' exceeded max_restarts ({}), not restarting",
                full_id,
                record.policy.max_restarts,
            )
            record.status = ServiceStatus.FAILED
            return

        await self.stop_service(full_id)
        await asyncio.sleep(record.policy.restart_delay_seconds)
        record.restart_count += 1
        await self.start_service(record, runtime, config)

    def health_check(self, full_id: str) -> dict[str, Any]:
        """Return health status for a service."""
        instance = self._instances.get(full_id)
        record = self._records.get(full_id)
        if not instance or not record:
            return {"status": "not_found"}
        base: dict[str, Any] = {
            "status": record.status.value,
            "restart_count": record.restart_count,
        }
        try:
            base.update(instance.health_check())
        except Exception:
            base["health_check_error"] = True
        return base

    async def stop_all(self) -> None:
        """Stop all running services."""
        for full_id in list(self._instances):
            await self.stop_service(full_id)

    def get_status(self, full_id: str) -> ServiceStatus:
        record = self._records.get(full_id)
        return record.status if record else ServiceStatus.STOPPED


# ---------------------------------------------------------------------------
# BootResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BootResult:
    """Result of :meth:`PluginManager.boot`."""

    loaded: list[str] = dataclasses.field(default_factory=list)
    failed: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------


class PluginManager:
    """Orchestrates the full plugin lifecycle."""

    ACTIVATION_TIMEOUT_SECONDS: float = 30.0

    def __init__(
        self,
        registry: PluginRegistry,
        discovery: PluginDiscovery,
        loader: PluginLoader,
        hook_dispatcher: HookDispatcher,
        agent_loop: Any,
        plugin_config: dict[str, dict[str, Any]],
        storage_base: Path,
        state_store: PluginStateStore | None = None,
    ) -> None:
        self._registry = registry
        self._discovery = discovery
        self._loader = loader
        self._hooks = hook_dispatcher
        self._agent_loop = agent_loop
        self._plugin_config = plugin_config
        self._storage_base = storage_base
        self._state_store = state_store
        self._supervisor = ServiceSupervisor(hook_dispatcher=hook_dispatcher)
        self._activation_order: list[str] = []

    @property
    def registry(self) -> PluginRegistry:
        return self._registry

    @property
    def supervisor(self) -> ServiceSupervisor:
        return self._supervisor

    async def boot(self) -> BootResult:
        """Full boot sequence: discover → validate → resolve → register → activate."""
        result = BootResult()

        # 1. Discover
        candidates = self._discovery.discover_all()
        logger.info("Discovered {} plugin candidate(s)", len(candidates))

        # 2. Validate requirements
        valid: list[PluginCandidate] = []
        for candidate in candidates:
            errors = self._loader.validate_candidate(candidate)
            if errors:
                logger.error(
                    "Plugin '{}' validation failed: {}",
                    candidate.manifest.id,
                    "; ".join(errors),
                )
                result.failed.append(candidate.manifest.id)
            else:
                valid.append(candidate)

        # 3. Filter by enabled state (config + state store)
        enabled = [
            c for c in valid if self._plugin_config.get(c.manifest.id, {}).get("enabled", True)
        ]
        if self._state_store:
            state_filtered: list[PluginCandidate] = []
            for c in enabled:
                state = self._state_store.get(c.manifest.id)
                if state and not state.enabled:
                    logger.info("Plugin '{}' deactivated via state store", c.manifest.id)
                    continue
                state_filtered.append(c)
            enabled = state_filtered

            # Record state entries for all discovered plugins
            for c in valid:
                if not self._state_store.get(c.manifest.id):
                    from aeloon.plugins._sdk.state_store import PluginState

                    source = "bundled" if c.source == 10 else "workspace"
                    self._state_store.set(
                        PluginState(
                            plugin_id=c.manifest.id,
                            installed_at="",
                            source=source,
                            enabled=True,
                            version=c.manifest.version,
                        )
                    )

        # 4. Resolve dependency order
        try:
            ordered = self._loader.resolve_load_order(enabled)
        except CircularDependencyError as exc:
            logger.error("Circular dependency: {}", exc)
            for pid in exc.cycle_members:
                result.failed.append(pid)
            ordered = [c for c in enabled if c.manifest.id not in exc.cycle_members]

        # 5. Load, register, activate each plugin
        for candidate in ordered:
            pid = candidate.manifest.id
            plugin_cfg = self._plugin_config.get(pid, {})

            try:
                # Import and instantiate
                cls = self._loader.load_plugin_class(candidate.manifest)
                instance = self._loader.instantiate(cls)

                # Create runtime and API
                runtime = PluginRuntime(
                    agent_loop=self._agent_loop,
                    plugin_id=pid,
                    config=plugin_cfg,
                    storage_base=self._storage_base,
                )
                api = PluginAPI(
                    plugin_id=pid,
                    version=candidate.manifest.version,
                    config=plugin_cfg,
                    runtime=runtime,
                    registry=self._registry,
                )
                # Wire supervisor for service control
                api._supervisor = self._supervisor  # type: ignore[attr-defined]

                # Record in registry as discovered
                self._registry.add_plugin(
                    PluginRecord(
                        plugin_id=pid,
                        manifest=candidate.manifest,
                        instance=instance,
                        api=api,
                    )
                )

                # Phase 1: register (synchronous, staged)
                instance.register(api)
                api._commit()

                # Dispatch PLUGIN_REGISTERED hook
                await self._hooks.dispatch_notify(
                    HookEvent.PLUGIN_REGISTERED,
                    plugin_id=pid,
                )

                # Phase 2: activate (async, timeout-guarded, error-isolated)
                try:
                    await asyncio.wait_for(
                        instance.activate(api),
                        timeout=self.ACTIVATION_TIMEOUT_SECONDS,
                    )
                    self._registry.set_status(pid, "active")
                    self._activation_order.append(pid)
                    result.loaded.append(pid)

                    await self._hooks.dispatch_notify(HookEvent.PLUGIN_ACTIVATED, plugin_id=pid)
                except (asyncio.TimeoutError, Exception) as exc:
                    self._registry.set_status(pid, "error", error=str(exc))
                    result.failed.append(pid)
                    logger.error("Plugin '{}' activation failed: {}", pid, exc)

            except Exception as exc:
                # register() or import failed — rollback all records
                self._registry.rollback_plugin(pid)
                result.failed.append(pid)
                logger.error("Plugin '{}' load/register failed: {}", pid, exc)

        return result

    async def shutdown(self) -> None:
        """Shutdown: stop services first, then deactivate plugins in reverse order."""
        # 1. Stop all supervised services
        await self._supervisor.stop_all()

        # 2. Deactivate plugins in reverse activation order
        for pid in reversed(self._activation_order):
            record = self._registry.get_plugin(pid)
            if not record:
                continue
            try:
                await asyncio.wait_for(
                    record.instance.deactivate(),
                    timeout=self.ACTIVATION_TIMEOUT_SECONDS,
                )
                self._registry.set_status(pid, "discovered")
                await self._hooks.dispatch_notify(HookEvent.PLUGIN_DEACTIVATED, plugin_id=pid)
            except Exception as exc:
                logger.error("Plugin '{}' deactivation failed: {}", pid, exc)

        self._activation_order.clear()
