"""Plugin registry — single source of truth for all plugin metadata.

The registry tracks every plugin's lifecycle state and the records it has
committed (commands, tools, services, …).  All mutations go through
:meth:`commit_plugin` / :meth:`rollback_plugin` to prevent partial state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

from loguru import logger

from aeloon.plugins._sdk.base import Plugin
from aeloon.plugins._sdk.manifest import PluginManifest
from aeloon.plugins._sdk.types import (
    CLIRecord,
    CommandMiddlewareRecord,
    CommandRecord,
    ConfigSchemaRecord,
    HookRecord,
    MiddlewareRecord,
    ServiceRecord,
    StatusProviderRecord,
    ToolRecord,
)


class RegistrationConflictError(Exception):
    """Raised when a name is already registered by a different plugin."""


@dataclasses.dataclass
class PluginRecord:
    """Top-level record tracking a plugin's lifecycle state."""

    plugin_id: str
    manifest: PluginManifest
    instance: Plugin
    api: Any  # PluginAPI — kept as Any to avoid circular import
    status: Literal["discovered", "registered", "active", "error"] = "discovered"
    error: str | None = None


class PluginRegistry:
    """Central metadata-rich registry with staged commit / rollback."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginRecord] = {}
        self._commands: dict[str, CommandRecord] = {}
        self._tools: dict[str, ToolRecord] = {}
        self._services: dict[str, ServiceRecord] = {}
        self._middlewares: dict[str, MiddlewareRecord] = {}
        self._command_middlewares: dict[str, CommandMiddlewareRecord] = {}
        self._cli: dict[str, CLIRecord] = {}
        self._hooks: dict[str, list[HookRecord]] = {}
        self._config_schemas: dict[str, ConfigSchemaRecord] = {}
        self._status_providers: dict[str, StatusProviderRecord] = {}

    # ------------------------------------------------------------------
    # Plugin record management
    # ------------------------------------------------------------------

    def add_plugin(self, record: PluginRecord) -> None:
        """Register a plugin record in ``discovered`` state."""
        self._plugins[record.plugin_id] = record

    def set_status(
        self,
        plugin_id: str,
        status: Literal["discovered", "registered", "active", "error"],
        error: str | None = None,
    ) -> None:
        record = self._plugins.get(plugin_id)
        if record:
            record.status = status
            record.error = error

    def get_plugin(self, plugin_id: str) -> PluginRecord | None:
        return self._plugins.get(plugin_id)

    @property
    def plugins(self) -> Mapping[str, PluginRecord]:
        return dict(self._plugins)

    # ------------------------------------------------------------------
    # Staged commit / rollback
    # ------------------------------------------------------------------

    def commit_plugin(
        self,
        plugin_id: str,
        *,
        commands: list[CommandRecord] | None = None,
        tools: list[ToolRecord] | None = None,
        services: list[ServiceRecord] | None = None,
        middlewares: list[MiddlewareRecord] | None = None,
        command_middlewares: list[CommandMiddlewareRecord] | None = None,
        cli: list[CLIRecord] | None = None,
        hooks: list[HookRecord] | None = None,
        config_schemas: list[ConfigSchemaRecord] | None = None,
        status_providers: list[StatusProviderRecord] | None = None,
    ) -> None:
        """Atomically commit all records for *plugin_id*.

        Raises :class:`RegistrationConflictError` if any name collides
        with a record already committed by a *different* plugin.  On
        conflict **nothing** is written (atomic check-then-write).
        """
        commands = commands or []
        tools = tools or []
        services = services or []
        middlewares = middlewares or []
        command_middlewares = command_middlewares or []
        cli = cli or []
        hooks = hooks or []
        config_schemas = config_schemas or []
        status_providers = status_providers or []

        # --- Conflict detection (all-or-nothing) ---
        for cmd in commands:
            existing = self._commands.get(cmd.name)
            if existing and existing.plugin_id != plugin_id:
                raise RegistrationConflictError(
                    f"Command '{cmd.name}' already registered by '{existing.plugin_id}'"
                )
        for tool in tools:
            existing = self._tools.get(tool.name)
            if existing and existing.plugin_id != plugin_id:
                raise RegistrationConflictError(
                    f"Tool '{tool.name}' already registered by '{existing.plugin_id}'"
                )
        for svc in services:
            existing = self._services.get(svc.full_id)
            if existing and existing.plugin_id != plugin_id:
                raise RegistrationConflictError(
                    f"Service '{svc.full_id}' already registered by '{existing.plugin_id}'"
                )
        for cli_rec in cli:
            existing = self._cli.get(cli_rec.name)
            if existing and existing.plugin_id != plugin_id:
                raise RegistrationConflictError(
                    f"CLI '{cli_rec.name}' already registered by '{existing.plugin_id}'"
                )
        for sp in status_providers:
            existing = self._status_providers.get(sp.name)
            if existing and existing.plugin_id != plugin_id:
                raise RegistrationConflictError(
                    f"Status provider '{sp.name}' already registered by '{existing.plugin_id}'"
                )

        # --- Write phase (no early return possible) ---
        for cmd in commands:
            self._commands[cmd.name] = cmd
        for tool in tools:
            self._tools[tool.name] = tool
        for svc in services:
            self._services[svc.full_id] = svc
        for mw in middlewares:
            qualified = f"{plugin_id}.{mw.name}"
            self._middlewares[qualified] = mw
        for mw in command_middlewares:
            qualified = f"{plugin_id}.{mw.name}"
            self._command_middlewares[qualified] = mw
        for cli_rec in cli:
            self._cli[cli_rec.name] = cli_rec
        for hook in hooks:
            self._hooks.setdefault(hook.event, []).append(hook)
            self._hooks[hook.event].sort(key=lambda h: h.priority, reverse=True)
        for cs in config_schemas:
            self._config_schemas[plugin_id] = cs
        for sp in status_providers:
            self._status_providers[sp.name] = sp

        self.set_status(plugin_id, "registered")
        logger.debug("Plugin '{}' records committed", plugin_id)

    def rollback_plugin(self, plugin_id: str) -> None:
        """Remove **all** records associated with *plugin_id*."""
        self._commands = {k: v for k, v in self._commands.items() if v.plugin_id != plugin_id}
        self._tools = {k: v for k, v in self._tools.items() if v.plugin_id != plugin_id}
        self._services = {k: v for k, v in self._services.items() if v.plugin_id != plugin_id}
        self._middlewares = {k: v for k, v in self._middlewares.items() if v.plugin_id != plugin_id}
        self._command_middlewares = {
            k: v for k, v in self._command_middlewares.items() if v.plugin_id != plugin_id
        }
        self._cli = {k: v for k, v in self._cli.items() if v.plugin_id != plugin_id}
        for event in self._hooks:
            self._hooks[event] = [h for h in self._hooks[event] if h.plugin_id != plugin_id]
        self._config_schemas.pop(plugin_id, None)
        self._status_providers = {
            k: v for k, v in self._status_providers.items() if v.plugin_id != plugin_id
        }
        self._plugins.pop(plugin_id, None)
        logger.debug("Plugin '{}' records rolled back", plugin_id)

    # ------------------------------------------------------------------
    # Query methods (read-only copies)
    # ------------------------------------------------------------------

    @property
    def commands(self) -> dict[str, CommandRecord]:
        return dict(self._commands)

    @property
    def tools(self) -> dict[str, ToolRecord]:
        return dict(self._tools)

    @property
    def services(self) -> dict[str, ServiceRecord]:
        return dict(self._services)

    @property
    def middlewares(self) -> list[MiddlewareRecord]:
        return list(self._middlewares.values())

    @property
    def command_middlewares(self) -> list[CommandMiddlewareRecord]:
        return list(self._command_middlewares.values())

    @property
    def cli_registrars(self) -> dict[str, CLIRecord]:
        return dict(self._cli)

    def hooks_for_event(self, event: str) -> list[HookRecord]:
        """Return hooks for *event*, sorted by priority (highest first)."""
        return list(self._hooks.get(event, []))

    def get_config_schema(self, plugin_id: str) -> type | None:
        rec = self._config_schemas.get(plugin_id)
        return rec.schema_cls if rec else None

    @property
    def status_providers(self) -> list[StatusProviderRecord]:
        """All status providers ordered by priority (highest first)."""
        return sorted(self._status_providers.values(), key=lambda r: r.priority, reverse=True)
