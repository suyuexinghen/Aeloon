"""Plugin registration API.

:class:`PluginAPI` is the primary surface injected into
:meth:`Plugin.register` and :meth:`Plugin.activate`.  All ``register_*``
methods are **synchronous**, **idempotent**, and **free of I/O** — they
accumulate pending records that are committed atomically on success or
rolled back on failure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from aeloon.plugins._sdk.base import PluginService, ServiceStatus
from aeloon.plugins._sdk.cli_builder import build_cli_group_builder
from aeloon.plugins._sdk.hooks import HookType
from aeloon.plugins._sdk.types import (
    CLIBuilder,
    CLICommandSpec,
    CLIRecord,
    CommandHandler,
    CommandMiddleware,
    CommandMiddlewareRecord,
    CommandRecord,
    ConfigSchemaRecord,
    HookRecord,
    MiddlewareRecord,
    ServicePolicy,
    ServiceRecord,
    StatusProviderRecord,
    ToolRecord,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from aeloon.plugins._sdk.registry import PluginRegistry
    from aeloon.plugins._sdk.runtime import PluginRuntime


class PluginAPI:
    """Registration API injected into ``Plugin.register()`` and ``Plugin.activate()``."""

    def __init__(
        self,
        plugin_id: str,
        version: str,
        config: Mapping[str, Any],
        runtime: PluginRuntime,
        registry: PluginRegistry,
    ) -> None:
        self._plugin_id = plugin_id
        self._version = version
        self._config = config
        self._runtime = runtime
        self._registry = registry

        # Pending records — committed only after register() succeeds.
        self._pending_commands: list[CommandRecord] = []
        self._pending_tools: list[ToolRecord] = []
        self._pending_services: list[ServiceRecord] = []
        self._pending_middlewares: list[MiddlewareRecord] = []
        self._pending_command_middlewares: list[CommandMiddlewareRecord] = []
        self._pending_cli: list[CLIRecord] = []
        self._pending_cli_specs: dict[str, list[CLICommandSpec]] = {}
        self._pending_hooks: list[HookRecord] = []
        self._pending_config_schemas: list[ConfigSchemaRecord] = []
        self._pending_status_providers: list[StatusProviderRecord] = []

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._plugin_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config

    @property
    def runtime(self) -> PluginRuntime:
        return self._runtime

    # ------------------------------------------------------------------
    # Registration methods (synchronous, idempotent, no I/O)
    # ------------------------------------------------------------------

    def register_command(
        self, name: str, handler: CommandHandler, *, description: str = ""
    ) -> None:
        """Register a slash command.  Idempotent: re-registering same name replaces."""
        self._pending_commands = [r for r in self._pending_commands if r.name != name]
        self._pending_commands.append(
            CommandRecord(
                plugin_id=self._plugin_id,
                name=name,
                handler=handler,
                description=description,
            )
        )

    def register_tool(self, tool: Any) -> None:
        """Register an agent tool (must have a ``name`` attribute)."""
        tool_name = tool.name
        self._pending_tools = [r for r in self._pending_tools if r.name != tool_name]
        self._pending_tools.append(ToolRecord(plugin_id=self._plugin_id, name=tool_name, tool=tool))

    def register_service(
        self,
        name: str,
        service_cls: type[PluginService],
        *,
        policy: ServicePolicy | None = None,
    ) -> None:
        """Register a managed service."""
        full_id = f"{self._plugin_id}.{name}"
        self._pending_services = [r for r in self._pending_services if r.name != name]
        self._pending_services.append(
            ServiceRecord(
                plugin_id=self._plugin_id,
                name=name,
                full_id=full_id,
                service_cls=service_cls,
                policy=policy or ServicePolicy(),
            )
        )

    def register_cli(
        self,
        name: str,
        builder: CLIBuilder | None = None,
        *,
        commands: tuple[CLICommandSpec, ...] = (),
        handler: CommandHandler | None = None,
        description: str = "",
    ) -> None:
        """Register a plugin CLI group and, optionally, its slash handler."""
        if handler is not None:
            self.register_command(name, handler, description=description)

        if builder is None:
            if not commands:
                raise ValueError("register_cli() requires a builder or declarative commands")
            builder = build_cli_group_builder(self._plugin_id, commands)

        self._pending_cli = [r for r in self._pending_cli if r.name != name]
        self._pending_cli_specs[name] = list(commands)
        self._pending_cli.append(
            CLIRecord(
                plugin_id=self._plugin_id,
                name=name,
                builder=builder,
                commands=commands,
            )
        )

    def register_cli_command(self, spec: CLICommandSpec) -> None:
        """Register one declarative CLI command under a plugin-owned group."""
        specs = list(self._pending_cli_specs.get(spec.group_name, []))
        specs = [existing for existing in specs if existing.command_name != spec.command_name]
        specs.append(spec)
        self.register_cli(spec.group_name, commands=tuple(specs))

    def register_middleware(self, name: str, middleware: Any) -> None:
        """Register an agent middleware instance."""
        self._pending_middlewares = [r for r in self._pending_middlewares if r.name != name]
        self._pending_middlewares.append(
            MiddlewareRecord(plugin_id=self._plugin_id, name=name, middleware=middleware)
        )

    def register_command_middleware(self, name: str, middleware: CommandMiddleware) -> None:
        """Register a dispatcher-level command middleware instance."""
        self._pending_command_middlewares = [
            record for record in self._pending_command_middlewares if record.name != name
        ]
        self._pending_command_middlewares.append(
            CommandMiddlewareRecord(
                plugin_id=self._plugin_id,
                name=name,
                middleware=middleware,
            )
        )

    def register_hook(
        self,
        event: str,
        handler: Callable[..., Any],
        *,
        kind: HookType = HookType.NOTIFY,
        priority: int = 0,
        matcher: str | None = None,
    ) -> None:
        """Register a lifecycle / event hook.

        Args:
            event: The hook event name (e.g., ``'before_tool_call'``).
            handler: The callback function to invoke.
            kind: Dispatch mode (notify, mutate, reduce, guard).
            priority: Higher priority handlers are called first.
            matcher: Optional regex to filter events by contextual value
                (e.g., ``'Bash|Edit'`` to match specific tool names).
        """
        self._pending_hooks.append(
            HookRecord(
                plugin_id=self._plugin_id,
                event=event,
                kind=kind.value,
                priority=priority,
                handler=handler,
                matcher=matcher,
            )
        )

    def register_config_schema(self, schema_cls: type[BaseModel]) -> None:
        """Register a Pydantic model as this plugin's config schema."""
        self._pending_config_schemas = [
            r for r in self._pending_config_schemas if r.plugin_id != self._plugin_id
        ]
        self._pending_config_schemas.append(
            ConfigSchemaRecord(plugin_id=self._plugin_id, schema_cls=schema_cls)
        )

    def register_status_provider(
        self,
        name: str,
        provider: Any,
        *,
        priority: int = 0,
    ) -> None:
        """Register a status bar segment provider.

        The *provider* callable receives a
        :class:`~aeloon.plugins._sdk.types.StatusContext` and returns a
        ``str``, :class:`StatusSegment`, or list of ``StatusSegment``.
        It **must** be synchronous — it is called during prompt_toolkit
        rendering.
        """
        self._pending_status_providers = [
            r for r in self._pending_status_providers if r.name != name
        ]
        self._pending_status_providers.append(
            StatusProviderRecord(
                plugin_id=self._plugin_id,
                name=name,
                provider=provider,
                priority=priority,
            )
        )

    # ------------------------------------------------------------------
    # Service control (async — used during activate phase)
    # ------------------------------------------------------------------

    async def start_service(
        self,
        name: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """Start a registered service via the :class:`ServiceSupervisor`."""
        full_id = f"{self._plugin_id}.{name}"
        record = self._registry.services.get(full_id)
        if not record:
            logger.warning("Service '{}' not found in registry", full_id)
            return
        # Delegate to manager's supervisor (attached later during boot)
        if hasattr(self, "_supervisor"):
            cfg = dict(self._config)
            if config_overrides:
                cfg.update(config_overrides)
            await self._supervisor.start_service(record, self._runtime, cfg)

    async def stop_service(self, name: str) -> None:
        """Stop a running service."""
        full_id = f"{self._plugin_id}.{name}"
        if hasattr(self, "_supervisor"):
            await self._supervisor.stop_service(full_id)

    def list_service_status(self) -> dict[str, ServiceStatus]:
        """Return status of all services owned by this plugin."""
        result: dict[str, ServiceStatus] = {}
        for full_id, rec in self._registry.services.items():
            if rec.plugin_id == self._plugin_id:
                result[rec.name] = rec.status
        return result

    def get_plugin(self, plugin_id: str) -> Any:
        """Reserved inter-plugin communication stub.  Returns ``None`` in v0.1."""
        return None

    # ------------------------------------------------------------------
    # Internal methods (called by PluginManager)
    # ------------------------------------------------------------------

    def _commit(self) -> None:
        """Commit all pending records to the registry."""
        self._registry.commit_plugin(
            self._plugin_id,
            commands=self._pending_commands,
            tools=self._pending_tools,
            services=self._pending_services,
            middlewares=self._pending_middlewares,
            command_middlewares=self._pending_command_middlewares,
            cli=self._pending_cli,
            hooks=self._pending_hooks,
            config_schemas=self._pending_config_schemas,
            status_providers=self._pending_status_providers,
        )
        self._clear_pending()

    def _rollback(self) -> None:
        """Discard all pending records without writing to the registry."""
        self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending_commands.clear()
        self._pending_tools.clear()
        self._pending_services.clear()
        self._pending_middlewares.clear()
        self._pending_command_middlewares.clear()
        self._pending_cli.clear()
        self._pending_cli_specs.clear()
        self._pending_hooks.clear()
        self._pending_config_schemas.clear()
        self._pending_status_providers.clear()
