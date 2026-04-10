"""Registration record dataclasses, context types, and type aliases.

All records are plain dataclasses so they can be stored in the
:class:`~aeloon.plugins._sdk.registry.PluginRegistry` without coupling
to framework internals.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from aeloon.plugins._sdk.base import PluginService, ServiceStatus

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CommandHandler = Callable[["CommandContext", str], Awaitable[str | None]]
"""Async callback invoked when a plugin command is triggered.

Signature: ``async def handler(ctx: CommandContext, args: str) -> str | None``
"""

CLIBuilder = Callable[[Any], None]
"""Callable that receives a :class:`typer.Typer` and attaches sub-commands."""


@dataclasses.dataclass(frozen=True)
class CLIMessageOptionSpec:
    """Declarative metadata for a single message-like CLI option."""

    flags: tuple[str, ...] = ("--message", "-m")
    help: str = ""
    required: bool = True
    default: str = ""
    parameter_kind: Literal["option", "argument"] = "option"


@dataclasses.dataclass(frozen=True)
class CLIFlagSpec:
    """Declarative metadata for a boolean CLI flag."""

    name: str
    flags: tuple[str, ...]
    help: str = ""
    default: bool = False
    value_when_true: str = ""
    value_when_false: str = ""


@dataclasses.dataclass(frozen=True)
class CLICommandSpec:
    """Declarative metadata for a simple plugin CLI command."""

    group_name: str
    command_name: str
    help: str
    plugin_command: str
    group_help: str = ""
    args_template: str = "{message}"
    message: CLIMessageOptionSpec | None = dataclasses.field(default_factory=CLIMessageOptionSpec)
    flags: tuple[CLIFlagSpec, ...] = ()
    slash_paths: tuple[tuple[str, ...], ...] = ()

    @property
    def slash_path(self) -> tuple[str, ...]:
        """Return the slash-visible path for this command."""
        return (self.group_name, self.command_name)

    def iter_slash_paths(self) -> tuple[tuple[str, ...], ...]:
        """Return all slash-visible paths for this command."""
        return (self.slash_path, *self.slash_paths)


@dataclasses.dataclass(frozen=True)
class CLICommandGroup:
    """Shared metadata for a plugin-owned CLI command group."""

    name: str
    help: str
    plugin_command: str | None = None

    def command(
        self,
        command_name: str,
        help: str,
        *,
        args_template: str,
        message: CLIMessageOptionSpec | None = None,
        flags: tuple[CLIFlagSpec, ...] = (),
        slash_paths: tuple[tuple[str, ...], ...] = (),
    ) -> CLICommandSpec:
        """Create one CLI command spec under this group."""
        return CLICommandSpec(
            group_name=self.name,
            command_name=command_name,
            help=help,
            plugin_command=self.plugin_command or self.name,
            group_help=self.help,
            args_template=args_template,
            message=message,
            flags=flags,
            slash_paths=slash_paths,
        )


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CommandContext:
    """Immutable context passed to plugin command handlers."""

    session_key: str
    channel: str
    reply: Callable[[str], Awaitable[None]]
    send_progress: Callable[..., Awaitable[None]]
    plugin_config: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class CommandExecutionContext:
    """Immutable execution context passed to command middlewares."""

    session_key: str
    channel: str
    chat_id: str
    sender_id: str
    metadata: Mapping[str, Any]
    is_builtin: bool
    plugin_id: str | None = None
    plugin_config: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    reply: Callable[[str], Awaitable[None]] | None = None
    send_progress: Callable[..., Awaitable[None]] | None = None


# ---------------------------------------------------------------------------
# Service policy
# ---------------------------------------------------------------------------


class ServicePolicy(BaseModel):
    """Restart and timeout policy for a managed :class:`PluginService`."""

    restart_policy: Literal["never", "on-failure", "always"] = "on-failure"
    max_restarts: int = 3
    restart_delay_seconds: float = 5.0
    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0


class CommandMiddleware(Protocol):
    """Protocol for dispatcher-level command middlewares."""

    async def before(self, cmd: str, args: str, ctx: CommandExecutionContext) -> None:
        """Run before the command handler."""

    async def after(self, cmd: str, result: Any, ctx: CommandExecutionContext) -> None:
        """Run after the command handler returns."""


# ---------------------------------------------------------------------------
# Registration records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CommandRecord:
    """Registry entry for a plugin command."""

    plugin_id: str
    name: str
    handler: CommandHandler
    description: str = ""


@dataclasses.dataclass
class ToolRecord:
    """Registry entry for a plugin tool."""

    plugin_id: str
    name: str
    tool: Any  # Tool instance (aeloon.agent.tools.base.Tool)


@dataclasses.dataclass
class ServiceRecord:
    """Registry entry for a plugin service."""

    plugin_id: str
    name: str
    full_id: str  # qualified: "plugin_id.name"
    service_cls: type[PluginService]
    policy: ServicePolicy = dataclasses.field(default_factory=ServicePolicy)
    status: ServiceStatus = ServiceStatus.STOPPED
    restart_count: int = 0


@dataclasses.dataclass
class MiddlewareRecord:
    """Registry entry for a plugin middleware."""

    plugin_id: str
    name: str
    middleware: Any  # BaseAgentMiddleware instance


@dataclasses.dataclass
class CommandMiddlewareRecord:
    """Registry entry for a dispatcher-level command middleware."""

    plugin_id: str
    name: str
    middleware: CommandMiddleware


@dataclasses.dataclass
class CLIRecord:
    """Registry entry for a plugin CLI sub-command builder."""

    plugin_id: str
    name: str
    builder: CLIBuilder
    commands: tuple[CLICommandSpec, ...] = ()


@dataclasses.dataclass
class HookRecord:
    """Registry entry for a plugin hook handler."""

    plugin_id: str
    event: str
    kind: str  # HookType value
    priority: int
    handler: Callable[..., Any]
    matcher: str | None = None  # regex for event filtering (None = match all)


@dataclasses.dataclass
class HookDecision:
    """Return type for GUARD-mode hooks — can allow, deny, or modify values.

    When returned from a guard hook handler:
    * ``allow=True`` (default) — the action proceeds.
    * ``allow=False`` — the action is **blocked**; ``reason`` is surfaced
      to the agent as an error message.
    * ``modified_value`` — if set, replaces the guarded value (e.g. tool
      call arguments) before the action proceeds.
    """

    allow: bool = True
    reason: str = ""
    modified_value: Any = None


@dataclasses.dataclass
class ConfigSchemaRecord:
    """Registry entry for a plugin config schema class."""

    plugin_id: str
    schema_cls: type[BaseModel]


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StatusContext:
    """Immutable context passed to status providers."""

    session_key: str
    channel: str
    model: str
    context_tokens_used: int
    context_tokens_total: int
    terminal_width: int


@dataclasses.dataclass
class StatusSegment:
    """A single styled segment for the status bar."""

    text: str
    style: str = ""  # prompt_toolkit style string (e.g. "bold ansired")
    priority: int = 0  # higher = displayed first (leftmost)


@dataclasses.dataclass
class StatusProviderRecord:
    """Registry entry for a plugin status provider."""

    plugin_id: str
    name: str
    provider: Callable[[StatusContext], str | StatusSegment | list[StatusSegment]]
    priority: int = 0  # plugin-level ordering
