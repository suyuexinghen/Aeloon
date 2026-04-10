"""Aeloon Plugin SDK — plugin lifecycle management framework."""

from aeloon.plugins._sdk.acp import (
    ACPClient,
    ACPError,
    ACPLayer,
    ACPTransport,
    BackendProfile,
    ConnectionState,
    DelegateResult,
    SessionInfo,
    SessionMap,
)
from aeloon.plugins._sdk.base import Plugin, PluginService, ServiceStatus
from aeloon.plugins._sdk.config_hooks import (
    CommandHookHandler,
    ConfigHookAdapter,
    HookHandler,
    HttpHookHandler,
)
from aeloon.plugins._sdk.hooks import HookEvent, HookType
from aeloon.plugins._sdk.manifest import PluginManifest, PluginProvides, PluginRequires
from aeloon.plugins._sdk.runtime import LegacyRuntimeAdapter, PluginLLMProxy, PluginRuntime
from aeloon.plugins._sdk.types import (
    CLICommandGroup,
    CLICommandSpec,
    CLIFlagSpec,
    CLIMessageOptionSpec,
    CommandContext,
    CommandExecutionContext,
    CommandMiddleware,
    HookDecision,
    ServicePolicy,
    StatusContext,
    StatusSegment,
)

__all__ = [
    # ACP integration
    "ACPClient",
    "ACPError",
    "ACPLayer",
    "ACPTransport",
    "BackendProfile",
    "CLICommandGroup",
    "CLICommandSpec",
    "CLIFlagSpec",
    "CLIMessageOptionSpec",
    "CommandExecutionContext",
    "CommandMiddleware",
    "CommandContext",
    "CommandHookHandler",
    "ConfigHookAdapter",
    "ConnectionState",
    "DelegateResult",
    "HookDecision",
    "HookEvent",
    "HookHandler",
    "HookType",
    "HttpHookHandler",
    "LegacyRuntimeAdapter",
    "Plugin",
    "PluginLLMProxy",
    "PluginManifest",
    "PluginProvides",
    "PluginRequires",
    "PluginRuntime",
    "PluginService",
    "ServicePolicy",
    "ServiceStatus",
    "SessionInfo",
    "SessionMap",
    "StatusContext",
    "StatusSegment",
]
