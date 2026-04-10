"""Shared plugin and config bootstrap helpers for CLI flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aeloon.cli.app import console
from aeloon.core.config.schema import Config


def merge_missing_defaults(existing: Any, defaults: Any) -> Any:
    """Recursively fill in missing values from defaults without overwriting user config."""
    if not isinstance(existing, dict) or not isinstance(defaults, dict):
        return existing
    merged = dict(existing)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
        else:
            merged[key] = merge_missing_defaults(merged[key], value)
    return merged


def onboard_plugins(config_path: Path) -> None:
    """Inject default config for all discovered channels."""
    from aeloon.channels.registry import discover_all

    all_channels = discover_all()
    if not all_channels:
        return

    with open(config_path, encoding="utf-8") as handle:
        data = json.load(handle)

    channels = data.setdefault("channels", {})
    for name, cls in all_channels.items():
        if name not in channels:
            channels[name] = cls.default_config()
        else:
            channels[name] = merge_missing_defaults(channels[name], cls.default_config())

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


async def boot_plugins(agent_loop: Any, config: Config, *, quiet: bool = False) -> Any:
    """Create and boot the PluginManager. Returns the manager or None."""
    from aeloon.plugins._sdk.discovery import PluginDiscovery
    from aeloon.plugins._sdk.hooks import HookDispatcher
    from aeloon.plugins._sdk.loader import PluginLoader
    from aeloon.plugins._sdk.manager import PluginManager
    from aeloon.plugins._sdk.registry import PluginRegistry
    from aeloon.plugins._sdk.state_store import PluginStateStore

    registry = PluginRegistry()
    discovery = PluginDiscovery(
        bundled_dir=Path(__file__).resolve().parents[2] / "plugins",
        workspace_dir=Path.home() / ".aeloon" / "plugins",
    )
    manager = PluginManager(
        registry=registry,
        discovery=discovery,
        loader=PluginLoader(),
        hook_dispatcher=HookDispatcher(registry.hooks_for_event),
        agent_loop=agent_loop,
        plugin_config=config.plugins if hasattr(config, "plugins") else {},
        storage_base=config.workspace_path / ".aeloon" / "plugin_storage",
        state_store=PluginStateStore(Path.home() / ".aeloon" / "plugin_state.json"),
    )
    result = await manager.boot()
    if not quiet and result.loaded:
        console.print(f"[green]✓[/green] Plugins loaded: {', '.join(result.loaded)}")
    if not quiet and result.failed:
        console.print(f"[red]✗[/red] Plugins failed: {', '.join(result.failed)}")
    for tool_record in registry.tools.values():
        agent_loop.tools.register(tool_record.tool)
    if config.hooks:
        from aeloon.plugins._sdk.config_hooks import ConfigHookAdapter

        ConfigHookAdapter(registry).load_from_config(config.hooks)
    return manager
