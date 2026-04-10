"""Built-in plugin administration slash commands."""

from __future__ import annotations

from pathlib import Path

from aeloon.cli.registry import CommandSpec
from aeloon.core.agent.commands import BuiltinHandlerMap, CommandEnv
from aeloon.core.bus.events import InboundMessage, OutboundMessage

SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="plugin",
        help="Manage plugins.",
        cli_path=("plugin",),
        slash_path=("plugin",),
        slash_paths=(
            ("plugin", "list"),
            ("plugin", "error"),
            ("plugin", "error", "<name>"),
            ("plugin", "install"),
            ("plugin", "install", "<archive-path>"),
            ("plugin", "remove"),
            ("plugin", "remove", "<name>"),
            ("plugin", "activate"),
            ("plugin", "activate", "<name>"),
            ("plugin", "deactivate"),
            ("plugin", "deactivate", "<name>"),
        ),
    ),
)


def _plugin_list(env: CommandEnv, msg: InboundMessage) -> OutboundMessage:
    """List all plugins with status."""
    from aeloon.plugins._sdk.admin import format_runtime_plugin_list

    pm = env.plugin_manager
    content = (
        format_runtime_plugin_list(pm)
        if pm
        else "Plugins:\n──────────────────────────────────────────────────\n  (plugin manager not available)\n"
    )
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def _plugin_install(
    env: CommandEnv,
    msg: InboundMessage,
    archive_path: str,
) -> OutboundMessage:
    """Install a plugin from an archive."""
    from aeloon.core.config.loader import get_aeloon_home
    from aeloon.plugins._sdk.admin import install_plugin_archive

    path = Path(archive_path).expanduser().resolve()
    workspace_dir = get_aeloon_home() / "plugins"
    pm = env.plugin_manager
    state_store = pm._state_store if pm and pm._state_store else None
    if state_store is None:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Plugin manager not available.",
        )

    result = install_plugin_archive(
        archive=path,
        workspace_dir=workspace_dir,
        state_store=state_store,
    )
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result.message)


def _plugin_error(env: CommandEnv, msg: InboundMessage, name: str) -> OutboundMessage:
    """Show error details for broken plugins."""
    from aeloon.plugins._sdk.admin import format_plugin_errors

    pm = env.plugin_manager
    if not pm:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Plugin manager not available.",
        )
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=format_plugin_errors(pm, name),
    )


async def _plugin_remove(env: CommandEnv, msg: InboundMessage, name: str) -> OutboundMessage:
    """Remove a workspace-installed plugin."""
    from aeloon.core.config.loader import get_aeloon_home
    from aeloon.plugins._sdk.admin import remove_workspace_plugin

    workspace_dir = get_aeloon_home() / "plugins"
    pm = env.plugin_manager
    state_store = pm._state_store if pm and pm._state_store else None
    if state_store is None:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Plugin manager not available.",
        )
    result = remove_workspace_plugin(
        name=name,
        workspace_dir=workspace_dir,
        state_store=state_store,
    )
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result.message)


def _plugin_activate(env: CommandEnv, msg: InboundMessage, name: str) -> OutboundMessage:
    """Activate a plugin."""
    from aeloon.plugins._sdk.admin import set_plugin_enabled

    pm = env.plugin_manager
    result = set_plugin_enabled(
        name=name,
        enabled=True,
        state_store=pm._state_store if pm else None,
    )
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result.message)


def _plugin_deactivate(env: CommandEnv, msg: InboundMessage, name: str) -> OutboundMessage:
    """Deactivate a plugin."""
    from aeloon.plugins._sdk.admin import set_plugin_enabled

    pm = env.plugin_manager
    result = set_plugin_enabled(
        name=name,
        enabled=False,
        state_store=pm._state_store if pm else None,
    )
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result.message)


async def handle_plugin(env: CommandEnv, msg: InboundMessage, args_str: str) -> OutboundMessage:
    """Handle `/plugin` slash command."""
    args = args_str.split() if args_str else []
    sub = args[0] if args else "list"
    rest = args[1:] if len(args) > 1 else []

    if sub == "list":
        return _plugin_list(env, msg)
    if sub == "install" and rest:
        return await _plugin_install(env, msg, " ".join(rest))
    if sub == "error":
        name = rest[0] if rest else ""
        return _plugin_error(env, msg, name)
    if sub == "remove" and rest:
        return await _plugin_remove(env, msg, rest[0])
    if sub == "activate" and rest:
        return _plugin_activate(env, msg, rest[0])
    if sub == "deactivate" and rest:
        return _plugin_deactivate(env, msg, rest[0])

    usage = (
        "Usage:\n"
        "- `/plugin list` — List installed plugins\n"
        "- `/plugin install <archive-path>` — Install a plugin\n"
        "- `/plugin error [name]` — Show error details\n"
        "- `/plugin remove <name>` — Remove a plugin\n"
        "- `/plugin activate <name>` — Activate a plugin\n"
        "- `/plugin deactivate <name>` — Deactivate a plugin"
    )
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=usage)


HANDLERS: BuiltinHandlerMap = {
    "plugin": handle_plugin,
}
