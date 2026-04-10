"""Built-in informational slash commands."""

from __future__ import annotations

from aeloon.cli.registry import CommandSpec
from aeloon.core.agent.commands import BuiltinHandlerMap, CommandEnv
from aeloon.core.bus.events import InboundMessage, OutboundMessage

SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(name="status", help="Show channel status", slash_path=("status",)),
    CommandSpec(name="help", help="Show available commands", slash_path=("help",)),
)


async def handle_help(env: CommandEnv, msg: InboundMessage, _args_str: str) -> OutboundMessage:
    """Render built-in and plugin slash command help."""
    plugin_catalog = env.plugin_catalog_fn()
    lines = [
        "# ♥️ aeloon",
        "",
        "## Commands",
        "",
    ]
    lines.extend(env.builtin_catalog.render_help_lines())

    plugin_lines = plugin_catalog.render_help_lines()
    if plugin_lines:
        lines.extend(["", "## Plugins", ""])
        lines.extend(plugin_lines)
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content="\n".join(lines),
    )


async def handle_status(env: CommandEnv, msg: InboundMessage, _args_str: str) -> OutboundMessage:
    """Show runtime, channel, and plugin state."""
    state_icons = {
        "pending": "⏳",
        "starting": "🔄",
        "running": "✅",
        "failed": "❌",
        "stopped": "⏹️",
    }

    lines: list[str] = ["Runtime Status:"]
    session_key = f"{msg.channel}:{msg.chat_id}"
    try:
        session = env.sessions.get_or_create(session_key)
        estimated, _source = env.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        estimated = 0
    context_total = max(0, int(env.context_window_tokens))
    ratio = (estimated / context_total * 100) if context_total > 0 else 0.0
    lines.append(f"Model: {env.model}")
    lines.append(f"Context: {estimated}/{context_total} ({ratio:.0f}%)")

    lines.append("")
    lines.append("Channel Status:")
    if env.channel_manager is None:
        lines.append("Channel status is not available (no channel manager).")
    else:
        status = env.channel_manager.get_status()
        if not status:
            lines.append("No channels configured.")
        else:
            for name, info in status.items():
                state = info["state"]
                icon = state_icons.get(state, "❓")
                line = f"{icon} {info['display_name']} ({name}): {state}"
                if "error" in info:
                    line += f" — {info['error']}"
                lines.append(line)

    pm = env.plugin_manager
    if pm:
        service_lines: list[str] = []
        for full_id, service in sorted(pm.registry.services.items()):
            service_lines.append(f"- {full_id}: {service.status.value}")

        if service_lines:
            lines.append("")
            lines.append("Plugin Status:")
            lines.extend(service_lines)

    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content="\n".join(lines),
    )


HANDLERS: BuiltinHandlerMap = {
    "help": handle_help,
    "status": handle_status,
}
