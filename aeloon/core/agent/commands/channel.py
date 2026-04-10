"""Built-in channel management slash commands."""

from __future__ import annotations

from aeloon.cli.registry import CommandSpec
from aeloon.core.agent.commands import BuiltinHandlerMap, CommandEnv
from aeloon.core.bus.events import InboundMessage, OutboundMessage

SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="channel",
        help="Manage one channel.",
        cli_path=("channel",),
        slash_path=("channel",),
        slash_paths=(
            ("channel", "list"),
            ("channel", "status"),
            ("channel", "status", "<name>"),
            ("channel", "wechat"),
            ("channel", "wechat", "login"),
            ("channel", "wechat", "logout"),
            ("channel", "wechat", "status"),
            ("channel", "feishu"),
            ("channel", "feishu", "login"),
            ("channel", "feishu", "logout"),
            ("channel", "feishu", "status"),
            ("channel", "whatsapp"),
            ("channel", "whatsapp", "login"),
        ),
    ),
    CommandSpec(
        name="wechat",
        help="WeChat login management",
        cli_path=("wechat",),
        slash_path=("wechat",),
        slash_paths=(("wechat", "login"), ("wechat", "logout"), ("wechat", "status")),
    ),
    CommandSpec(
        name="feishu",
        help="Feishu login management",
        cli_path=("feishu",),
        slash_path=("feishu",),
        slash_paths=(("feishu", "login"), ("feishu", "logout"), ("feishu", "status")),
    ),
    CommandSpec(
        name="whatsapp",
        help="WhatsApp login management",
        cli_path=("whatsapp",),
        slash_paths=(("whatsapp", "login"),),
    ),
)


def _channel_enabled(env: CommandEnv, name: str) -> bool:
    """Return whether one channel is enabled in runtime config."""
    config = env.channels_config
    if config is None:
        return False
    section = getattr(config, name, None)
    if section is None:
        return False
    if isinstance(section, dict):
        return section.get("enabled", False)
    return getattr(section, "enabled", False)


async def handle_wechat(env: CommandEnv, msg: InboundMessage, args_str: str) -> OutboundMessage:
    """Handle `/wechat` slash command."""
    return await env.channel_auth.handle_wechat_command(
        msg,
        args_str.split() if args_str else [],
        env.as_bus_namespace(),
    )


async def handle_feishu(env: CommandEnv, msg: InboundMessage, args_str: str) -> OutboundMessage:
    """Handle `/feishu` slash command."""
    return await env.channel_auth.handle_feishu_command(msg, args_str.split() if args_str else [])


async def handle_channel(env: CommandEnv, msg: InboundMessage, args_str: str) -> OutboundMessage:
    """Handle `/channel` slash command."""
    from aeloon.channels.registry import discover_all

    args = args_str.split() if args_str else []
    if not args:
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Usage: /channel list | /channel status [name] | /channel <wechat|feishu|whatsapp> <action>",
        )

    subcommand = args[0].lower()
    if subcommand == "list":
        lines = ["# Channels", ""]
        for name, cls in sorted(discover_all().items()):
            enabled = _channel_enabled(env, name)
            lines.append(
                f"- `{name}` ({cls.display_name}) — {'enabled' if enabled else 'disabled'}"
            )
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="\n".join(lines),
        )

    if subcommand == "status":
        target = args[1].lower() if len(args) > 1 else None
        lines = ["# Channel Status", ""]
        for name, cls in sorted(discover_all().items()):
            if target and name != target:
                continue
            enabled = _channel_enabled(env, name)
            lines.append(
                f"- `{name}` ({cls.display_name}) — {'enabled' if enabled else 'disabled'}"
            )
        if len(lines) == 2:
            lines.append(f"Unknown channel: {target}")
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="\n".join(lines),
        )

    channel_name = subcommand
    remainder = " ".join(args[1:])
    if channel_name == "wechat":
        return await handle_wechat(env, msg, remainder)
    if channel_name == "feishu":
        return await handle_feishu(env, msg, remainder)
    if channel_name == "whatsapp":
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="Use `aeloon channel whatsapp login` for WhatsApp login.",
        )

    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=f"Unknown channel: {channel_name}. Use /channel list to see available channels.",
    )


HANDLERS: BuiltinHandlerMap = {
    "channel": handle_channel,
    "feishu": handle_feishu,
    "wechat": handle_wechat,
}
