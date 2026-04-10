"""Declarative CLI metadata for the ACP bridge plugin."""

from __future__ import annotations

from aeloon.plugins._sdk import CLICommandGroup, CLICommandSpec, CLIMessageOptionSpec


def acp_cli_specs(plugin_command: str = "acp") -> tuple[CLICommandSpec, ...]:
    """Return slash/CLI metadata for the ACP bridge command group."""
    cli = CLICommandGroup(name="acp", help="ACP bridge commands", plugin_command=plugin_command)
    return (
        cli.command(
            "connect",
            help="Connect to an ACP backend.",
            args_template="connect {message}",
            message=CLIMessageOptionSpec(
                help="Optional ACP profile name",
                required=False,
                default="",
                parameter_kind="argument",
            ),
            slash_paths=(("acp", "connect", "<profile>"),),
        ),
        cli.command(
            "chat",
            help="Send a message to the connected ACP agent.",
            args_template="chat {message}",
            message=CLIMessageOptionSpec(
                help="Message to delegate",
                parameter_kind="argument",
            ),
            slash_paths=(
                ("acp", "chat", "<message>"),
                ("acp", "ask", "<message>"),
                ("acp", "delegate", "<message>"),
            ),
        ),
        cli.command(
            "disconnect",
            help="Disconnect from the current ACP backend.",
            args_template="disconnect",
            message=None,
        ),
        cli.command(
            "status",
            help="Show ACP connection state.",
            args_template="status",
            message=None,
        ),
        cli.command(
            "help",
            help="Show ACP bridge help.",
            args_template="help",
            message=None,
        ),
    )
