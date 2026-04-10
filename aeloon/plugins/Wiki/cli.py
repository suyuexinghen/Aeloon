"""Declarative CLI metadata for the Wiki plugin."""

from __future__ import annotations

from aeloon.plugins._sdk import CLICommandGroup, CLICommandSpec, CLIFlagSpec, CLIMessageOptionSpec


def wiki_cli_specs(plugin_command: str = "wiki") -> tuple[CLICommandSpec, ...]:
    """Return slash/CLI metadata for the Wiki plugin command group."""
    cli = CLICommandGroup(name="wiki", help="Local wiki workflows", plugin_command=plugin_command)
    return (
        cli.command(
            "init",
            help="Initialize the knowledge base root.",
            args_template="init {message}",
            message=CLIMessageOptionSpec(
                help="Optional wiki root path",
                required=False,
                default="",
                parameter_kind="argument",
            ),
            slash_paths=(("wiki", "init", "<path>"),),
        ),
        cli.command(
            "status", help="Show wiki repository status.", args_template="status", message=None
        ),
        cli.command(
            "add",
            help="Ingest and digest one local path or free-form text.",
            args_template="add {message}",
            message=CLIMessageOptionSpec(
                help="Path or text to ingest",
                parameter_kind="argument",
            ),
            slash_paths=(("wiki", "add", "<path-or-text>"),),
        ),
        cli.command(
            "digest", help="Digest pending wiki sources.", args_template="digest", message=None
        ),
        cli.command(
            "list",
            help="List tracked wiki sources and entries.",
            args_template="list",
            message=None,
        ),
        cli.command(
            "get",
            help="Show one wiki entry.",
            args_template="get {message}",
            message=CLIMessageOptionSpec(
                help="Wiki entry ID or path",
                parameter_kind="argument",
            ),
            slash_paths=(("wiki", "get", "<entry>"),),
        ),
        cli.command(
            "map",
            help="Render the wiki relation map.",
            args_template="map {message}",
            message=CLIMessageOptionSpec(
                help="Optional wiki entry",
                required=False,
                default="",
                parameter_kind="argument",
            ),
            slash_paths=(("wiki", "map", "<entry>"),),
        ),
        cli.command(
            "jobs",
            help="Show the current wiki background task.",
            args_template="jobs",
            message=None,
        ),
        cli.command(
            "use",
            help="Control wiki grounding mode during chat.",
            args_template="use {message}",
            message=CLIMessageOptionSpec(
                help="Mode: off, prefer-local, local-only, or status",
                parameter_kind="argument",
            ),
            slash_paths=(
                ("wiki", "use", "off"),
                ("wiki", "use", "prefer-local"),
                ("wiki", "use", "local-only"),
                ("wiki", "use", "status"),
            ),
        ),
        cli.command(
            "attach",
            help="Toggle wiki auto-attachment for future media.",
            args_template="attach {message}",
            message=CLIMessageOptionSpec(
                help="Mode: on, off, or status",
                parameter_kind="argument",
            ),
            slash_paths=(
                ("wiki", "attach", "on"),
                ("wiki", "attach", "off"),
                ("wiki", "attach", "status"),
            ),
        ),
        cli.command(
            "remove",
            help="Remove the current wiki repository.",
            args_template="remove{confirm}",
            message=None,
            flags=(
                CLIFlagSpec(
                    name="confirm",
                    flags=("--confirm",),
                    help="Delete the wiki repository",
                    value_when_true=" --confirm",
                ),
            ),
        ),
    )
