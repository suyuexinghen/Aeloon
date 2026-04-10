"""PluginCreatorPlugin — meta-plugin entry point for creating other plugins.

Registers the ``/pc`` command group through one Plugin SDK entrypoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aeloon.plugins._sdk import CLICommandGroup, CLIMessageOptionSpec, CommandContext, Plugin

from .pipeline import PluginCreatorPipeline

if TYPE_CHECKING:
    from aeloon.plugins._sdk.api import PluginAPI


class PluginCreatorPlugin(Plugin):
    """PluginCreator — intelligent plugin development workflow agent."""

    def __init__(self) -> None:
        self._api: PluginAPI | None = None
        self._pipeline: PluginCreatorPipeline | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self, api: PluginAPI) -> None:
        cli = CLICommandGroup(name="pc", help="PluginCreator — plugin development workflow")
        commands = (
            cli.command(
                "plan",
                help="Create a plugin plan from a requirement description.",
                args_template="plan {message}",
                message=CLIMessageOptionSpec(help="Plugin requirement description"),
            ),
            cli.command(
                "status",
                help="Show the latest plugin planning status.",
                args_template="status",
                message=None,
            ),
            cli.command(
                "history",
                help="Show recent plugin planning history.",
                args_template="history",
                message=None,
            ),
        )
        api.register_cli(
            "pc",
            commands=commands,
            handler=self._handle_command,
            description="Create and manage plugin plans",
        )

        from .config import PluginCreatorConfig

        api.register_config_schema(PluginCreatorConfig)
        self._api = api

    async def activate(self, api: PluginAPI) -> None:
        api.runtime.storage_path.mkdir(parents=True, exist_ok=True)

    async def deactivate(self) -> None:
        self._pipeline = None

    # ------------------------------------------------------------------
    # Command handler
    # ------------------------------------------------------------------

    async def _handle_command(self, ctx: CommandContext, args: str) -> str | None:
        """Route ``/pc`` subcommands or run a planning task."""
        from .pipeline import get_help_text

        parts = args.strip().split(None, 1)
        subcommand = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if not args.strip() or subcommand in ("help", "--help", "-h"):
            return get_help_text()

        if subcommand == "status":
            return self._get_or_create_pipeline().get_status()

        if subcommand == "history":
            return self._get_or_create_pipeline().get_history()

        if subcommand == "plan":
            pipeline = self._get_or_create_pipeline()
            output, _pkg = await pipeline.plan(rest, project_id=ctx.session_key)
            if output:
                await ctx.reply(output)
            return None

        # Default: treat entire args as plan requirement
        pipeline = self._get_or_create_pipeline()
        output, _pkg = await pipeline.plan(
            args,
            project_id=ctx.session_key,
        )
        if output:
            await ctx.reply(output)
        return None

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def _get_or_create_pipeline(self) -> PluginCreatorPipeline:
        if self._pipeline is None:
            assert self._api is not None
            self._pipeline = PluginCreatorPipeline(
                runtime=self._api.runtime,
                storage_dir=str(self._api.runtime.storage_path),
            )
        return self._pipeline
