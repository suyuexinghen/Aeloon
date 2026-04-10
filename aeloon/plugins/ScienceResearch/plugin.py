"""SciencePlugin — first Task Plugin, wrapping the AI4S science pipeline.

Registers the ``/sr`` command group through one Plugin SDK entrypoint.
Core DAG / validator / storage logic is unchanged; only the integration
surface (config, routing, runtime access) moves behind plugin APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aeloon.plugins._sdk import CLICommandGroup, CLIMessageOptionSpec, CommandContext, Plugin

from .pipeline import SciencePipeline

if TYPE_CHECKING:
    from aeloon.plugins._sdk.api import PluginAPI


class SciencePlugin(Plugin):
    """AI4S Science Agent — Task Plugin entry point."""

    def __init__(self) -> None:
        self._api: PluginAPI | None = None
        self._pipeline: SciencePipeline | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self, api: PluginAPI) -> None:
        cli = CLICommandGroup(name="sr", help="AI4S Science Agent")
        commands = (
            cli.command(
                "run",
                help="Run an AI4S science research task.",
                args_template="run {message}",
                message=CLIMessageOptionSpec(help="Science research task"),
            ),
            cli.command(
                "status",
                help="Show the latest science task status.",
                args_template="status",
                message=None,
            ),
            cli.command(
                "history",
                help="Show recent science task history.",
                args_template="history",
                message=None,
            ),
        )
        api.register_cli(
            "sr",
            commands=commands,
            handler=self._handle_command,
            description="Run an AI4S science research task",
        )

        from .config import ScienceConfig

        api.register_config_schema(ScienceConfig)
        self._api = api

    async def activate(self, api: PluginAPI) -> None:
        # Ensure plugin storage directory exists
        api.runtime.storage_path.mkdir(parents=True, exist_ok=True)

        # Register ArxivTool
        from .config import ArxivConfig
        from .tools.arxiv import ArxivTool

        arxiv_config = ArxivConfig()
        api.register_tool(ArxivTool(config=arxiv_config))

    async def deactivate(self) -> None:
        self._pipeline = None

    # ------------------------------------------------------------------
    # Command handler
    # ------------------------------------------------------------------

    async def _handle_command(self, ctx: CommandContext, args: str) -> str | None:
        """Route ``/sr`` subcommands or run a research task."""
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

        if subcommand == "run":
            if not rest:
                return get_help_text()
            args = rest

        # Run a science task with the full query
        pipeline = self._get_or_create_pipeline()
        output, _task = await pipeline.run(
            query=args,
            on_progress=ctx.send_progress,
            session_id=ctx.session_key,
        )
        # Send directly via reply callback so large outputs reach the channel
        # reliably (avoids the dispatcher return-value path which may silently
        # drop oversized messages).
        if output:
            await ctx.reply(output)
        return None

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def _get_or_create_pipeline(self) -> SciencePipeline:
        if self._pipeline is None:
            assert self._api is not None

            self._pipeline = SciencePipeline(
                runtime=self._api.runtime,
                storage_dir=str(self._api.runtime.storage_path),
            )
        return self._pipeline
