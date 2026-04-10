"""Wiki plugin entry point."""

from __future__ import annotations

import asyncio
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aeloon.core.bus.events import OutboundMessage
from aeloon.plugins._sdk import CommandContext, HookEvent, Plugin

from .cli import wiki_cli_specs
from .config import WikiConfig
from .middleware.query_context import WikiQueryMiddleware
from .services.digest_service import DigestService
from .services.ingest_service import IngestService
from .services.manifest_service import ManifestService
from .services.query_service import QueryService
from .services.repo_service import RepoService
from .services.usage_mode import SessionToggleStore, UsageModeStore

if TYPE_CHECKING:
    from aeloon.plugins._sdk.api import PluginAPI
    from aeloon.plugins._sdk.runtime import PluginRuntime


@dataclass(slots=True)
class _BackgroundJob:
    args: str
    started_at: float
    task: asyncio.Task[None]


class WikiPlugin(Plugin):
    """Hybrid plugin for wiki ingestion and query grounding."""

    def __init__(self) -> None:
        self._config = WikiConfig()
        self._runtime: PluginRuntime | None = None
        self._repo_service: RepoService | None = None
        self._manifest_service: ManifestService | None = None
        self._ingest_service: IngestService | None = None
        self._digest_service: DigestService | None = None
        self._query_service: QueryService | None = None
        self._usage_modes = UsageModeStore()
        self._attachment_auto_add = SessionToggleStore()
        self._background_tasks: dict[str, _BackgroundJob] = {}
        self._auto_attach_tasks: set[asyncio.Task[None]] = set()
        self._query_middleware = WikiQueryMiddleware()

    def register(self, api: PluginAPI) -> None:
        api.register_cli(
            "wiki",
            commands=wiki_cli_specs("wiki"),
            handler=self._handle_command,
            description="Wiki workflows",
        )
        api.register_config_schema(WikiConfig)
        api.register_middleware("wiki_query_context", self._query_middleware)
        api.register_hook(
            HookEvent.MESSAGE_RECEIVED.value,
            self._handle_message_received,
        )

    async def activate(self, api: PluginAPI) -> None:
        self._config = WikiConfig.model_validate(dict(api.config))
        self._runtime = api.runtime
        self._query_middleware.set_enabled(self._config.auto_query_enabled)
        self._repo_service = RepoService(api.runtime.storage_path, self._config)
        self._manifest_service = ManifestService(self._repo_service)
        self._ingest_service = IngestService(self._repo_service, self._manifest_service, self._config)
        self._digest_service = DigestService(
            self._repo_service,
            self._manifest_service,
            self._ingest_service,
            api.runtime.llm,
        )
        self._query_service = QueryService(self._repo_service)
        self._query_middleware.set_query_service(self._query_service)
        self._query_middleware.set_usage_mode_store(self._usage_modes)

    async def deactivate(self) -> None:
        for job in self._background_tasks.values():
            job.task.cancel()
        self._background_tasks.clear()
        for task in list(self._auto_attach_tasks):
            task.cancel()
        self._auto_attach_tasks.clear()
        self._query_middleware.set_query_service(None)
        self._query_middleware.set_usage_mode_store(None)
        self._runtime = None
        self._query_service = None
        self._digest_service = None
        self._ingest_service = None
        self._manifest_service = None
        self._repo_service = None

    async def _handle_command(self, ctx: CommandContext, args: str) -> str | None:
        if self._should_run_in_background(args):
            running = self._background_tasks.get(ctx.session_key)
            if running is not None and not running.task.done():
                return "A wiki task is already running in the background for this session."
            task = asyncio.create_task(self._run_background_command(ctx, args))
            self._background_tasks[ctx.session_key] = _BackgroundJob(
                args=args.strip(),
                started_at=time.time(),
                task=task,
            )
            return "Wiki task started in the background. I will send the result when it finishes."
        return await self._execute_command(ctx, args)

    async def _run_background_command(self, ctx: CommandContext, args: str) -> None:
        try:
            result = await self._execute_command(ctx, args)
            if result:
                await ctx.reply(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await ctx.reply(f"Error: {exc}")
        finally:
            current = self._background_tasks.get(ctx.session_key)
            if current is not None and current.task is asyncio.current_task():
                self._background_tasks.pop(ctx.session_key, None)

    async def _execute_command(self, ctx: CommandContext, args: str) -> str:
        from .command import handle_wiki_command

        assert self._repo_service is not None
        assert self._manifest_service is not None
        assert self._query_service is not None
        assert self._ingest_service is not None
        assert self._digest_service is not None
        return await handle_wiki_command(
            ctx,
            args,
            repo_service=self._repo_service,
            manifest_service=self._manifest_service,
            query_service=self._query_service,
            usage_mode_store=self._usage_modes,
            attachment_store=self._attachment_auto_add,
            get_job_status=self._job_status,
            ingest_service=self._ingest_service,
            digest_service=self._digest_service,
        )

    async def _handle_message_received(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        media: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        await self._query_middleware.capture_message_context(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            media=media,
            **kwargs,
        )
        if not media or not self._attachment_auto_add.is_enabled(session_key):
            return
        task = asyncio.create_task(
            self._auto_ingest_media(channel=channel, chat_id=chat_id, media=list(media))
        )
        self._auto_attach_tasks.add(task)
        task.add_done_callback(self._auto_attach_tasks.discard)

    async def _auto_ingest_media(
        self,
        *,
        channel: str,
        chat_id: str,
        media: list[str],
    ) -> None:
        if (
            self._runtime is None
            or self._repo_service is None
            or self._ingest_service is None
            or self._digest_service is None
        ):
            return

        ingested = []
        errors: list[str] = []
        for item in media:
            try:
                ingested.append(await self._ingest_service.ingest_file(item))
            except Exception as exc:
                errors.append(f"{Path(item).name}: {exc}")

        lines = ["## Wiki Attachment Import"]
        if ingested:
            results = await self._digest_service.digest_pending()
            duplicate_count = sum(1 for source in ingested if source.duplicate)
            if results:
                lines.extend(["", self._digest_service.format_status_table(results)])
            elif duplicate_count == len(ingested):
                lines.extend(["", f"All {duplicate_count} attachment(s) were already in the wiki."])
            else:
                lines.extend(["", "No pending raw sources to digest."])
        if errors:
            lines.extend(["", "### Errors"])
            lines.extend(f"- {item}" for item in errors)
        if len(lines) == 1:
            return
        await self._runtime.agent_loop.bus.publish_outbound(
            OutboundMessage(channel=channel, chat_id=chat_id, content="\n".join(lines))
        )

    def _should_run_in_background(self, args: str) -> bool:
        stripped = args.strip()
        if not stripped:
            return False
        if stripped.startswith(("http://", "https://")):
            return True
        if self._ingest_service is not None and self._ingest_service.extract_source_urls(stripped):
            return True
        try:
            parts = shlex.split(stripped)
        except ValueError:
            return False
        if not parts:
            return False
        return parts[0].lower() in {"add", "digest"}

    def _job_status(self, session_key: str) -> str:
        job = self._background_tasks.get(session_key)
        if job is None or job.task.done():
            return "No wiki background task is running."
        elapsed = max(0, int(time.time() - job.started_at))
        return (
            "Wiki background task is running.\n"
            f"- command: `{job.args}`\n"
            f"- elapsed_seconds: {elapsed}"
        )
