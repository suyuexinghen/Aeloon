"""Command router for `/wiki`."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from aeloon.plugins._sdk.types import CommandContext

    from .services.digest_service import DigestService
    from .services.ingest_service import IngestService
    from .services.manifest_service import ManifestService
    from .services.query_service import QueryService
    from .services.repo_service import RepoService
    from .services.usage_mode import SessionToggleStore, UsageModeStore


_HELP_TEXT = """## Wiki

- `/wiki init [path]`: initialize the knowledge base root
- `/wiki <URL|text>`: ingest and digest one or more URLs / arXiv refs from free-form text
- `/wiki add <path|text>`: ingest and digest one local file path or many refs from text
- `/wiki digest`: re-run digest
- `/wiki list`: list tracked raw sources and wiki entries
- `/wiki get <entry>`: show one wiki entry
- `/wiki map [entry]`: render the wiki relation map as Mermaid
- `/wiki jobs`: show the current background wiki task
- `/wiki use <off|prefer-local|local-only|status>`: control wiki use during chat
- `/wiki attach <on|off|status>`: auto-add future message attachments for this session
- `/wiki remove --confirm`: delete the current knowledge base
- `/wiki status`: show repo status
"""

_INIT_REQUIRED = "Wiki is not initialized. Run `/wiki init [path]` first."


async def handle_wiki_command(
    ctx: "CommandContext",
    args: str,
    *,
    repo_service: "RepoService",
    manifest_service: "ManifestService",
    query_service: "QueryService",
    usage_mode_store: "UsageModeStore",
    attachment_store: "SessionToggleStore",
    get_job_status: Callable[[str], str],
    ingest_service: "IngestService",
    digest_service: "DigestService",
) -> str:
    """Route `/wiki` subcommands."""
    args = args.strip()
    if not args or args in {"help", "--help", "-h"}:
        return _HELP_TEXT

    if args.startswith(("http://", "https://")):
        return await _ingest_and_digest(
            args,
            repo_service=repo_service,
            ingest_service=ingest_service,
            digest_service=digest_service,
        )

    try:
        parts = shlex.split(args)
    except ValueError as exc:
        return f"Error: {exc}"

    if not parts:
        return _HELP_TEXT

    subcommand = parts[0].lower()
    rest = parts[1:]
    raw_rest = args[len(parts[0]) :].strip()

    if subcommand == "init":
        target = rest[0] if rest else None
        layout = repo_service.initialize(target)
        return f"Initialized wiki at `{layout.root}`."

    if subcommand == "status":
        status = repo_service.build_status()
        mode = usage_mode_store.get_mode(ctx.session_key)
        lines = [
            "## Wiki Status",
            "",
            f"- repo_root: `{status.repo_root}`",
            f"- initialized: {'yes' if status.initialized else 'no'}",
            f"- use_mode: {mode}",
            f"- raw_sources: {status.raw_sources}",
            f"- domains: {status.domains}",
            f"- summaries: {status.summaries}",
            f"- concepts: {status.concepts}",
        ]
        for note in status.notes:
            lines.append(f"- note: {note}")
        return "\n".join(lines)

    if subcommand == "jobs":
        return get_job_status(ctx.session_key)

    if subcommand == "remove":
        if not repo_service.is_initialized():
            return _INIT_REQUIRED
        if "--confirm" not in rest:
            return (
                f"About to delete the wiki at `{repo_service.repo_root}`.\n"
                "Re-run `/wiki remove --confirm` to proceed."
            )
        removed = repo_service.remove_knowledge_base()
        return f"Removed wiki at `{removed}`."

    if subcommand == "use":
        return _handle_use_mode(ctx.session_key, rest, usage_mode_store)

    if subcommand == "attach":
        return _handle_attach_mode(
            session_key=ctx.session_key,
            args=rest,
            attachment_store=attachment_store,
        )

    if not repo_service.is_initialized():
        return _INIT_REQUIRED

    if subcommand == "add":
        if not raw_rest:
            return "Usage: /wiki add <path|text>"
        return await _ingest_and_digest(
            raw_rest,
            repo_service=repo_service,
            ingest_service=ingest_service,
            digest_service=digest_service,
        )
    if subcommand == "digest":
        results = await digest_service.digest_pending()
        return digest_service.format_status_table(results)
    if subcommand == "list":
        return _format_list_output(manifest_service, query_service)
    if subcommand == "get":
        if not rest:
            return "Usage: /wiki get <entry>"
        entry = query_service.get_entry(rest[0])
        if entry is None:
            return f"Wiki entry not found: {rest[0]}"
        return entry.text
    if subcommand == "map":
        map_text = query_service.format_map(rest[0] if rest else None)
        if not map_text:
            target = rest[0] if rest else "wiki"
            return f"Wiki map is empty for: {target}"
        return map_text
    if ingest_service.extract_source_urls(args):
        return await _ingest_and_digest(
            args,
            repo_service=repo_service,
            ingest_service=ingest_service,
            digest_service=digest_service,
        )
    return f"Unknown subcommand: {subcommand}. Use /wiki help for usage."


async def _ingest_and_digest(
    raw_input: str,
    *,
    repo_service: "RepoService",
    ingest_service: "IngestService",
    digest_service: "DigestService",
) -> str:
    if not repo_service.is_initialized():
        return _INIT_REQUIRED
    await ingest_service.ingest_input(raw_input)
    results = await digest_service.digest_pending()
    return digest_service.format_status_table(results)


def _format_list_output(
    manifest_service: "ManifestService",
    query_service: "QueryService",
) -> str:
    sources = manifest_service.load()["sources"]
    entries = query_service.list_entries()
    lines = ["## Wiki List", ""]

    lines.append("### Raw Sources")
    if not sources:
        lines.append("- none")
    else:
        for item in sources:
            display_name = str(item.get("display_name", "")).strip() or str(item.get("path", ""))
            status = str(item.get("status", "unknown"))
            lines.append(f"- `{display_name}` [{status}]")

    lines.append("")
    lines.append("### Wiki Entries")
    if not entries:
        lines.append("- none")
    else:
        for entry in entries:
            lines.append(f"- `{entry.entry_id}` -> `{entry.rel_path}`")

    return "\n".join(lines)


def _handle_use_mode(
    session_key: str,
    args: list[str],
    usage_mode_store: "UsageModeStore",
) -> str:
    if not args or args[0].lower() == "status":
        return f"Wiki use mode: {usage_mode_store.get_mode(session_key)}."

    mode = args[0].lower()
    if mode not in {"off", "prefer-local", "local-only"}:
        return "Usage: /wiki use <off|prefer-local|local-only|status>"
    usage_mode_store.set_mode(session_key, mode)
    return f"Wiki use mode set to {mode}."


def _handle_attach_mode(
    *,
    session_key: str,
    args: list[str],
    attachment_store: "SessionToggleStore",
) -> str:
    if not args or args[0].lower() == "status":
        state = "on" if attachment_store.is_enabled(session_key) else "off"
        return f"Wiki attachment auto-add: {state}."

    mode = args[0].lower()
    if mode not in {"on", "off"}:
        return "Usage: /wiki attach <on|off|status>"
    attachment_store.set_enabled(session_key, mode == "on")
    return f"Wiki attachment auto-add set to {mode}."
