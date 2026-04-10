"""Tests for the Wiki plugin scaffold."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.plugins._sdk.api import PluginAPI
from aeloon.plugins._sdk.manifest import load_manifest
from aeloon.plugins._sdk.registry import PluginRegistry
from aeloon.plugins._sdk.runtime import PluginRuntime


@pytest.fixture
def mock_agent_loop() -> MagicMock:
    loop = MagicMock()
    loop.provider = MagicMock()
    loop.provider.chat = AsyncMock(return_value=MagicMock(content="ok"))
    loop.bus = MagicMock()
    loop.bus.publish_outbound = AsyncMock()
    loop.model = "test-model"
    loop.profiler = MagicMock(enabled=False)
    return loop


def test_manifest_loads() -> None:
    manifest = load_manifest(
        Path("aeloon/plugins/Wiki/aeloon.plugin.json")
    )
    assert manifest.id == "aeloon.wiki"
    assert "wiki" in manifest.provides.commands


def test_register_creates_pending_records(mock_agent_loop: MagicMock, tmp_path: Path) -> None:
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )

    plugin = WikiPlugin()
    plugin.register(api)

    assert any(r.name == "wiki" for r in api._pending_commands)
    assert any(r.name == "wiki" for r in api._pending_cli)
    assert any(r.name == "wiki_query_context" for r in api._pending_middlewares)
    assert len(api._pending_config_schemas) == 1
    assert len(api._pending_hooks) == 1
    assert api._pending_hooks[0].event == "message_received"


def test_commit_after_register(mock_agent_loop: MagicMock, tmp_path: Path) -> None:
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )

    plugin = WikiPlugin()
    plugin.register(api)
    api._commit()

    assert "wiki" in registry.commands
    assert "wiki" in registry.cli_registrars
    assert any(spec.command_name == "status" for spec in registry.cli_registrars["wiki"].commands)
    assert any(record.name == "wiki_query_context" for record in registry.middlewares)
    assert len(registry.hooks_for_event("message_received")) == 1


@pytest.mark.asyncio
async def test_activate_does_not_create_default_repo_layout(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )

    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    repo_root = tmp_path / "aeloon" / "wiki" / "repo"
    assert not repo_root.exists()


@pytest.mark.asyncio
async def test_status_init_and_help_surface_minimal_commands(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx1 = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )
    assert await plugin._handle_command(ctx1, "status") == (
        "## Wiki Status\n\n"
        f"- repo_root: `{tmp_path / 'aeloon' / 'wiki' / 'repo'}`\n"
        "- initialized: no\n"
        "- use_mode: prefer-local\n"
        "- raw_sources: 0\n"
        "- domains: 0\n"
        "- summaries: 0\n"
        "- concepts: 0\n"
        "- note: Knowledge base is not initialized."
    )
    assert await plugin._handle_command(ctx1, "init") == (
        f"Initialized wiki at `{tmp_path / 'aeloon' / 'wiki' / 'repo'}`."
    )
    help_text = await plugin._handle_command(ctx1, "help")
    assert "/wiki list" in help_text
    assert "/wiki get <entry>" in help_text
    assert "/wiki map [entry]" in help_text
    assert "/wiki jobs" in help_text
    assert "/wiki use <off|prefer-local|local-only|status>" in help_text
    assert "/wiki attach <on|off|status>" in help_text
    assert "/wiki all" not in help_text
    assert "graph" not in help_text


@pytest.mark.asyncio
async def test_use_mode_can_be_changed_per_session(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )

    assert await plugin._handle_command(ctx, "use status") == "Wiki use mode: prefer-local."
    assert await plugin._handle_command(ctx, "use local-only") == "Wiki use mode set to local-only."
    assert await plugin._handle_command(ctx, "use status") == "Wiki use mode: local-only."


@pytest.mark.asyncio
async def test_attachment_auto_add_can_be_changed_per_session(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )

    assert await plugin._handle_command(ctx, "attach status") == "Wiki attachment auto-add: off."
    assert await plugin._handle_command(ctx, "attach on") == "Wiki attachment auto-add set to on."
    assert await plugin._handle_command(ctx, "attach status") == "Wiki attachment auto-add: on."
    assert await plugin._handle_command(ctx, "attach off") == "Wiki attachment auto-add set to off."


@pytest.mark.asyncio
async def test_remove_requires_confirmation_and_deletes_repo(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )
    await plugin._handle_command(ctx, "init")

    warning = await plugin._handle_command(ctx, "remove")
    assert "Re-run `/wiki remove --confirm`" in warning

    removed = await plugin._handle_command(ctx, "remove --confirm")
    assert "Removed wiki" in removed
    assert plugin._repo_service is not None
    assert not plugin._repo_service.repo_root.exists()


@pytest.mark.asyncio
async def test_list_and_get_show_wiki_entries(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )
    await plugin._handle_command(ctx, "init")
    assert plugin._repo_service is not None
    page = plugin._repo_service.layout.wiki_concepts / "concept-agent-systems.md"
    page.write_text(
        "---\n"
        'id: concept-agent-systems\n'
        'type: concept\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        'title: "Agent Systems"\n'
        'summary: "Core patterns."\n'
        "sources:\n"
        "links:\n"
        "depends_on:\n"
        "derived_from:\n"
        "---\n"
        "Agent systems coordinate tools.\n",
        encoding="utf-8",
    )

    listed = await plugin._handle_command(ctx, "list")
    assert "`concept-agent-systems` -> `wiki/concepts/concept-agent-systems.md`" in listed

    content = await plugin._handle_command(ctx, "get concept-agent-systems")
    assert content is not None
    assert "Agent systems coordinate tools." in content


@pytest.mark.asyncio
async def test_map_returns_mermaid_for_existing_entries(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )
    await plugin._handle_command(ctx, "init")
    assert plugin._repo_service is not None
    (plugin._repo_service.layout.wiki_domains / "domain-agent-systems.md").write_text(
        "---\n"
        "id: domain-agent-systems\n"
        "type: domain\n"
        'title: "Agent Systems"\n'
        'summary: "Domain grouping."\n'
        "member_refs:\n"
        "---\n"
        "Agent Systems groups related entries.\n",
        encoding="utf-8",
    )
    (plugin._repo_service.layout.wiki_domains / "domain-research-automation.md").write_text(
        "---\n"
        "id: domain-research-automation\n"
        "type: domain\n"
        'title: "Research Automation"\n'
        'summary: "Domain grouping."\n'
        "member_refs:\n"
        "---\n"
        "Research Automation groups related entries.\n",
        encoding="utf-8",
    )
    (plugin._repo_service.layout.wiki_concepts / "concept-agent-systems.md").write_text(
        "---\n"
        'id: concept-agent-systems\n'
        'type: concept\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        "  - domain-research-automation\n"
        'title: "Agent Systems"\n'
        'summary: "Core patterns."\n'
        "sources:\n"
        "links:\n"
        "depends_on:\n"
        "derived_from:\n"
        "---\n"
        "Agent systems coordinate tools.\n",
        encoding="utf-8",
    )
    (plugin._repo_service.layout.wiki_summaries / "summary-agent-overview.md").write_text(
        "---\n"
        'id: summary-agent-overview\n'
        'type: summary\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        'title: "Agent Overview"\n'
        'summary: "Overview."\n'
        "sources:\n"
        "links:\n"
        "depends_on:\n"
        "  - concept-agent-systems\n"
        "derived_from:\n"
        "---\n"
        "Summary body.\n",
        encoding="utf-8",
    )

    map_text = await plugin._handle_command(ctx, "map")
    assert map_text is not None
    assert map_text.startswith("```mermaid")
    assert 'root["Wiki"]' in map_text
    assert "domain-agent-systems" in map_text
    assert "-.->|domain_ref|" in map_text


@pytest.mark.asyncio
async def test_heavy_wiki_command_runs_in_background_and_replies_on_completion(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )
    await plugin._handle_command(ctx, "init")

    gate = asyncio.Event()

    async def _slow_ingest(_: str) -> None:
        await gate.wait()

    assert plugin._ingest_service is not None
    assert plugin._digest_service is not None
    plugin._ingest_service.ingest_url = AsyncMock(side_effect=_slow_ingest)  # type: ignore[method-assign]
    plugin._digest_service.digest_pending = AsyncMock(return_value=[])  # type: ignore[method-assign]

    immediate = await plugin._handle_command(ctx, "https://example.com/path")
    assert immediate == "Wiki task started in the background. I will send the result when it finishes."
    ctx.reply.assert_not_awaited()

    gate.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    ctx.reply.assert_awaited_once_with("No pending raw sources to digest.")


@pytest.mark.asyncio
async def test_jobs_reports_running_background_task(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )
    await plugin._handle_command(ctx, "init")

    gate = asyncio.Event()

    async def _slow_ingest(_: str) -> None:
        await gate.wait()

    assert plugin._ingest_service is not None
    plugin._ingest_service.ingest_url = AsyncMock(side_effect=_slow_ingest)  # type: ignore[method-assign]

    immediate = await plugin._handle_command(ctx, "https://example.com/path")
    assert immediate == "Wiki task started in the background. I will send the result when it finishes."

    jobs = await plugin._handle_command(ctx, "jobs")
    assert jobs is not None
    assert "Wiki background task is running." in jobs
    assert "`https://example.com/path`" in jobs

    gate.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_multiline_bibtex_input_ingests_all_detected_refs(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )
    await plugin._handle_command(ctx, "init")

    bibtex = """
@Article{Qin2024MooncakeAK,
 volume = {abs/2407.00079},
}

@misc{liu2025megascaleinferservingmixtureofexperts,
      eprint={2504.02263},
      url={https://arxiv.org/abs/2504.02263},
}
"""

    assert plugin._ingest_service is not None
    assert plugin._digest_service is not None
    plugin._should_run_in_background = lambda _: False  # type: ignore[method-assign]
    plugin._ingest_service.ingest_url = AsyncMock()  # type: ignore[method-assign]
    plugin._digest_service.digest_pending = AsyncMock(return_value=[])  # type: ignore[method-assign]

    result = await plugin._handle_command(ctx, bibtex)

    assert result == "No pending raw sources to digest."
    assert plugin._ingest_service.ingest_url.await_args_list[0].args == (
        "https://arxiv.org/abs/2407.00079",
    )
    assert plugin._ingest_service.ingest_url.await_args_list[1].args == (
        "https://arxiv.org/abs/2504.02263",
    )


@pytest.mark.asyncio
async def test_message_received_auto_ingests_attachments_when_enabled(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins._sdk.types import CommandContext
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    ctx = CommandContext(
        session_key="cli:one",
        channel="cli",
        reply=AsyncMock(),
        send_progress=AsyncMock(),
        plugin_config={},
    )
    await plugin._handle_command(ctx, "init")
    await plugin._handle_command(ctx, "attach on")

    assert plugin._ingest_service is not None
    assert plugin._digest_service is not None
    plugin._ingest_service.ingest_file = AsyncMock(side_effect=[MagicMock(duplicate=False)])  # type: ignore[method-assign]
    plugin._digest_service.digest_pending = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await plugin._handle_message_received(
        session_key="cli:one",
        channel="cli",
        chat_id="chat-1",
        content="see attached",
        media=["/tmp/example.pdf"],
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    plugin._ingest_service.ingest_file.assert_awaited_once_with("/tmp/example.pdf")
    mock_agent_loop.bus.publish_outbound.assert_awaited()
    outbound = mock_agent_loop.bus.publish_outbound.await_args.args[0]
    assert "## Wiki Attachment Import" in outbound.content


@pytest.mark.asyncio
async def test_message_received_skips_attachment_ingest_when_disabled(
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    from aeloon.plugins.Wiki.plugin import WikiPlugin

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.wiki",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.wiki",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )
    plugin = WikiPlugin()
    plugin.register(api)
    await plugin.activate(api)

    assert plugin._ingest_service is not None
    plugin._ingest_service.ingest_file = AsyncMock()  # type: ignore[method-assign]

    await plugin._handle_message_received(
        session_key="cli:one",
        channel="cli",
        chat_id="chat-1",
        content="see attached",
        media=["/tmp/example.pdf"],
    )
    await asyncio.sleep(0)

    plugin._ingest_service.ingest_file.assert_not_awaited()
