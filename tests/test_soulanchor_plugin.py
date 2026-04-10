"""Tests for SoulAnchor plugin registration and /sa command routing.

Follows the same patterns as test_science_plugin.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.plugins._sdk.api import PluginAPI
from aeloon.plugins._sdk.manifest import load_manifest
from aeloon.plugins._sdk.registry import PluginRegistry
from aeloon.plugins._sdk.runtime import PluginRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_loop() -> MagicMock:
    loop = MagicMock()
    loop.provider = MagicMock()
    loop.provider.chat = AsyncMock(return_value=MagicMock(content="ok"))
    loop.model = "test-model"
    loop.process_direct = AsyncMock(return_value="result")
    loop.profiler = MagicMock(enabled=False)
    return loop


@pytest.fixture
def plugin_api(mock_agent_loop: MagicMock, tmp_path: Path) -> PluginAPI:
    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.soulanchor",
        config={},
        storage_base=tmp_path,
    )
    return PluginAPI(
        plugin_id="aeloon.soulanchor",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestSoulAnchorManifest:
    def test_bundled_manifest_loads(self) -> None:
        manifest_path = (
            Path(__file__).parent.parent
            / "aeloon"
            / "plugins"
            / "SoulAnchor"
            / "aeloon.plugin.json"
        )
        if not manifest_path.exists():
            pytest.skip("Manifest not found")
        m = load_manifest(manifest_path)
        assert m.id == "aeloon.soulanchor"
        assert m.name == "SoulAnchor"
        assert "sa" in m.provides.commands

    def test_manifest_has_services(self) -> None:
        manifest_path = (
            Path(__file__).parent.parent
            / "aeloon"
            / "plugins"
            / "SoulAnchor"
            / "aeloon.plugin.json"
        )
        if not manifest_path.exists():
            pytest.skip("Manifest not found")
        m = load_manifest(manifest_path)
        assert "memory_consolidation" in m.provides.services


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestSoulAnchorConfig:
    def test_defaults(self) -> None:
        from aeloon.plugins.SoulAnchor.config import SoulAnchorConfig

        cfg = SoulAnchorConfig()
        assert cfg.enabled is True
        assert cfg.storage_backend == "jsonl"
        assert cfg.bones_salt == "soulanchor-2026-v1"
        assert cfg.working_memory_capacity == 50
        assert cfg.semantic_memory_capacity == 200

    def test_custom_values(self) -> None:
        from aeloon.plugins.SoulAnchor.config import SoulAnchorConfig

        cfg = SoulAnchorConfig(max_entities=10, bones_salt="my-salt")
        assert cfg.max_entities == 10
        assert cfg.bones_salt == "my-salt"


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestSoulAnchorRegistration:
    def test_register_creates_command(self, plugin_api: PluginAPI) -> None:
        from aeloon.plugins.SoulAnchor.plugin import SoulAnchorPlugin

        plugin = SoulAnchorPlugin()
        plugin.register(plugin_api)

        assert any(r.name == "sa" for r in plugin_api._pending_commands)

    def test_register_creates_service(self, plugin_api: PluginAPI) -> None:
        from aeloon.plugins.SoulAnchor.plugin import SoulAnchorPlugin

        plugin = SoulAnchorPlugin()
        plugin.register(plugin_api)

        assert any(r.name == "memory_consolidation" for r in plugin_api._pending_services)

    def test_register_creates_config_schema(self, plugin_api: PluginAPI) -> None:
        from aeloon.plugins.SoulAnchor.plugin import SoulAnchorPlugin

        plugin = SoulAnchorPlugin()
        plugin.register(plugin_api)

        assert len(plugin_api._pending_config_schemas) == 1

    def test_register_creates_cli(self, plugin_api: PluginAPI) -> None:
        from aeloon.plugins.SoulAnchor.plugin import SoulAnchorPlugin

        plugin = SoulAnchorPlugin()
        plugin.register(plugin_api)

        assert any(r.name == "sa" for r in plugin_api._pending_cli)

    def test_commit_registers_command(self, mock_agent_loop: MagicMock, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.plugin import SoulAnchorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.soulanchor",
            config={},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="aeloon.soulanchor",
            version="0.1.0",
            config={},
            runtime=runtime,
            registry=registry,
        )
        plugin = SoulAnchorPlugin()
        plugin.register(api)
        api._commit()

        assert "sa" in registry.commands
        assert registry.commands["sa"].plugin_id == "aeloon.soulanchor"


# ---------------------------------------------------------------------------
# Activation tests
# ---------------------------------------------------------------------------


class TestSoulAnchorActivation:
    @pytest.mark.asyncio
    async def test_activate_creates_storage(
        self, mock_agent_loop: MagicMock, tmp_path: Path
    ) -> None:
        from aeloon.plugins.SoulAnchor.plugin import SoulAnchorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.soulanchor",
            config={"bones_salt": "test"},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="aeloon.soulanchor",
            version="0.1.0",
            config={"bones_salt": "test"},
            runtime=runtime,
            registry=registry,
        )
        plugin = SoulAnchorPlugin()
        plugin.register(api)
        api._commit()
        await plugin.activate(api)

        assert api.runtime.storage_path.exists()


# ---------------------------------------------------------------------------
# Command routing tests
# ---------------------------------------------------------------------------


class TestSaCommandRouting:
    @pytest.fixture
    def activated_plugin(self, mock_agent_loop: MagicMock, tmp_path: Path) -> tuple:
        """Return (plugin, api) after activation."""
        import asyncio

        from aeloon.plugins.SoulAnchor.plugin import SoulAnchorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.soulanchor",
            config={"bones_salt": "test"},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="aeloon.soulanchor",
            version="0.1.0",
            config={"bones_salt": "test"},
            runtime=runtime,
            registry=registry,
        )
        plugin = SoulAnchorPlugin()
        plugin.register(api)
        api._commit()
        asyncio.get_event_loop().run_until_complete(plugin.activate(api))
        return plugin, api

    def _make_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.session_key = "test_session"
        ctx.channel = "test"
        ctx.reply = AsyncMock()
        ctx.send_progress = AsyncMock()
        ctx.plugin_config = {}
        return ctx

    @pytest.mark.asyncio
    async def test_help_command(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        result = await plugin._handle_sa_command(ctx, "help")
        assert result is not None
        assert "SoulAnchor" in result

    @pytest.mark.asyncio
    async def test_empty_args_returns_help(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        result = await plugin._handle_sa_command(ctx, "")
        assert result is not None
        assert "Usage" in result or "SoulAnchor" in result

    @pytest.mark.asyncio
    async def test_create_command(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        result = await plugin._handle_sa_command(ctx, "create dev_bot --role developer")
        assert result is not None
        assert "dev_bot" in result or "Created" in result

    @pytest.mark.asyncio
    async def test_list_command_empty(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        result = await plugin._handle_sa_command(ctx, "list")
        assert result is not None

    @pytest.mark.asyncio
    async def test_info_missing_id(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        result = await plugin._handle_sa_command(ctx, "info")
        assert result is not None
        assert "Usage" in result

    @pytest.mark.asyncio
    async def test_unknown_subcommand(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        result = await plugin._handle_sa_command(ctx, "bogus_cmd")
        assert result is not None
        assert "Unknown" in result

    @pytest.mark.asyncio
    async def test_create_then_list(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        await plugin._handle_sa_command(ctx, "create tester --role tester")
        result = await plugin._handle_sa_command(ctx, "list")
        assert result is not None
        assert "tester" in result

    @pytest.mark.asyncio
    async def test_create_then_info(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        await plugin._handle_sa_command(ctx, "create arch_one --role architect")
        result = await plugin._handle_sa_command(ctx, "info arch_one")
        assert result is not None
        assert "arch_one" in result or "architect" in result.lower()

    @pytest.mark.asyncio
    async def test_create_then_archive(self, activated_plugin: tuple) -> None:
        plugin, api = activated_plugin
        ctx = self._make_ctx()
        await plugin._handle_sa_command(ctx, "create to_archive --role dev")
        result = await plugin._handle_sa_command(ctx, "archive to_archive")
        assert result is not None
        assert "archived" in result.lower()
