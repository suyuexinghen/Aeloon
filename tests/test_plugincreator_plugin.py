"""Tests for PluginCreator plugin registration and lifecycle."""

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
    loop.provider.chat = AsyncMock(return_value=MagicMock(content="test response"))
    loop.model = "test-model"
    loop.process_direct = AsyncMock(return_value="pc result")
    loop.profiler = MagicMock(enabled=False)
    return loop


@pytest.fixture
def pc_plugin_dir() -> Path:
    return Path(__file__).parent.parent / "aeloon" / "plugins" / "PluginCreator"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestPluginCreatorConfig:
    def test_import_and_defaults(self) -> None:
        from aeloon.plugins.PluginCreator.config import PluginCreatorConfig

        cfg = PluginCreatorConfig()
        assert cfg.enabled is False
        assert cfg.workspace_dir == "~/.aeloon/plugincreator/workspaces"
        assert cfg.default_maturity == "mvp"
        assert cfg.plan_first is True

    def test_custom_values(self) -> None:
        from aeloon.plugins.PluginCreator.config import PluginCreatorConfig

        cfg = PluginCreatorConfig(enabled=True, default_maturity="prototype")
        assert cfg.enabled is True
        assert cfg.default_maturity == "prototype"


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestPluginCreatorManifest:
    def test_load_bundled_manifest(self, pc_plugin_dir: Path) -> None:
        manifest_path = pc_plugin_dir / "aeloon.plugin.json"
        if not manifest_path.exists():
            pytest.skip("Bundled manifest not found")
        m = load_manifest(manifest_path)
        assert m.id == "aeloon.plugincreator"
        assert m.name == "PluginCreator"
        assert "pc" in m.provides.commands


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestPluginCreatorRegistration:
    def test_register_creates_pending_records(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.PluginCreator.plugin import PluginCreatorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.plugincreator",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.plugincreator",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = PluginCreatorPlugin()
        plugin.register(api)

        assert any(r.name == "pc" for r in api._pending_commands)
        assert any(r.name == "pc" for r in api._pending_cli)
        assert {spec.command_name for spec in api._pending_cli[0].commands} == {
            "plan",
            "status",
            "history",
        }
        assert len(api._pending_config_schemas) == 1

    def test_commit_after_register(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.PluginCreator.plugin import PluginCreatorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.plugincreator",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.plugincreator",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = PluginCreatorPlugin()
        plugin.register(api)
        api._commit()

        assert "pc" in registry.commands
        assert registry.commands["pc"].plugin_id == "aeloon.plugincreator"
        assert "pc" in registry.cli_registrars


# ---------------------------------------------------------------------------
# Activation tests
# ---------------------------------------------------------------------------


class TestPluginCreatorActivation:
    @pytest.mark.asyncio
    async def test_activate_creates_storage(
        self, mock_agent_loop: MagicMock, tmp_path: Path
    ) -> None:
        from aeloon.plugins.PluginCreator.plugin import PluginCreatorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.plugincreator",
            config={"enabled": True},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="aeloon.plugincreator",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = PluginCreatorPlugin()
        plugin.register(api)
        await plugin.activate(api)

        assert runtime.storage_path.exists()

    @pytest.mark.asyncio
    async def test_deactivate_resets_pipeline(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.PluginCreator.plugin import PluginCreatorPlugin

        plugin = PluginCreatorPlugin()
        plugin._pipeline = MagicMock()
        await plugin.deactivate()
        assert plugin._pipeline is None


# ---------------------------------------------------------------------------
# Command routing tests
# ---------------------------------------------------------------------------


class TestPluginCreatorCommandRouting:
    @pytest.mark.asyncio
    async def test_help_command(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.PluginCreator.plugin import PluginCreatorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.plugincreator",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.plugincreator",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = PluginCreatorPlugin()
        plugin.register(api)

        ctx = MagicMock()
        ctx.session_key = "test"
        ctx.send_progress = AsyncMock()

        result = await plugin._handle_command(ctx, "help")
        assert "PluginCreator" in result
        assert "/pc" in result

    @pytest.mark.asyncio
    async def test_empty_args_returns_help(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.PluginCreator.plugin import PluginCreatorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.plugincreator",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.plugincreator",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = PluginCreatorPlugin()
        plugin.register(api)

        ctx = MagicMock()
        ctx.session_key = "test"
        ctx.send_progress = AsyncMock()

        result = await plugin._handle_command(ctx, "")
        assert "PluginCreator" in result

    @pytest.mark.asyncio
    async def test_status_command(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.PluginCreator.plugin import PluginCreatorPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.plugincreator",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.plugincreator",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = PluginCreatorPlugin()
        plugin.register(api)

        ctx = MagicMock()
        ctx.session_key = "test"
        ctx.send_progress = AsyncMock()

        result = await plugin._handle_command(ctx, "status")
        assert "No PluginCreator plans" in result
