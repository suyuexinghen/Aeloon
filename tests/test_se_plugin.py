"""Tests for the SE-agent plugin registration and lifecycle."""

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
    loop.process_direct = AsyncMock(return_value="se result")
    loop.profiler = MagicMock(enabled=False)
    return loop


@pytest.fixture
def se_plugin_dir() -> Path:
    return Path(__file__).parent.parent / "aeloon" / "plugins" / "se"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestSEConfig:
    def test_import_and_defaults(self) -> None:
        from aeloon.plugins.SoftwareEngineering.config import SEConfig, SEGovernanceConfig

        cfg = SEConfig()
        assert cfg.enabled is False
        assert cfg.workspace_dir == "~/.aeloon/se/workspaces"
        assert cfg.default_budget_tokens == 50_000

        gov = SEGovernanceConfig()
        assert gov.enable_audit is True
        assert gov.max_repair_cycles == 3

    def test_custom_values(self) -> None:
        from aeloon.plugins.SoftwareEngineering.config import SEConfig

        cfg = SEConfig(enabled=True, default_budget_tokens=100_000)
        assert cfg.enabled is True
        assert cfg.default_budget_tokens == 100_000


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestSEManifest:
    def test_load_bundled_manifest(self, se_plugin_dir: Path) -> None:
        manifest_path = se_plugin_dir / "aeloon.plugin.json"
        if not manifest_path.exists():
            pytest.skip("Bundled manifest not found")
        m = load_manifest(manifest_path)
        assert m.id == "aeloon.se"
        assert m.name == "SE-agent"
        assert "se" in m.provides.commands
        assert "aeloon.soulanchor" in m.requires.plugins


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestSEPluginRegistration:
    def test_register_creates_pending_records(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.SoftwareEngineering.plugin import SEPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.se",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.se",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = SEPlugin()
        plugin.register(api)

        assert any(r.name == "se" for r in api._pending_commands)
        assert any(r.name == "se" for r in api._pending_cli)
        assert len(api._pending_config_schemas) == 1
        assert any(t.name == "test_runner" for t in api._pending_tools)
        assert any(t.name == "linter" for t in api._pending_tools)

    def test_commit_after_register(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.SoftwareEngineering.plugin import SEPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.se",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.se",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = SEPlugin()
        plugin.register(api)
        api._commit()

        assert "se" in registry.commands
        assert registry.commands["se"].plugin_id == "aeloon.se"


# ---------------------------------------------------------------------------
# Activation tests
# ---------------------------------------------------------------------------


class TestSEPluginActivation:
    @pytest.mark.asyncio
    async def test_activate_creates_storage(
        self, mock_agent_loop: MagicMock, tmp_path: Path
    ) -> None:
        from aeloon.plugins.SoftwareEngineering.plugin import SEPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.se",
            config={"enabled": True},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="aeloon.se",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = SEPlugin()
        plugin.register(api)
        await plugin.activate(api)

        assert runtime.storage_path.exists()

    @pytest.mark.asyncio
    async def test_deactivate_resets_pipeline(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.SoftwareEngineering.plugin import SEPlugin

        plugin = SEPlugin()
        plugin._pipeline = MagicMock()
        await plugin.deactivate()
        assert plugin._pipeline is None


# ---------------------------------------------------------------------------
# Command routing tests
# ---------------------------------------------------------------------------


class TestSECommandRouting:
    @pytest.mark.asyncio
    async def test_help_command(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.SoftwareEngineering.plugin import SEPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.se",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.se",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = SEPlugin()
        plugin.register(api)

        ctx = MagicMock()
        ctx.session_key = "test"
        ctx.send_progress = AsyncMock()

        result = await plugin._handle_command(ctx, "help")
        assert "SE-agent" in result
        assert "/se" in result

    @pytest.mark.asyncio
    async def test_empty_args_returns_help(self, mock_agent_loop: MagicMock) -> None:
        from aeloon.plugins.SoftwareEngineering.plugin import SEPlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.se",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.se",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = SEPlugin()
        plugin.register(api)

        ctx = MagicMock()
        ctx.session_key = "test"
        ctx.send_progress = AsyncMock()

        result = await plugin._handle_command(ctx, "")
        assert "SE-agent" in result
