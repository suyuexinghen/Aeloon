"""Tests for the Aeloon Science Plugin (SP2 — Science Plugin Migration).

Covers: config extraction, manifest loading, plugin registration,
plugin activation, and plugin-path pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aeloon.plugins._sdk.api import PluginAPI
from aeloon.plugins._sdk.discovery import PluginDiscovery
from aeloon.plugins._sdk.hooks import HookDispatcher
from aeloon.plugins._sdk.loader import PluginLoader
from aeloon.plugins._sdk.manager import PluginManager
from aeloon.plugins._sdk.manifest import load_manifest
from aeloon.plugins._sdk.registry import PluginRegistry
from aeloon.plugins._sdk.runtime import PluginRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_loop() -> MagicMock:
    """Mock AgentLoop with process_direct and provider."""
    loop = MagicMock()
    loop.provider = MagicMock()
    loop.provider.chat = AsyncMock(return_value=MagicMock(content="test response"))
    loop.model = "test-model"
    loop.process_direct = AsyncMock(return_value="science result")
    loop.profiler = MagicMock(enabled=False)
    return loop


@pytest.fixture
def science_plugin_dir(tmp_path: Path) -> Path:
    """Create a science plugin directory with manifest."""
    plugin_path = tmp_path / "aeloon.science"
    plugin_path.mkdir()
    manifest = {
        "id": "aeloon.science",
        "name": "AI4S Science Agent",
        "version": "0.1.0",
        "description": "AI for Science agent mode",
        "author": "AetherHeart",
        "entry": "aeloon.plugins.ScienceResearch.plugin:SciencePlugin",
        "provides": {
            "commands": ["sr"],
            "middlewares": ["science_audit", "science_budget", "science_risk_gate"],
        },
        "requires": {"aeloon_version": ">=0.1.0"},
    }
    (plugin_path / "aeloon.plugin.json").write_text(json.dumps(manifest))
    return plugin_path


# ---------------------------------------------------------------------------
# TestScienceConfig
# ---------------------------------------------------------------------------


class TestScienceConfig:
    """Test config extraction — import from plugin module path."""

    def test_import_from_new_module(self) -> None:
        """ScienceConfig importable from aeloon.plugins.ScienceResearch.config."""
        from aeloon.plugins.ScienceResearch.config import GovernanceConfig, ScienceConfig

        cfg = ScienceConfig()
        assert cfg.enabled is False
        assert cfg.storage_dir == "~/.aeloon/science"
        assert cfg.default_budget_tokens == 50_000
        gov = GovernanceConfig()
        assert gov.enable_audit is True
        assert gov.risk_level == "green"

    def test_config_with_custom_values(self) -> None:
        """ScienceConfig accepts custom values."""
        from aeloon.plugins.ScienceResearch.config import ScienceConfig

        cfg = ScienceConfig(
            enabled=True,
            storage_dir="/tmp/science",
            default_budget_tokens=100_000,
        )
        assert cfg.enabled is True
        assert cfg.storage_dir == "/tmp/science"
        assert cfg.default_budget_tokens == 100_000


# ---------------------------------------------------------------------------
# TestScienceManifest
# ---------------------------------------------------------------------------


class TestScienceManifest:
    """Test manifest loading from the real plugin directory."""

    def test_load_bundled_manifest(self) -> None:
        """Load the actual aeloon.plugin.json from aeloon/plugins/ScienceResearch/."""
        manifest_path = (
            Path(__file__).parent.parent
            / "aeloon"
            / "plugins"
            / "ScienceResearch"
            / "aeloon.plugin.json"
        )
        if not manifest_path.exists():
            pytest.skip("Bundled manifest not found (expected in source tree)")
        m = load_manifest(manifest_path)
        assert m.id == "aeloon.science"
        assert m.name == "AI4S Science Agent"
        assert "sr" in m.provides.commands

    def test_manifest_from_fixture(self, science_plugin_dir: Path) -> None:
        """Load manifest from test fixture."""
        m = load_manifest(science_plugin_dir / "aeloon.plugin.json")
        assert m.id == "aeloon.science"
        assert m.version == "0.1.0"
        assert m.entry == "aeloon.plugins.ScienceResearch.plugin:SciencePlugin"


# ---------------------------------------------------------------------------
# TestSciencePluginRegistration
# ---------------------------------------------------------------------------


class TestSciencePluginRegistration:
    """Test SciencePlugin.register() correctly registers command, CLI, config."""

    def test_register_creates_pending_records(self, mock_agent_loop: MagicMock) -> None:
        """register() populates pending commands, CLI, and config schema."""
        from aeloon.plugins.ScienceResearch.plugin import SciencePlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.science",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.science",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = SciencePlugin()
        plugin.register(api)

        # Check pending records (not yet committed)
        assert any(r.name == "sr" for r in api._pending_commands)
        assert any(r.name == "sr" for r in api._pending_cli)
        assert {spec.command_name for spec in api._pending_cli[0].commands} == {
            "run",
            "status",
            "history",
        }
        assert len(api._pending_config_schemas) == 1

    def test_commit_after_register(self, mock_agent_loop: MagicMock) -> None:
        """After commit, command is in registry."""
        from aeloon.plugins.ScienceResearch.plugin import SciencePlugin

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.science",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.science",
            version="0.1.0",
            config={"enabled": True},
            runtime=runtime,
            registry=registry,
        )

        plugin = SciencePlugin()
        plugin.register(api)
        api._commit()

        assert "sr" in registry.commands
        assert registry.commands["sr"].plugin_id == "aeloon.science"
        assert "sr" in registry.cli_registrars


# ---------------------------------------------------------------------------
# TestSciencePluginActivation
# ---------------------------------------------------------------------------


class TestSciencePluginActivation:
    """Test full plugin boot lifecycle with mocked agent loop."""

    @pytest.mark.asyncio
    async def test_boot_discovers_and_activates(
        self, science_plugin_dir: Path, mock_agent_loop: MagicMock
    ) -> None:
        """Full boot: discover -> register -> activate."""
        from aeloon.plugins.ScienceResearch.plugin import SciencePlugin

        with patch("aeloon.plugins._sdk.loader.importlib.import_module") as mock_import:
            mod = MagicMock()
            mod.SciencePlugin = SciencePlugin
            mock_import.return_value = mod

            registry = PluginRegistry()
            discovery = PluginDiscovery(bundled_dir=science_plugin_dir.parent)
            loader = PluginLoader()
            hooks = HookDispatcher()
            manager = PluginManager(
                registry=registry,
                discovery=discovery,
                loader=loader,
                hook_dispatcher=hooks,
                agent_loop=mock_agent_loop,
                plugin_config={"aeloon.science": {"enabled": True}},
                storage_base=Path("/tmp"),
            )

            result = await manager.boot()
            assert "aeloon.science" in result.loaded
            assert "sr" in registry.commands


# ---------------------------------------------------------------------------
# TestSciencePipelinePluginPath
# ---------------------------------------------------------------------------


class TestSciencePipelinePluginPath:
    """Test that SciencePipeline(runtime=...) works via PluginRuntime."""

    def test_init_with_runtime(self, mock_agent_loop: MagicMock) -> None:
        """Plugin-path construction with PluginRuntime."""
        from aeloon.plugins.ScienceResearch.pipeline import SciencePipeline

        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.science",
            config={"enabled": True},
            storage_base=Path("/tmp"),
        )
        pipeline = SciencePipeline(runtime=runtime, storage_dir="/tmp/test-science")
        assert pipeline._runtime is runtime

    def test_runtime_storage_path_used(self, mock_agent_loop: MagicMock, tmp_path: Path) -> None:
        """When runtime is provided without storage_dir, runtime.storage_path is used."""
        from aeloon.plugins.ScienceResearch.pipeline import SciencePipeline

        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.science",
            config={},
            storage_base=tmp_path,
        )
        pipeline = SciencePipeline(runtime=runtime)
        # Storage path should be derived from runtime
        assert pipeline._runtime.storage_path.parent == tmp_path / "aeloon"


# ---------------------------------------------------------------------------
# TestOrchestratorPluginPath
# ---------------------------------------------------------------------------


class TestOrchestratorPluginPath:
    """Test orchestrator runtime-based construction."""

    def test_dag_orchestrator_with_runtime(self, mock_agent_loop: MagicMock) -> None:
        """DAGOrchestrator(runtime=...) stores the runtime."""
        from aeloon.plugins.ScienceResearch.orchestrator import DAGOrchestrator

        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.science",
            config={},
            storage_base=Path("/tmp"),
        )
        orch = DAGOrchestrator(runtime=runtime)
        assert orch._runtime is runtime

    def test_sequential_orchestrator_with_runtime(self, mock_agent_loop: MagicMock) -> None:
        """SequentialOrchestrator(runtime=...) stores the runtime."""
        from aeloon.plugins.ScienceResearch.orchestrator import SequentialOrchestrator

        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.science",
            config={},
            storage_base=Path("/tmp"),
        )
        orch = SequentialOrchestrator(runtime=runtime)
        assert orch._runtime is runtime


# ---------------------------------------------------------------------------
# TestSciencePluginIntegration (P3-2)
# ---------------------------------------------------------------------------


class TestSciencePluginIntegration:
    """End-to-end integration tests for Science plugin."""

    @pytest.mark.asyncio
    async def test_boot_to_command_routing(
        self, science_plugin_dir: Path, mock_agent_loop: MagicMock
    ) -> None:
        """Full cycle: boot -> command in registry -> handler callable."""
        from aeloon.plugins.ScienceResearch.plugin import SciencePlugin

        with patch("aeloon.plugins._sdk.loader.importlib.import_module") as mock_import:
            mod = MagicMock()
            mod.SciencePlugin = SciencePlugin
            mock_import.return_value = mod

            registry = PluginRegistry()
            manager = PluginManager(
                registry=registry,
                discovery=PluginDiscovery(bundled_dir=science_plugin_dir.parent),
                loader=PluginLoader(),
                hook_dispatcher=HookDispatcher(),
                agent_loop=mock_agent_loop,
                plugin_config={"aeloon.science": {"enabled": True}},
                storage_base=Path("/tmp"),
            )
            result = await manager.boot()
            assert "aeloon.science" in result.loaded

            # Command handler is callable
            record = registry.commands["sr"]
            assert callable(record.handler)

    @pytest.mark.asyncio
    async def test_deactivation_resets_status(
        self, science_plugin_dir: Path, mock_agent_loop: MagicMock
    ) -> None:
        """After shutdown, plugin status reverts to 'discovered'."""
        from aeloon.plugins.ScienceResearch.plugin import SciencePlugin

        with patch("aeloon.plugins._sdk.loader.importlib.import_module") as mock_import:
            mod = MagicMock()
            mod.SciencePlugin = SciencePlugin
            mock_import.return_value = mod

            registry = PluginRegistry()
            manager = PluginManager(
                registry=registry,
                discovery=PluginDiscovery(bundled_dir=science_plugin_dir.parent),
                loader=PluginLoader(),
                hook_dispatcher=HookDispatcher(),
                agent_loop=mock_agent_loop,
                plugin_config={"aeloon.science": {"enabled": True}},
                storage_base=Path("/tmp"),
            )
            await manager.boot()
            assert registry.get_plugin("aeloon.science").status == "active"

            await manager.shutdown()
            assert registry.get_plugin("aeloon.science").status == "discovered"

    def test_config_from_plugins_dict(self, mock_agent_loop: MagicMock) -> None:
        """Plugin config propagated from Config.plugins to PluginAPI."""

        registry = PluginRegistry()
        plugin_cfg = {"enabled": True, "default_budget_tokens": 99999}
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.science",
            config=plugin_cfg,
            storage_base=Path("/tmp"),
        )
        api = PluginAPI(
            plugin_id="aeloon.science",
            version="0.1.0",
            config=plugin_cfg,
            runtime=runtime,
            registry=registry,
        )
        assert api.config["default_budget_tokens"] == 99999
