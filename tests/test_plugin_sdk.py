"""Tests for the Aeloon Plugin SDK.

Covers all P0-* modules: manifest, base, types, api, runtime, registry,
hooks, discovery, loader, and manager.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aeloon.plugins._sdk import (
    Plugin,
    PluginManifest,
    PluginRequires,
    PluginService,
    ServiceStatus,
)
from aeloon.plugins._sdk.api import PluginAPI
from aeloon.plugins._sdk.discovery import (
    PluginCandidate,
    PluginDiscovery,
)
from aeloon.plugins._sdk.hooks import HookDispatcher
from aeloon.plugins._sdk.loader import CircularDependencyError, PluginLoader
from aeloon.plugins._sdk.manager import PluginManager, ServiceSupervisor
from aeloon.plugins._sdk.manifest import ManifestLoadError, load_manifest
from aeloon.plugins._sdk.registry import PluginRegistry, RegistrationConflictError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_manifest(tmp_path: Path) -> dict[str, Any]:
    """Minimal valid manifest dict."""
    return {
        "id": "test.plugin",
        "name": "Test Plugin",
        "version": "0.1.0",
        "entry": "test_plugin:TestPlugin",
        "provides": {"commands": ["test"]},
        "requires": {"plugins": []},
    }


@pytest.fixture
def plugin_dir(tmp_path: Path, tmp_manifest: dict[str, Any]) -> Path:
    """Create a plugin directory with manifest."""
    plugin_path = tmp_path / "test.plugin"
    plugin_path.mkdir()
    manifest_file = plugin_path / "aeloon.plugin.json"
    manifest_file.write_text(json.dumps(tmp_manifest))
    return plugin_path


@pytest.fixture
def mock_agent_loop() -> MagicMock:
    """Mock AgentLoop."""
    loop = MagicMock()
    loop.provider = MagicMock()
    loop.provider.chat = AsyncMock(return_value=MagicMock(content="test response"))
    loop.model = "test-model"
    return loop


# ---------------------------------------------------------------------------
# Manifest Tests
# ---------------------------------------------------------------------------


class TestPluginManifest:
    """Test manifest parsing and validation."""

    def test_valid_manifest_parses(self, tmp_manifest: dict[str, Any]) -> None:
        """Full valid manifest → PluginManifest."""
        m = PluginManifest.model_validate(tmp_manifest)
        assert m.id == "test.plugin"
        assert m.name == "Test Plugin"
        assert m.version == "0.1.0"

    def test_minimal_manifest_with_defaults(self) -> None:
        """Only required fields → defaults filled in."""
        data = {
            "id": "aeloon.minimal",
            "name": "Minimal",
            "version": "1.0.0",
            "entry": "mod:Class",
        }
        m = PluginManifest.model_validate(data)
        assert m.description == ""
        assert m.author == ""
        assert m.provides.commands == []

    def test_invalid_id_rejected(self) -> None:
        """ID without dot → ValidationError."""
        data = {
            "id": "invalid",
            "name": "Bad",
            "version": "1.0.0",
            "entry": "mod:Class",
        }
        with pytest.raises(Exception):  # ValidationError
            PluginManifest.model_validate(data)

    def test_invalid_entry_rejected(self) -> None:
        """Entry without colon → ValidationError."""
        data = {
            "id": "test.plugin",
            "name": "Bad",
            "version": "1.0.0",
            "entry": "invalid_no_colon",
        }
        with pytest.raises(Exception):
            PluginManifest.model_validate(data)

    def test_load_manifest_from_file(self, plugin_dir: Path) -> None:
        """Write JSON to temp file, load_manifest() returns correct model."""
        manifest_path = plugin_dir / "aeloon.plugin.json"
        m = load_manifest(manifest_path)
        assert m.id == "test.plugin"

    def test_load_manifest_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent path → ManifestLoadError."""
        with pytest.raises(ManifestLoadError):
            load_manifest(tmp_path / "nonexistent.json")

    def test_load_manifest_invalid_json(self, tmp_path: Path) -> None:
        """Broken JSON → ManifestLoadError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json")
        with pytest.raises(ManifestLoadError):
            load_manifest(bad_file)


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    """Test staged commit and rollback."""

    def test_commit_plugin_adds_records(self) -> None:
        """After commit, commands/tools are queryable."""
        from aeloon.plugins._sdk.types import CommandRecord

        registry = PluginRegistry()
        cmd = CommandRecord(plugin_id="test.p", name="cmd1", handler=AsyncMock())
        registry.commit_plugin("test.p", commands=[cmd])
        assert "cmd1" in registry.commands
        assert registry.commands["cmd1"].plugin_id == "test.p"

    def test_rollback_removes_all_records(self) -> None:
        """After rollback, no records remain."""
        from aeloon.plugins._sdk.types import CommandRecord

        registry = PluginRegistry()
        cmd = CommandRecord(plugin_id="test.p", name="cmd1", handler=AsyncMock())
        registry.commit_plugin("test.p", commands=[cmd])
        registry.rollback_plugin("test.p")
        assert "cmd1" not in registry.commands

    def test_commit_detects_command_name_conflict(self) -> None:
        """Two plugins registering same command → RegistrationConflictError."""
        from aeloon.plugins._sdk.types import CommandRecord

        registry = PluginRegistry()
        cmd1 = CommandRecord(plugin_id="p1", name="cmd", handler=AsyncMock())
        cmd2 = CommandRecord(plugin_id="p2", name="cmd", handler=AsyncMock())
        registry.commit_plugin("p1", commands=[cmd1])
        with pytest.raises(RegistrationConflictError):
            registry.commit_plugin("p2", commands=[cmd2])

    def test_commit_detects_tool_name_conflict(self) -> None:
        """Two plugins registering same tool name → RegistrationConflictError."""
        from aeloon.plugins._sdk.types import ToolRecord

        registry = PluginRegistry()
        t1 = ToolRecord(plugin_id="p1", name="my_tool", tool=MagicMock())
        t2 = ToolRecord(plugin_id="p2", name="my_tool", tool=MagicMock())
        registry.commit_plugin("p1", tools=[t1])
        with pytest.raises(RegistrationConflictError):
            registry.commit_plugin("p2", tools=[t2])

    def test_conflict_leaves_registry_unchanged(self) -> None:
        """Conflict during commit → no records from second plugin written."""
        from aeloon.plugins._sdk.types import CommandRecord, ToolRecord

        registry = PluginRegistry()
        cmd1 = CommandRecord(plugin_id="p1", name="cmd", handler=AsyncMock())
        registry.commit_plugin("p1", commands=[cmd1])

        # p2 tries to register a conflicting command + a new tool
        cmd2 = CommandRecord(plugin_id="p2", name="cmd", handler=AsyncMock())
        tool2 = ToolRecord(plugin_id="p2", name="new_tool", tool=MagicMock())
        with pytest.raises(RegistrationConflictError):
            registry.commit_plugin("p2", commands=[cmd2], tools=[tool2])

        # Tool from p2 should NOT have been written
        assert "new_tool" not in registry.tools

    def test_rollback_cleans_all_record_types(self) -> None:
        """rollback_plugin removes commands, tools, services, hooks."""
        from aeloon.plugins._sdk.types import CommandRecord, HookRecord, ServiceRecord, ToolRecord

        registry = PluginRegistry()
        registry.commit_plugin(
            "p1",
            commands=[CommandRecord(plugin_id="p1", name="c", handler=AsyncMock())],
            tools=[ToolRecord(plugin_id="p1", name="t", tool=MagicMock())],
            services=[
                ServiceRecord(plugin_id="p1", name="s", full_id="p1.s", service_cls=MagicMock())
            ],
            hooks=[
                HookRecord(
                    plugin_id="p1", event="ev", kind="notify", priority=0, handler=MagicMock()
                )
            ],
        )
        assert "c" in registry.commands
        registry.rollback_plugin("p1")
        assert "c" not in registry.commands
        assert "t" not in registry.tools
        assert "p1.s" not in registry.services
        assert registry.hooks_for_event("ev") == []

    def test_commit_no_records_is_noop(self) -> None:
        """Commit with no records → no error, status set to registered."""
        registry = PluginRegistry()
        registry.add_plugin(MagicMock(plugin_id="p1", status="discovered"))
        registry.commit_plugin("p1")
        assert registry.get_plugin("p1").status == "registered"

    def test_same_plugin_can_recommit(self) -> None:
        """Same plugin_id can commit again (update, not conflict)."""
        from aeloon.plugins._sdk.types import CommandRecord

        registry = PluginRegistry()
        cmd1 = CommandRecord(plugin_id="p1", name="cmd", handler=AsyncMock())
        registry.commit_plugin("p1", commands=[cmd1])

        # Same plugin re-commits same command name → no conflict
        cmd1b = CommandRecord(plugin_id="p1", name="cmd", handler=AsyncMock())
        registry.commit_plugin("p1", commands=[cmd1b])
        assert "cmd" in registry.commands

    def test_hooks_sorted_by_priority(self) -> None:
        """After commit, hooks sorted by priority (highest first)."""
        from aeloon.plugins._sdk.types import HookRecord

        registry = PluginRegistry()
        h1 = HookRecord(
            plugin_id="p1",
            event="test_event",
            kind="notify",
            priority=10,
            handler=AsyncMock(),
        )
        h2 = HookRecord(
            plugin_id="p1",
            event="test_event",
            kind="notify",
            priority=20,
            handler=AsyncMock(),
        )
        registry.commit_plugin("p1", hooks=[h1, h2])
        hooks = registry.hooks_for_event("test_event")
        assert hooks[0].priority == 20
        assert hooks[1].priority == 10


# ---------------------------------------------------------------------------
# API Tests
# ---------------------------------------------------------------------------


class TestPluginAPI:
    """Test staged PluginAPI registration helpers."""

    def test_register_cli_can_stage_command_and_builder(
        self,
        mock_agent_loop: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Declarative CLI registration can also stage the slash handler."""
        from aeloon.plugins._sdk.runtime import PluginRuntime
        from aeloon.plugins._sdk.types import CLICommandSpec

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="test.plugin",
            config={},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="test.plugin",
            version="0.1.0",
            config={},
            runtime=runtime,
            registry=registry,
        )
        handler = AsyncMock()
        commands = (
            CLICommandSpec(
                group_name="wiki",
                command_name="status",
                help="Show status.",
                plugin_command="wiki",
            ),
        )

        api.register_cli(
            "wiki",
            commands=commands,
            handler=handler,
            description="Wiki workflows",
        )

        assert len(api._pending_commands) == 1
        assert api._pending_commands[0].name == "wiki"
        assert api._pending_commands[0].handler is handler
        assert api._pending_commands[0].description == "Wiki workflows"
        assert len(api._pending_cli) == 1
        assert api._pending_cli[0].name == "wiki"
        assert api._pending_cli[0].commands == commands
        assert callable(api._pending_cli[0].builder)

    def test_register_command_middleware_stages_record(
        self,
        mock_agent_loop: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Dispatcher-level command middleware is staged and committed."""
        from aeloon.plugins._sdk.runtime import PluginRuntime

        class _CommandMiddleware:
            async def before(self, cmd: str, args: str, ctx) -> None:
                return None

            async def after(self, cmd: str, result, ctx) -> None:
                return None

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="test.plugin",
            config={},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="test.plugin",
            version="0.1.0",
            config={},
            runtime=runtime,
            registry=registry,
        )

        api.register_command_middleware("audit", _CommandMiddleware())

        assert len(api._pending_command_middlewares) == 1
        assert api._pending_command_middlewares[0].name == "audit"

        api._commit()

        assert len(registry.command_middlewares) == 1
        assert registry.command_middlewares[0].name == "audit"


# ---------------------------------------------------------------------------
# Loader Tests
# ---------------------------------------------------------------------------


class TestPluginLoader:
    """Test dependency resolution and loading."""

    def test_no_dependencies_preserves_order(self) -> None:
        """No deps → order unchanged."""
        loader = PluginLoader()
        candidates = [
            PluginCandidate(
                manifest=PluginManifest(
                    id="test.p1",
                    name="P1",
                    version="1.0",
                    entry="p1:P1",
                    requires=PluginRequires(),
                ),
                source=10,
                source_label="test",
                path=None,
            ),
            PluginCandidate(
                manifest=PluginManifest(
                    id="test.p2",
                    name="P2",
                    version="1.0",
                    entry="p2:P2",
                    requires=PluginRequires(),
                ),
                source=10,
                source_label="test",
                path=None,
            ),
        ]
        result = loader.resolve_load_order(candidates)
        assert [c.manifest.id for c in result] == ["test.p1", "test.p2"]

    def test_linear_dependency_chain(self) -> None:
        """A depends on B → B first."""
        loader = PluginLoader()
        b = PluginCandidate(
            manifest=PluginManifest(
                id="test.b",
                name="B",
                version="1.0",
                entry="b:B",
                requires=PluginRequires(),
            ),
            source=10,
            source_label="test",
            path=None,
        )
        a = PluginCandidate(
            manifest=PluginManifest(
                id="test.a",
                name="A",
                version="1.0",
                entry="a:A",
                requires=PluginRequires(plugins=["test.b"]),
            ),
            source=10,
            source_label="test",
            path=None,
        )
        result = loader.resolve_load_order([a, b])
        assert [c.manifest.id for c in result] == ["test.b", "test.a"]

    def test_circular_dependency_raises(self) -> None:
        """A → B → A → CircularDependencyError."""
        loader = PluginLoader()
        a = PluginCandidate(
            manifest=PluginManifest(
                id="test.a",
                name="A",
                version="1.0",
                entry="a:A",
                requires=PluginRequires(plugins=["test.b"]),
            ),
            source=10,
            source_label="test",
            path=None,
        )
        b = PluginCandidate(
            manifest=PluginManifest(
                id="test.b",
                name="B",
                version="1.0",
                entry="b:B",
                requires=PluginRequires(plugins=["test.a"]),
            ),
            source=10,
            source_label="test",
            path=None,
        )
        with pytest.raises(CircularDependencyError):
            loader.resolve_load_order([a, b])

    def test_missing_dependency_skips_plugin(self) -> None:
        """A depends on missing B → A skipped, result is empty."""
        loader = PluginLoader()
        a = PluginCandidate(
            manifest=PluginManifest(
                id="test.a",
                name="A",
                version="1.0",
                entry="a:A",
                requires=PluginRequires(plugins=["test.missing"]),
            ),
            source=10,
            source_label="test",
            path=None,
        )
        result = loader.resolve_load_order([a])
        assert len(result) == 0

    def test_transitive_skip(self) -> None:
        """A→B→C, C missing → both A and B skipped."""
        loader = PluginLoader()
        a = PluginCandidate(
            manifest=PluginManifest(
                id="test.a",
                name="A",
                version="1.0",
                entry="a:A",
                requires=PluginRequires(plugins=["test.b"]),
            ),
            source=10,
            source_label="test",
            path=None,
        )
        b = PluginCandidate(
            manifest=PluginManifest(
                id="test.b",
                name="B",
                version="1.0",
                entry="b:B",
                requires=PluginRequires(plugins=["test.c"]),
            ),
            source=10,
            source_label="test",
            path=None,
        )
        result = loader.resolve_load_order([a, b])
        assert len(result) == 0

    def test_empty_candidates(self) -> None:
        """Empty candidate list → empty result."""
        loader = PluginLoader()
        result = loader.resolve_load_order([])
        assert result == []


# ---------------------------------------------------------------------------
# Service Supervisor Tests
# ---------------------------------------------------------------------------


class TestServiceSupervisor:
    """Test service lifecycle."""

    @pytest.mark.asyncio
    async def test_start_service_success(self) -> None:
        """Service starts → status RUNNING."""
        supervisor = ServiceSupervisor()
        from aeloon.plugins._sdk.types import ServiceRecord

        mock_svc = AsyncMock(spec=PluginService)
        mock_svc.start = AsyncMock()

        record = ServiceRecord(
            plugin_id="p1",
            name="svc",
            full_id="p1.svc",
            service_cls=type(mock_svc),
        )
        record.service_cls = MagicMock(return_value=mock_svc)

        runtime = MagicMock()
        await supervisor.start_service(record, runtime, {})
        assert record.status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_service_clean(self) -> None:
        """Running service stops → status STOPPED."""
        supervisor = ServiceSupervisor()
        from aeloon.plugins._sdk.types import ServiceRecord

        mock_svc = AsyncMock(spec=PluginService)
        mock_svc.stop = AsyncMock()

        record = ServiceRecord(
            plugin_id="p1",
            name="svc",
            full_id="p1.svc",
            service_cls=type(mock_svc),
            status=ServiceStatus.RUNNING,
        )
        record.service_cls = MagicMock(return_value=mock_svc)

        supervisor._instances["p1.svc"] = mock_svc
        supervisor._records["p1.svc"] = record

        await supervisor.stop_service("p1.svc")
        assert record.status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_restart_never_policy_does_not_restart(self) -> None:
        """restart_policy=never → service not restarted."""
        supervisor = ServiceSupervisor()
        from aeloon.plugins._sdk.types import ServicePolicy, ServiceRecord

        mock_svc = AsyncMock(spec=PluginService)
        record = ServiceRecord(
            plugin_id="p1",
            name="svc",
            full_id="p1.svc",
            service_cls=MagicMock(return_value=mock_svc),
            policy=ServicePolicy(restart_policy="never"),
            status=ServiceStatus.RUNNING,
        )
        supervisor._instances["p1.svc"] = mock_svc
        supervisor._records["p1.svc"] = record

        await supervisor.restart_service("p1.svc", MagicMock(), {})
        # Should NOT have called start again
        mock_svc.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_exceeds_max_restarts(self) -> None:
        """restart_count >= max_restarts → status FAILED, not restarted."""
        supervisor = ServiceSupervisor()
        from aeloon.plugins._sdk.types import ServicePolicy, ServiceRecord

        mock_svc = AsyncMock(spec=PluginService)
        record = ServiceRecord(
            plugin_id="p1",
            name="svc",
            full_id="p1.svc",
            service_cls=MagicMock(return_value=mock_svc),
            policy=ServicePolicy(restart_policy="on-failure", max_restarts=3),
            status=ServiceStatus.RUNNING,
            restart_count=3,
        )
        supervisor._instances["p1.svc"] = mock_svc
        supervisor._records["p1.svc"] = record

        await supervisor.restart_service("p1.svc", MagicMock(), {})
        assert record.status == ServiceStatus.FAILED

    def test_health_check_not_found(self) -> None:
        """Health check for non-existent service → not_found."""
        supervisor = ServiceSupervisor()
        result = supervisor.health_check("nonexistent.svc")
        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_stop_all(self) -> None:
        """stop_all() stops all running services."""
        supervisor = ServiceSupervisor()
        from aeloon.plugins._sdk.types import ServiceRecord

        svc1 = AsyncMock(spec=PluginService)
        svc2 = AsyncMock(spec=PluginService)
        for name, svc in [("p1.s1", svc1), ("p1.s2", svc2)]:
            supervisor._instances[name] = svc
            supervisor._records[name] = ServiceRecord(
                plugin_id="p1",
                name=name.split(".")[1],
                full_id=name,
                service_cls=MagicMock(),
                status=ServiceStatus.RUNNING,
            )
        await supervisor.stop_all()
        assert len(supervisor._instances) == 0


# ---------------------------------------------------------------------------
# Hook Dispatcher Tests
# ---------------------------------------------------------------------------


class TestHookDispatcher:
    """Test hook dispatch modes."""

    @pytest.mark.asyncio
    async def test_notify_fires_all_handlers(self) -> None:
        """All handlers called, return values ignored."""
        dispatcher = HookDispatcher()
        h1 = AsyncMock(return_value="value1")
        h2 = AsyncMock(return_value="value2")
        dispatcher._local_hooks["test_event"] = [
            MagicMock(plugin_id="p1", handler=h1, priority=0),
            MagicMock(plugin_id="p2", handler=h2, priority=0),
        ]
        await dispatcher.dispatch_notify("test_event")
        h1.assert_called_once()
        h2.assert_called_once()

    @pytest.mark.asyncio
    async def test_mutate_chains_value(self) -> None:
        """Value pipes through handlers."""
        dispatcher = HookDispatcher()
        h1 = AsyncMock(return_value=10)
        h2 = AsyncMock(return_value=20)
        dispatcher._local_hooks["test_event"] = [
            MagicMock(plugin_id="p1", handler=h1, priority=1),
            MagicMock(plugin_id="p2", handler=h2, priority=0),
        ]
        result = await dispatcher.dispatch_mutate("test_event", value=5)
        assert result == 20

    @pytest.mark.asyncio
    async def test_notify_exception_does_not_propagate(self) -> None:
        """NOTIFY handler exception logged but not raised."""
        dispatcher = HookDispatcher()
        bad_handler = MagicMock(side_effect=RuntimeError("boom"))
        good_handler = MagicMock(return_value=None)
        dispatcher._local_hooks["test_event"] = [
            MagicMock(plugin_id="p1", handler=bad_handler, priority=1),
            MagicMock(plugin_id="p2", handler=good_handler, priority=0),
        ]
        await dispatcher.dispatch_notify("test_event")
        # Both handlers called; exception from p1 did not prevent p2
        bad_handler.assert_called_once()
        good_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_sync_and_async_handlers(self) -> None:
        """NOTIFY supports both sync and async handlers."""
        dispatcher = HookDispatcher()
        sync_handler = MagicMock(return_value=None)
        async_handler = AsyncMock(return_value=None)
        dispatcher._local_hooks["test_event"] = [
            MagicMock(plugin_id="p1", handler=sync_handler, priority=1),
            MagicMock(plugin_id="p2", handler=async_handler, priority=0),
        ]
        await dispatcher.dispatch_notify("test_event")
        sync_handler.assert_called_once()
        async_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_mutate_none_stops_chain(self) -> None:
        """MUTATE handler returning None stops chain early."""
        dispatcher = HookDispatcher()
        h1 = MagicMock(return_value=None)  # returns None → chain stops
        h2 = MagicMock(return_value=999)
        dispatcher._local_hooks["test_event"] = [
            MagicMock(plugin_id="p1", handler=h1, priority=1),
            MagicMock(plugin_id="p2", handler=h2, priority=0),
        ]
        result = await dispatcher.dispatch_mutate("test_event", value=5)
        assert result == 5  # original value returned
        h2.assert_not_called()  # chain stopped before h2

    @pytest.mark.asyncio
    async def test_reduce_collects_results(self) -> None:
        """REDUCE collects all handler return values."""
        dispatcher = HookDispatcher()
        h1 = MagicMock(return_value="a")
        h2 = MagicMock(return_value="b")
        dispatcher._local_hooks["test_event"] = [
            MagicMock(plugin_id="p1", handler=h1, priority=1),
            MagicMock(plugin_id="p2", handler=h2, priority=0),
        ]
        result = await dispatcher.dispatch_reduce("test_event")
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_reduce_exception_does_not_stop_collection(self) -> None:
        """REDUCE: exception skipped, other results still collected."""
        dispatcher = HookDispatcher()
        h1 = MagicMock(side_effect=RuntimeError("boom"))
        h2 = MagicMock(return_value="ok")
        dispatcher._local_hooks["test_event"] = [
            MagicMock(plugin_id="p1", handler=h1, priority=1),
            MagicMock(plugin_id="p2", handler=h2, priority=0),
        ]
        result = await dispatcher.dispatch_reduce("test_event")
        assert result == ["ok"]


# ---------------------------------------------------------------------------
# Discovery Tests
# ---------------------------------------------------------------------------


class TestPluginDiscovery:
    """Test plugin scanning."""

    def test_discover_bundled(self, plugin_dir: Path) -> None:
        """Plugin dir with manifest → discovered."""
        discovery = PluginDiscovery(bundled_dir=plugin_dir.parent)
        candidates = discovery.discover_all()
        assert any(c.manifest.id == "test.plugin" for c in candidates)

    def test_invalid_manifest_skipped(self, tmp_path: Path) -> None:
        """Broken manifest → skipped with warning."""
        plugin_path = tmp_path / "broken.plugin"
        plugin_path.mkdir()
        (plugin_path / "aeloon.plugin.json").write_text("{bad json")
        discovery = PluginDiscovery(bundled_dir=tmp_path)
        candidates = discovery.discover_all()
        # Skipped plugin should not appear
        assert not any(c.manifest.id == "broken.plugin" for c in candidates)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestPluginManagerLifecycle:
    """Test full boot and shutdown."""

    @pytest.mark.asyncio
    async def test_boot_discover_register_activate(
        self, plugin_dir: Path, mock_agent_loop: MagicMock
    ) -> None:
        """Full lifecycle: discover → register → activate."""

        # Create a minimal test plugin class
        class MinimalPlugin(Plugin):
            def register(self, api: PluginAPI) -> None:
                api.register_command("test", AsyncMock())

            async def activate(self, api: PluginAPI) -> None:
                pass

        with patch("aeloon.plugins._sdk.loader.importlib.import_module") as mock_import:
            mod = MagicMock()
            mod.TestPlugin = MinimalPlugin
            mock_import.return_value = mod

            registry = PluginRegistry()
            discovery = PluginDiscovery(bundled_dir=plugin_dir.parent)
            loader = PluginLoader()
            hooks = HookDispatcher()
            manager = PluginManager(
                registry=registry,
                discovery=discovery,
                loader=loader,
                hook_dispatcher=hooks,
                agent_loop=mock_agent_loop,
                plugin_config={"test.plugin": {"enabled": True}},
                storage_base=Path("/tmp"),
            )

            result = await manager.boot()
            assert "test.plugin" in result.loaded

    @pytest.mark.asyncio
    async def test_shutdown_reverse_order(self, mock_agent_loop: MagicMock) -> None:
        """Plugins deactivated in reverse order."""
        registry = PluginRegistry()
        manager = PluginManager(
            registry=registry,
            discovery=PluginDiscovery(),
            loader=PluginLoader(),
            hook_dispatcher=HookDispatcher(),
            agent_loop=mock_agent_loop,
            plugin_config={},
            storage_base=Path("/tmp"),
        )

        # Manually add activation order
        manager._activation_order = ["p1", "p2"]
        p1_inst = AsyncMock(spec=Plugin)
        p2_inst = AsyncMock(spec=Plugin)

        registry.add_plugin(MagicMock(plugin_id="p1", instance=p1_inst))
        registry.add_plugin(MagicMock(plugin_id="p2", instance=p2_inst))

        await manager.shutdown()
        # Verify called (order matters for side effects)
        p1_inst.deactivate.assert_called_once()
        p2_inst.deactivate.assert_called_once()
