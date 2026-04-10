"""Tests for AeloonFS plugin scaffold and extension points."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# FS1-5: Tool policy adapter
# ---------------------------------------------------------------------------


class TestToolPolicyAdapter:
    """Tests for aeloon.agent.tools.policy module."""

    def test_default_no_policy(self) -> None:
        from aeloon.core.agent.tools.policy import get_exec_policy, get_file_policy

        assert get_file_policy() is None
        assert get_exec_policy() is None

    def test_set_and_clear_file_policy(self) -> None:
        from aeloon.core.agent.tools.policy import get_file_policy, set_file_policy

        mock_policy = MagicMock()
        set_file_policy(mock_policy)
        assert get_file_policy() is mock_policy
        set_file_policy(None)
        assert get_file_policy() is None

    def test_set_and_clear_exec_policy(self) -> None:
        from aeloon.core.agent.tools.policy import get_exec_policy, set_exec_policy

        mock_policy = MagicMock()
        set_exec_policy(mock_policy)
        assert get_exec_policy() is mock_policy
        set_exec_policy(None)
        assert get_exec_policy() is None


# ---------------------------------------------------------------------------
# FS2-1: Plugin manifest
# ---------------------------------------------------------------------------


class TestFsPluginManifest:
    """Tests for aeloon.plugin discovery and manifest validation."""

    def test_manifest_is_valid_json(self) -> None:
        manifest_path = (
            Path(__file__).parent.parent
            / "aeloon"
            / "plugins"
            / "FilesystemSnapshot"
            / "aeloon.plugin.json"
        )
        assert manifest_path.exists(), "Manifest file missing"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["id"] == "aeloon.fs"
        assert data["entry"] == "aeloon.plugins.FilesystemSnapshot.plugin:FsPlugin"
        assert "fs" in data["provides"]["commands"]
        assert "snapshot_control" in data["provides"]["tools"]
        assert "snapshot_service" in data["provides"]["services"]
        assert "audit_buffer" in data["provides"]["services"]

    def test_entry_point_importable(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.plugin import FsPlugin

        assert FsPlugin is not None


# ---------------------------------------------------------------------------
# FS2-3: Config schema
# ---------------------------------------------------------------------------


class TestFsConfig:
    """Tests for aeloon.plugins.FilesystemSnapshot.config.FsConfig."""

    def test_default_config(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.config import FsConfig

        cfg = FsConfig()
        assert cfg.enabled is False
        assert cfg.snapshot.enabled is False
        assert cfg.snapshot.backend == "local"
        assert cfg.audit.enabled is False
        assert cfg.audit.backend == "sqlite"
        assert cfg.sandbox.enabled is False

    def test_config_from_dict(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.config import FsConfig

        cfg = FsConfig(
            enabled=True,
            snapshot={"enabled": True, "backend": "btrfs"},
        )
        assert cfg.enabled is True
        assert cfg.snapshot.enabled is True
        assert cfg.snapshot.backend == "btrfs"

    def test_config_extra_fields_ignored(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.config import FsConfig

        cfg = FsConfig(enabled=True, future_setting="some-value")
        assert cfg.enabled is True


# ---------------------------------------------------------------------------
# FS2-4: /fs command routing
# ---------------------------------------------------------------------------


class TestFsCommand:
    """Tests for /fs command handler."""

    def _make_plugin(self) -> MagicMock:
        plugin = MagicMock()
        plugin.health_check.return_value = {"status": "ok", "services": 0}
        plugin._api = None
        plugin._services = {}
        return plugin

    def test_help_subcommand(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.commands import handle_fs_command

        plugin = self._make_plugin()
        ctx = MagicMock()
        result = handle_fs_command(plugin, ctx, "help")
        assert "AeloonFS" in result
        assert "/fs status" in result

    def test_no_args_returns_help(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.commands import handle_fs_command

        plugin = self._make_plugin()
        ctx = MagicMock()
        result = handle_fs_command(plugin, ctx, "")
        assert "AeloonFS" in result

    def test_status_no_services(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.commands import handle_fs_command

        plugin = self._make_plugin()
        ctx = MagicMock()
        result = handle_fs_command(plugin, ctx, "status")
        assert "AeloonFS" in result

    def test_unknown_subcommand(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.commands import handle_fs_command

        plugin = self._make_plugin()
        ctx = MagicMock()
        result = handle_fs_command(plugin, ctx, "rollback")
        assert "Unknown" in result or "unknown" in result.lower()

    def test_snapshots_no_service(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.commands import handle_fs_command

        plugin = self._make_plugin()
        ctx = MagicMock()
        result = handle_fs_command(plugin, ctx, "snapshots")
        assert "not available" in result.lower()

    def test_audit_no_service(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.commands import handle_fs_command

        plugin = self._make_plugin()
        ctx = MagicMock()
        result = handle_fs_command(plugin, ctx, "audit")
        assert "not available" in result.lower()


# ---------------------------------------------------------------------------
# FS2-6: Service lifecycle
# ---------------------------------------------------------------------------


class TestSnapshotService:
    """Tests for SnapshotService."""

    async def test_start_stop(self, tmp_path: Path) -> None:
        from aeloon.plugins.FilesystemSnapshot.services import SnapshotService

        svc = SnapshotService()
        runtime = MagicMock()
        runtime.storage_path = tmp_path
        assert svc.status.value == "stopped"

        await svc.start(runtime, {"snapshot": {"max_snapshots": 10}})
        assert svc.status.value == "running"

        health = svc.health_check()
        assert health["status"] == "running"
        assert health["snapshots"] == 0

        await svc.stop()
        assert svc.status.value == "stopped"

    async def test_create_snapshot_file(self, tmp_path: Path) -> None:
        from aeloon.plugins.FilesystemSnapshot.services import SnapshotService

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        svc = SnapshotService()
        runtime = MagicMock()
        runtime.storage_path = tmp_path / "storage"
        await svc.start(runtime, {})

        result = await svc.create_snapshot(str(test_file), label="test")
        assert result.get("ok") is True
        assert "snapshot_id" in result

        snapshots = svc.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["label"] == "test"

        await svc.stop()

    async def test_create_snapshot_nonexistent(self, tmp_path: Path) -> None:
        from aeloon.plugins.FilesystemSnapshot.services import SnapshotService

        svc = SnapshotService()
        runtime = MagicMock()
        runtime.storage_path = tmp_path
        await svc.start(runtime, {})

        result = await svc.create_snapshot("/nonexistent/path/file.txt")
        assert "error" in result

        await svc.stop()


class TestAuditBufferService:
    """Tests for AuditBufferService."""

    async def test_start_stop(self, tmp_path: Path) -> None:
        from aeloon.plugins.FilesystemSnapshot.services import AuditBufferService

        svc = AuditBufferService()
        runtime = MagicMock()
        runtime.storage_path = tmp_path
        assert svc.status.value == "stopped"

        await svc.start(runtime, {})
        assert svc.status.value == "running"

        health = svc.health_check()
        assert health["status"] == "running"
        assert health["buffer_size"] == 0

        await svc.stop()
        assert svc.status.value == "stopped"

    async def test_record_and_flush(self, tmp_path: Path) -> None:
        from aeloon.plugins.FilesystemSnapshot.services import AuditBufferService

        svc = AuditBufferService()
        runtime = MagicMock()
        runtime.storage_path = tmp_path
        await svc.start(runtime, {})

        svc.record("write", "/tmp/test.py", result_summary="OK")
        svc.record("exec", "ls", result_summary="file1\nfile2")

        assert svc.health_check()["buffer_size"] == 2

        # Force flush
        svc._flush_buffer()
        assert svc.health_check()["buffer_size"] == 0

        # Verify data in SQLite
        assert svc._db is not None
        rows = svc._db.execute("SELECT COUNT(*) FROM audit_log").fetchone()
        assert rows[0] == 2

        await svc.stop()


# ---------------------------------------------------------------------------
# FS2-2: Plugin registration
# ---------------------------------------------------------------------------


class TestFsPluginRegistration:
    """Tests for FsPlugin.register()."""

    def test_register_calls_api(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.plugin import FsPlugin

        plugin = FsPlugin()
        api = MagicMock()
        plugin.register(api)

        api.register_command.assert_called_once_with("fs", plugin._handle_command, description=ANY)
        api.register_cli.assert_called_once_with("fs", plugin._build_cli)
        api.register_config_schema.assert_called_once()
        api.register_tool.assert_called_once()
        assert api.register_service.call_count == 2

    async def test_deactivate_clears_policies(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.plugin import FsPlugin

        from aeloon.core.agent.tools.policy import get_file_policy, set_file_policy

        mock_policy = MagicMock()
        set_file_policy(mock_policy)
        assert get_file_policy() is mock_policy

        plugin = FsPlugin()
        plugin._api = MagicMock()
        await plugin.deactivate()

        assert get_file_policy() is None


# ---------------------------------------------------------------------------
# FS3-4: SnapshotControlTool
# ---------------------------------------------------------------------------


class TestSnapshotControlTool:
    """Tests for the snapshot_control tool."""

    def test_tool_properties(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.plugin import FsPlugin
        from aeloon.plugins.FilesystemSnapshot.tools import SnapshotControlTool

        plugin = FsPlugin()
        tool = SnapshotControlTool(plugin=plugin)
        assert tool.name == "snapshot_control"
        params = tool.parameters
        assert "action" in params["properties"]
        assert params["required"] == ["action"]

    async def test_execute_status_no_service(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.plugin import FsPlugin
        from aeloon.plugins.FilesystemSnapshot.tools import SnapshotControlTool

        plugin = FsPlugin()
        plugin._services = {}
        tool = SnapshotControlTool(plugin=plugin)
        result = await tool.execute(action="status")
        data = json.loads(result)
        assert data["status"] == "not_available"

    async def test_execute_list_no_service(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.plugin import FsPlugin
        from aeloon.plugins.FilesystemSnapshot.tools import SnapshotControlTool

        plugin = FsPlugin()
        plugin._services = {}
        tool = SnapshotControlTool(plugin=plugin)
        result = await tool.execute(action="list")
        data = json.loads(result)
        assert "error" in data

    async def test_execute_create_no_path(self) -> None:
        from aeloon.plugins.FilesystemSnapshot.plugin import FsPlugin
        from aeloon.plugins.FilesystemSnapshot.tools import SnapshotControlTool

        plugin = FsPlugin()
        mock_svc = MagicMock()
        plugin._services = {"snapshot_service": mock_svc}
        tool = SnapshotControlTool(plugin=plugin)
        result = await tool.execute(action="create")
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# Disabled-path: plugin absent → tools still work normally
# ---------------------------------------------------------------------------


class TestDisabledPathNonRegression:
    """Verify tools work identically when no plugin is loaded."""

    async def test_write_file_no_policy(self, tmp_path: Path) -> None:
        from aeloon.core.agent.tools.policy import set_file_policy

        set_file_policy(None)  # Ensure no policy

        from aeloon.core.agent.tools.filesystem import WriteFileTool

        tool = WriteFileTool()
        test_file = tmp_path / "test.txt"
        result = await tool.execute(path=str(test_file), content="hello")
        assert "Successfully" in result
        assert test_file.read_text() == "hello"

    async def test_edit_file_no_policy(self, tmp_path: Path) -> None:
        from aeloon.core.agent.tools.policy import set_file_policy

        set_file_policy(None)

        from aeloon.core.agent.tools.filesystem import EditFileTool, WriteFileTool

        write = WriteFileTool()
        test_file = tmp_path / "test.txt"
        await write.execute(path=str(test_file), content="hello world")

        edit = EditFileTool()
        result = await edit.execute(path=str(test_file), old_text="hello", new_text="goodbye")
        assert "Successfully" in result

    async def test_exec_no_policy(self) -> None:
        from aeloon.core.agent.tools.policy import set_exec_policy

        set_exec_policy(None)

        from aeloon.core.agent.tools.shell import ExecTool

        tool = ExecTool()
        result = await tool.execute(command="echo hello")
        assert "hello" in result

    async def test_write_file_with_veto_policy(self, tmp_path: Path) -> None:
        from aeloon.core.agent.tools.policy import set_file_policy

        mock_policy = AsyncMock()
        mock_policy.before_operation.return_value = "vetoed by policy"
        set_file_policy(mock_policy)

        try:
            from aeloon.core.agent.tools.filesystem import WriteFileTool

            tool = WriteFileTool()
            test_file = tmp_path / "blocked.txt"
            result = await tool.execute(path=str(test_file), content="should not write")
            assert "vetoed" in result
            assert not test_file.exists()
        finally:
            set_file_policy(None)

    async def test_write_file_with_passing_policy(self, tmp_path: Path) -> None:
        from aeloon.core.agent.tools.policy import set_file_policy

        mock_policy = AsyncMock()
        mock_policy.before_operation.return_value = None
        mock_policy.after_operation.return_value = "modified result"
        set_file_policy(mock_policy)

        try:
            from aeloon.core.agent.tools.filesystem import WriteFileTool

            tool = WriteFileTool()
            test_file = tmp_path / "allowed.txt"
            result = await tool.execute(path=str(test_file), content="data")
            assert result == "modified result"
            assert test_file.exists()
        finally:
            set_file_policy(None)


# ---------------------------------------------------------------------------
# FS1-1b: Hook dispatch in dispatcher
# ---------------------------------------------------------------------------


class TestMessageLifecycleHooks:
    """Tests for MESSAGE_RECEIVED and MESSAGE_SENT hook dispatch."""

    async def test_message_received_hook_dispatched(self) -> None:
        from aeloon.core.agent.dispatcher import Dispatcher
        from aeloon.core.bus.events import InboundMessage

        # Mock agent loop
        agent_loop = MagicMock()
        agent_loop.plugin_manager = MagicMock()
        agent_loop.plugin_manager._hooks = AsyncMock()
        agent_loop.sessions = MagicMock()
        agent_loop.runtime_settings = MagicMock()
        agent_loop.runtime_settings.show_deep_profile = False
        agent_loop.runtime_settings.show_profile = False
        agent_loop.profiler = MagicMock()
        agent_loop.profiler.enabled = False

        # Mock the process_turn to return immediately
        agent_loop.process_turn = AsyncMock(return_value="test response")

        dispatcher = Dispatcher(agent_loop)

        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="test",
            content="Hello world",
        )

        # process_message should call MESSAGE_RECEIVED hook
        await dispatcher.process_message(msg)

        # Verify hook was dispatched
        from aeloon.plugins._sdk.hooks import HookEvent

        agent_loop.plugin_manager._hooks.dispatch_notify.assert_called()
        call_args = agent_loop.plugin_manager._hooks.dispatch_notify.call_args_list[0]
        assert call_args[0][0] == HookEvent.MESSAGE_RECEIVED

    async def test_no_hook_when_no_plugin_manager(self) -> None:
        from aeloon.core.agent.dispatcher import Dispatcher
        from aeloon.core.bus.events import InboundMessage

        agent_loop = MagicMock()
        agent_loop.plugin_manager = None
        agent_loop.sessions = MagicMock()
        agent_loop.runtime_settings = MagicMock()
        agent_loop.runtime_settings.show_deep_profile = False
        agent_loop.runtime_settings.show_profile = False
        agent_loop.profiler = MagicMock()
        agent_loop.profiler.enabled = False
        agent_loop.process_turn = AsyncMock(return_value="test response")

        dispatcher = Dispatcher(agent_loop)

        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="test",
            content="Hello",
        )

        # Should not raise even without plugin_manager
        result = await dispatcher.process_message(msg)
        assert result is not None
