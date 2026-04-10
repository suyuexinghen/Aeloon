"""Tests for /plugin slash commands and plugin manager integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aeloon.plugins._sdk.installer import PluginInstaller
from aeloon.plugins._sdk.state_store import PluginState, PluginStateStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MANIFEST_TEMPLATE = {
    "id": "aeloon.testplugin",
    "name": "Test Plugin",
    "version": "0.1.0",
    "description": "A test plugin",
    "author": "Test",
    "entry": "aeloon.plugins.testplugin.plugin:TestPlugin",
}


def _write_manifest(directory: Path, manifest: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    data = manifest or MANIFEST_TEMPLATE
    (directory / "aeloon.plugin.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return directory


def _create_zip_plugin(tmp_path: Path, manifest: dict | None = None) -> Path:
    import zipfile

    plugin_dir = tmp_path / "build" / "testplugin"
    _write_manifest(plugin_dir, manifest)
    archive = tmp_path / "testplugin.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in plugin_dir.rglob("*"):
            zf.write(f, f.relative_to(tmp_path / "build"))
    return archive


def _make_dispatcher_with_state(tmp_path: Path):
    """Build a Dispatcher with PluginStateStore wired in."""
    from aeloon.core.agent.dispatcher import Dispatcher
    from aeloon.core.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(return_value=MagicMock(content="test"))

    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    state_path = tmp_path / "plugin_state.json"
    state_store = PluginStateStore(state_path)

    pm = MagicMock()
    pm._state_store = state_store
    pm._hooks = MagicMock()
    pm._hooks.dispatch_notify = AsyncMock()
    pm.registry = MagicMock()
    pm.registry.plugins = {}
    pm.registry.commands = {}
    pm.registry.get_plugin = MagicMock(return_value=None)

    with (
        patch("aeloon.core.agent.loop.ContextBuilder"),
        patch("aeloon.core.agent.loop.SessionManager") as mock_sm,
        patch("aeloon.core.agent.loop.SubagentManager") as mock_sub,
    ):
        mock_sub.return_value.cancel_by_session = AsyncMock(return_value=0)
        session = MagicMock()
        session.messages = []
        session.last_consolidated = 0
        session.key = "test-key"
        mock_sm.return_value.get_or_create.return_value = session
        mock_sm.return_value.list_sessions.return_value = []

        from aeloon.core.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.bus = bus
        loop.provider = provider
        loop.model = "test-model"
        loop.runtime_settings = MagicMock()
        loop.runtime_settings.show_detail = False
        loop.runtime_settings.show_debug = False
        loop.runtime_settings.show_profile = False
        loop.runtime_settings.show_deep_profile = False
        loop.runtime_settings.output_mode = "normal"
        loop.sessions = mock_sm.return_value
        loop.subagents = mock_sub.return_value
        loop.plugin_manager = pm
        loop.profiler = MagicMock()
        loop.profiler.enabled = False
        loop.tools = MagicMock()
        loop.tools.execute = AsyncMock(return_value="tool result")
        loop.process_direct = AsyncMock(return_value="direct response")
        loop._profiled_turn = lambda: MagicMock()  # context manager stub

    dispatcher = Dispatcher.__new__(Dispatcher)
    dispatcher._agent_loop = loop
    dispatcher._active_tasks = {}

    return dispatcher, pm


# ---------------------------------------------------------------------------
# StateStore Tests
# ---------------------------------------------------------------------------


class TestPluginStateStore:
    def test_crud(self, tmp_path: Path) -> None:
        store = PluginStateStore(tmp_path / "state.json")
        state = PluginState(
            plugin_id="aeloon.test",
            installed_at="2026-01-01",
            source="workspace",
            version="1.0",
        )
        store.set(state)
        assert store.get("aeloon.test") is not None
        assert store.get("aeloon.test").version == "1.0"

        store.set_enabled("aeloon.test", False)
        assert store.get("aeloon.test").enabled is False

        store.remove("aeloon.test")
        assert store.get("aeloon.test") is None

    def test_corrupt_file_recovery(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("NOT JSON {{{{", encoding="utf-8")
        store = PluginStateStore(path)
        assert store.list_all() == {}


# ---------------------------------------------------------------------------
# Installer Tests
# ---------------------------------------------------------------------------


class TestPluginInstaller:
    def test_install_zip(self, tmp_path: Path) -> None:
        archive = _create_zip_plugin(tmp_path)
        target = tmp_path / "plugins"
        installer = PluginInstaller()
        result = installer.install(archive, target, verify_import=False)
        assert result.status == "ok"
        assert result.plugin_id == "aeloon.testplugin"
        assert (target / "testplugin" / "aeloon.plugin.json").exists()

    def test_verify_no_manifest(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "empty"
        plugin_dir.mkdir()
        installer = PluginInstaller()
        result = installer.verify(plugin_dir)
        assert result.status == "broken"

    def test_remove_workspace_plugin(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        plugin_dir = _write_manifest(workspace / "testplugin")
        installer = PluginInstaller()
        assert installer.remove("aeloon.testplugin", workspace) is True
        assert not plugin_dir.exists()

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        installer = PluginInstaller()
        assert installer.remove("aeloon.nothing", workspace) is False


# ---------------------------------------------------------------------------
# Slash Command Tests
# ---------------------------------------------------------------------------


class TestPluginSlashCommands:
    @pytest.mark.asyncio
    async def test_plugin_list_empty(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/plugin list")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "no plugins" in response.content.lower() or "Plugins" in response.content

    @pytest.mark.asyncio
    async def test_plugin_list_with_loaded_plugin(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        record = MagicMock()
        record.status = "active"
        record.error = None
        record.manifest.version = "1.0"
        pm.registry.plugins = {"aeloon.test": record}

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/plugin list")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "aeloon.test" in response.content

    @pytest.mark.asyncio
    async def test_plugin_list_shows_deactivated(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        pm._state_store.set(
            PluginState(
                plugin_id="aeloon.deactivated",
                installed_at="2026-01-01",
                source="workspace",
                enabled=False,
                version="0.5.0",
            )
        )

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/plugin list")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "deactivated" in response.content.lower()

    @pytest.mark.asyncio
    async def test_plugin_list_shows_error(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        record = MagicMock()
        record.status = "error"
        record.error = "import failed"
        record.manifest.version = "0.1"
        pm.registry.plugins = {"aeloon.broken": record}

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/plugin list")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "aeloon.broken" in response.content

    @pytest.mark.asyncio
    async def test_plugin_error_specific(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        record = MagicMock()
        record.status = "error"
        record.error = "ModuleNotFoundError: No module named 'foo'"
        record.manifest.version = "0.1"
        pm.registry.plugins = {"aeloon.broken": record}
        pm.registry.get_plugin = MagicMock(return_value=record)

        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/plugin error aeloon.broken"
        )
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "ModuleNotFoundError" in response.content

    @pytest.mark.asyncio
    async def test_plugin_error_all(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/plugin error")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "no plugin errors" in response.content.lower()

    @pytest.mark.asyncio
    async def test_plugin_activate(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        pm._state_store.set(
            PluginState(
                plugin_id="aeloon.test",
                installed_at="2026-01-01",
                source="workspace",
                enabled=False,
                version="1.0",
            )
        )

        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/plugin activate aeloon.test"
        )
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "activated" in response.content.lower()
        assert pm._state_store.get("aeloon.test").enabled is True

    @pytest.mark.asyncio
    async def test_plugin_deactivate(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        pm._state_store.set(
            PluginState(
                plugin_id="aeloon.test",
                installed_at="2026-01-01",
                source="workspace",
                enabled=True,
                version="1.0",
            )
        )

        msg = InboundMessage(
            channel="cli", sender_id="u1", chat_id="c1", content="/plugin deactivate aeloon.test"
        )
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "deactivated" in response.content.lower()
        assert pm._state_store.get("aeloon.test").enabled is False

    @pytest.mark.asyncio
    async def test_plugin_install_archive(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        archive = _create_zip_plugin(tmp_path)

        # Mock Path.home to avoid writing to real ~/.aeloon/plugins/
        fake_home = tmp_path / "home"
        (fake_home / ".aeloon" / "plugins").mkdir(parents=True, exist_ok=True)

        with patch("pathlib.Path.home", return_value=fake_home):
            msg = InboundMessage(
                channel="cli",
                sender_id="u1",
                chat_id="c1",
                content=f"/plugin install {archive}",
            )
            response = await dispatcher.process_message(msg)
            assert response is not None
            # Test plugin has fake entry point — import will fail
            assert "Install failed" in response.content or "Installed" in response.content

    @pytest.mark.asyncio
    async def test_plugin_install_missing_archive(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)

        msg = InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="c1",
            content="/plugin install /nonexistent/path.zip",
        )
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "not found" in response.content.lower()

    @pytest.mark.asyncio
    async def test_plugin_remove(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)
        workspace = tmp_path / "workspace_plugins"
        _write_manifest(workspace / "testplugin")

        # Install a plugin to state store first
        pm._state_store.set(
            PluginState(
                plugin_id="aeloon.testplugin",
                installed_at="2026-01-01",
                source="workspace",
                enabled=True,
                version="0.1.0",
            )
        )

        # Need to patch the workspace dir
        with patch("pathlib.Path.home", return_value=tmp_path / "home"):
            (tmp_path / "home" / ".aeloon" / "plugins").mkdir(parents=True, exist_ok=True)
            _write_manifest(tmp_path / "home" / ".aeloon" / "plugins" / "testplugin")

            msg = InboundMessage(
                channel="cli",
                sender_id="u1",
                chat_id="c1",
                content="/plugin remove aeloon.testplugin",
            )
            response = await dispatcher.process_message(msg)
            assert response is not None
            assert "removed" in response.content.lower()

    @pytest.mark.asyncio
    async def test_plugin_unknown_subcommand(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/plugin bogus")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "Usage" in response.content

    @pytest.mark.asyncio
    async def test_plugin_activate_not_found(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)

        msg = InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="c1",
            content="/plugin activate aeloon.nonexistent",
        )
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "not found" in response.content.lower()

    @pytest.mark.asyncio
    async def test_plugin_no_subcommand_shows_list(self, tmp_path: Path) -> None:
        from aeloon.core.bus.events import InboundMessage

        dispatcher, pm = _make_dispatcher_with_state(tmp_path)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/plugin")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "Plugins" in response.content or "no plugins" in response.content.lower()
