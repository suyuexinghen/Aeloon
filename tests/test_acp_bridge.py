"""Tests for the ACP Bridge — types, session map, config, command routing.

These tests cover the bridge's internal logic without spawning real
ACP backend processes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.plugins._sdk.acp.session import SessionMap
from aeloon.plugins._sdk.acp.types import (
    ACPError,
    ACPLayer,
    BackendProfile,
    ConnectionState,
    DelegateResult,
    SessionInfo,
)
from aeloon.plugins._sdk.api import PluginAPI
from aeloon.plugins._sdk.registry import PluginRegistry
from aeloon.plugins._sdk.runtime import PluginRuntime
from aeloon.plugins.acp_bridge.config import (
    ACPBridgeConfig,
    PolicyConfig,
    ProfileConfig,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestConnectionState:
    def test_states_exist(self) -> None:
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.ERROR.value == "error"


class TestACPError:
    def test_str_format(self) -> None:
        err = ACPError(layer=ACPLayer.TRANSPORT, message="cmd not found")
        assert str(err) == "[transport] cmd not found"

    def test_details(self) -> None:
        err = ACPError(
            layer=ACPLayer.SESSION,
            message="timeout",
            details={"session_id": "abc"},
        )
        assert err.details["session_id"] == "abc"


class TestBackendProfile:
    def test_defaults(self) -> None:
        p = BackendProfile(name="test", command=["echo"])
        assert p.cwd == "~"
        assert p.timeout_seconds == 30.0
        assert p.env == {}


class TestDelegateResult:
    def test_defaults(self) -> None:
        r = DelegateResult(content="done")
        assert r.content == "done"
        assert r.usage == {}
        assert r.execution_meta == {}


# ---------------------------------------------------------------------------
# SessionMap
# ---------------------------------------------------------------------------


class TestSessionMap:
    def test_empty(self) -> None:
        sm = SessionMap()
        assert sm.get("key") is None
        assert not sm.has("key")
        assert sm.all_sessions() == []

    def test_set_and_get(self) -> None:
        sm = SessionMap()
        info = SessionInfo(
            acp_session_id="acp-1",
            aeloon_session_key="aeloon-1",
        )
        sm.set(info)
        assert sm.has("aeloon-1")
        result = sm.get("aeloon-1")
        assert result is not None
        assert result.acp_session_id == "acp-1"

    def test_remove(self) -> None:
        sm = SessionMap()
        info = SessionInfo(
            acp_session_id="acp-1",
            aeloon_session_key="aeloon-1",
        )
        sm.set(info)
        removed = sm.remove("aeloon-1")
        assert removed is not None
        assert removed.acp_session_id == "acp-1"
        assert not sm.has("aeloon-1")

    def test_remove_missing(self) -> None:
        sm = SessionMap()
        assert sm.remove("nonexistent") is None

    def test_clear(self) -> None:
        sm = SessionMap()
        for i in range(3):
            sm.set(
                SessionInfo(
                    acp_session_id=f"acp-{i}",
                    aeloon_session_key=f"aeloon-{i}",
                )
            )
        assert len(sm.all_sessions()) == 3
        sm.clear()
        assert len(sm.all_sessions()) == 0

    def test_overwrite(self) -> None:
        sm = SessionMap()
        sm.set(SessionInfo(acp_session_id="old", aeloon_session_key="k"))
        sm.set(SessionInfo(acp_session_id="new", aeloon_session_key="k"))
        result = sm.get("k")
        assert result is not None
        assert result.acp_session_id == "new"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestACPBridgeConfig:
    def test_defaults(self) -> None:
        cfg = ACPBridgeConfig()
        assert cfg.enabled is True
        assert cfg.default_profile == "claude_code"
        assert cfg.auto_connect is False
        assert cfg.profiles == {}
        assert cfg.policy.allow_file_read is False
        assert cfg.policy.allow_file_write is False
        assert cfg.policy.allow_shell is False

    def test_custom_profiles(self) -> None:
        cfg = ACPBridgeConfig(
            profiles={
                "my_agent": ProfileConfig(
                    command=["my-agent", "acp"],
                    cwd="/tmp",
                    timeout_seconds=60,
                ),
            },
        )
        assert "my_agent" in cfg.profiles
        assert cfg.profiles["my_agent"].command == ["my-agent", "acp"]
        assert cfg.profiles["my_agent"].cwd == "/tmp"

    def test_camel_case_alias(self) -> None:
        """Config should accept camelCase keys (e.g. from TOML)."""
        cfg = ACPBridgeConfig.model_validate(
            {
                "defaultProfile": "custom",
                "autoConnect": True,
            }
        )
        assert cfg.default_profile == "custom"
        assert cfg.auto_connect is True

    def test_policy_deny_by_default(self) -> None:
        policy = PolicyConfig()
        assert policy.allow_file_read is False
        assert policy.allow_file_write is False
        assert policy.allow_shell is False
        assert policy.auto_approve_safe_requests is False


class TestSaveProfileToConfig:
    def test_creates_new_file(self, tmp_path: Path, monkeypatch) -> None:
        from aeloon.plugins.acp_bridge.config import save_profile_to_acp_config

        config_file = tmp_path / ".aeloon" / "acp.json"
        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.config.Path.home",
            lambda: tmp_path,
        )

        save_profile_to_acp_config("test_agent", {"command": ["npx", "foo"]})

        import json

        saved = json.loads(config_file.read_text(encoding="utf-8"))
        assert "test_agent" in saved["profiles"]
        assert saved["profiles"]["test_agent"]["command"] == ["npx", "foo"]

    def test_merges_into_existing(self, tmp_path: Path, monkeypatch) -> None:
        import json

        from aeloon.plugins.acp_bridge.config import save_profile_to_acp_config

        aeloon_dir = tmp_path / ".aeloon"
        aeloon_dir.mkdir()
        config_file = aeloon_dir / "acp.json"
        config_file.write_text(
            json.dumps({"profiles": {"existing": {"command": ["old"]}}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.config.Path.home",
            lambda: tmp_path,
        )

        save_profile_to_acp_config("new_agent", {"command": ["new"]})

        saved = json.loads(config_file.read_text(encoding="utf-8"))
        assert "existing" in saved["profiles"]
        assert "new_agent" in saved["profiles"]


class TestACPBridgePluginRegistration:
    def test_register_creates_pending_records(self, tmp_path: Path) -> None:
        from aeloon.plugins.acp_bridge.plugin import ACPBridgePlugin

        agent_loop = MagicMock()
        agent_loop.workspace = tmp_path
        agent_loop.provider = MagicMock()
        agent_loop.model = "test-model"
        agent_loop.tools = MagicMock()
        agent_loop.bus = MagicMock()
        agent_loop.bus.publish_outbound = AsyncMock()

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=agent_loop,
            plugin_id="aeloon.acp_bridge",
            config={},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="aeloon.acp_bridge",
            version="0.1.0",
            config={},
            runtime=runtime,
            registry=registry,
        )

        plugin = ACPBridgePlugin()
        plugin.register(api)

        assert any(record.name == "acp" for record in api._pending_commands)
        assert any(record.name == "acp" for record in api._pending_cli)
        assert any(spec.command_name == "connect" for spec in api._pending_cli[0].commands)

    def test_commit_after_register(self, tmp_path: Path) -> None:
        from aeloon.plugins.acp_bridge.plugin import ACPBridgePlugin

        agent_loop = MagicMock()
        agent_loop.workspace = tmp_path
        agent_loop.provider = MagicMock()
        agent_loop.model = "test-model"
        agent_loop.tools = MagicMock()
        agent_loop.bus = MagicMock()
        agent_loop.bus.publish_outbound = AsyncMock()

        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=agent_loop,
            plugin_id="aeloon.acp_bridge",
            config={},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="aeloon.acp_bridge",
            version="0.1.0",
            config={},
            runtime=runtime,
            registry=registry,
        )

        plugin = ACPBridgePlugin()
        plugin.register(api)
        api._commit()

        assert "acp" in registry.commands
        assert "acp" in registry.cli_registrars


# ---------------------------------------------------------------------------
# ACPClient (unit-level, no real transport)
# ---------------------------------------------------------------------------


class TestACPClientState:
    def test_initial_state(self) -> None:
        from aeloon.plugins._sdk.acp.client import ACPClient

        client = ACPClient()
        assert client.state == ConnectionState.DISCONNECTED
        assert not client.is_connected
        assert client.last_error is None

    def test_require_connection_raises_when_disconnected(self) -> None:
        from aeloon.plugins._sdk.acp.client import ACPClient

        client = ACPClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client._require_connection()

    def test_health_check_disconnected(self) -> None:
        from aeloon.plugins._sdk.acp.client import ACPClient

        client = ACPClient()
        health = client.health_check()
        assert health["state"] == "disconnected"
        assert health["connected"] is False
        assert health["sessions"] == 0

    def test_extract_text_from_string_content_update(self) -> None:
        from aeloon.plugins._sdk.acp.client import _extract_text_from_update

        class Update:
            content = "hello"

        assert _extract_text_from_update(Update()) == "hello"

    def test_extract_text_from_delta_text_update(self) -> None:
        from aeloon.plugins._sdk.acp.client import _extract_text_from_update

        class Delta:
            text = "hello"

        class Update:
            delta = Delta()

        assert _extract_text_from_update(Update()) == "hello"

    def test_streaming_collector_tracks_unknown_update_types(self) -> None:
        from aeloon.plugins._sdk.acp.client import _StreamingCollector

        class UnknownUpdate:
            pass

        collector = _StreamingCollector()
        collector("session-1", UnknownUpdate())
        assert collector.update_types == ["UnknownUpdate"]
        assert collector.unknown_update_types == ["UnknownUpdate"]
        assert collector.chunks == []


# ---------------------------------------------------------------------------
# ACPConnectionService
# ---------------------------------------------------------------------------


class TestACPConnectionService:
    def test_initial_health(self) -> None:
        from aeloon.plugins.acp_bridge.service import ACPConnectionService

        svc = ACPConnectionService()
        health = svc.health_check()
        assert health["state"] == "disconnected"
        assert "profile" not in health


class TestACPCommands:
    @pytest.mark.asyncio
    async def test_list_profiles_includes_default_and_custom(self, monkeypatch) -> None:
        from aeloon.plugins._sdk.types import CommandContext
        from aeloon.plugins.acp_bridge.commands import _cmd_list

        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.commands._get_merged_plugin_config",
            lambda _ctx: {
                "default_profile": "kimi_cli",
                "profiles": {
                    "kimi_cli": {"command": ["kimi", "--acp"]},
                    "claude_acp": {"command": ["npx", "@agentclientprotocol/claude-agent-acp"]},
                },
            },
        )

        replies: list[str] = []
        progress: list[str] = []
        ctx = CommandContext(
            session_key="s",
            channel="cli",
            reply=lambda text: replies.append(text) or __import__("asyncio").sleep(0),
            send_progress=lambda *args, **kwargs: (
                progress.append(str(args[0])) or __import__("asyncio").sleep(0)
            ),
            plugin_config={},
        )

        result = await _cmd_list(ctx)
        assert "Available ACP backends:" in result
        assert "- kimi_cli (default) — kimi --acp" in result
        assert "- claude_acp — npx @agentclientprotocol/claude-agent-acp" in result


# ---------------------------------------------------------------------------
# /acp market
# ---------------------------------------------------------------------------


class TestMarketCommand:
    @pytest.mark.asyncio
    async def test_market_lists_agents(self, monkeypatch) -> None:
        from aeloon.plugins._sdk.types import CommandContext
        from aeloon.plugins.acp_bridge.commands import _cmd_market

        _fake_registry = {
            "version": "1.0.0",
            "agents": [
                {
                    "id": "claude-acp",
                    "name": "Claude Agent",
                    "version": "0.26.0",
                    "description": "ACP wrapper for Claude",
                    "distribution": {"npx": {"package": "@agentclientprotocol/claude-agent-acp"}},
                },
                {
                    "id": "gemini",
                    "name": "Gemini CLI",
                    "version": "0.37.0",
                    "description": "Google's CLI for Gemini",
                    "distribution": {"npx": {"package": "@anthropic-ai/gemini-cli"}},
                },
            ],
        }

        async def _fake_fetch():
            return _fake_registry

        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.commands._fetch_registry",
            _fake_fetch,
        )

        progress: list[str] = []
        ctx = CommandContext(
            session_key="s",
            channel="cli",
            reply=lambda text: None,
            send_progress=lambda *a, **kw: (
                progress.append(str(a[0])) or __import__("asyncio").sleep(0)
            ),
            plugin_config={},
        )

        result = await _cmd_market(ctx, "")
        assert "ACP Registry — 2 agent(s)" in result
        assert "| `claude-acp` | **Claude Agent**" in result
        assert "| `gemini` | **Gemini CLI**" in result
        assert "`npx`" in result
        assert "/acp install <id>" in result
        # Version column should NOT have a separate Link column
        assert "| Link |" not in result

    @pytest.mark.asyncio
    async def test_market_filter_keyword(self, monkeypatch) -> None:
        from aeloon.plugins._sdk.types import CommandContext
        from aeloon.plugins.acp_bridge.commands import _cmd_market

        _fake_registry = {
            "version": "1.0.0",
            "agents": [
                {
                    "id": "claude-acp",
                    "name": "Claude Agent",
                    "version": "0.26.0",
                    "description": "ACP wrapper",
                    "distribution": {"npx": {"package": "x"}},
                },
                {
                    "id": "gemini",
                    "name": "Gemini CLI",
                    "version": "0.37.0",
                    "description": "Google's CLI",
                    "distribution": {"npx": {"package": "y"}},
                },
            ],
        }

        async def _fake_fetch():
            return _fake_registry

        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.commands._fetch_registry",
            _fake_fetch,
        )

        ctx = CommandContext(
            session_key="s",
            channel="cli",
            reply=lambda text: None,
            send_progress=lambda *a, **kw: __import__("asyncio").sleep(0),
            plugin_config={},
        )

        result = await _cmd_market(ctx, "google")
        assert "1 agent(s)" in result
        assert "Gemini CLI" in result
        assert "Claude" not in result

    @pytest.mark.asyncio
    async def test_market_no_match(self, monkeypatch) -> None:
        from aeloon.plugins._sdk.types import CommandContext
        from aeloon.plugins.acp_bridge.commands import _cmd_market

        async def _fake_fetch():
            return {"version": "1.0.0", "agents": []}

        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.commands._fetch_registry",
            _fake_fetch,
        )

        ctx = CommandContext(
            session_key="s",
            channel="cli",
            reply=lambda text: None,
            send_progress=lambda *a, **kw: __import__("asyncio").sleep(0),
            plugin_config={},
        )

        result = await _cmd_market(ctx, "")
        assert "No agents found" in result


# ---------------------------------------------------------------------------
# /acp install
# ---------------------------------------------------------------------------


class TestInstallCommand:
    @pytest.mark.asyncio
    async def test_install_npx_agent(self, monkeypatch, tmp_path) -> None:

        from aeloon.plugins._sdk.types import CommandContext
        from aeloon.plugins.acp_bridge.commands import _cmd_install

        _fake_registry = {
            "version": "1.0.0",
            "agents": [
                {
                    "id": "gemini",
                    "name": "Gemini CLI",
                    "version": "0.37.0",
                    "description": "Google's CLI",
                    "distribution": {
                        "npx": {
                            "package": "@google/gemini-cli@0.37.0",
                            "args": ["--acp"],
                            "env": {"GEMINI_KEY": "x"},
                        },
                    },
                },
            ],
        }

        async def _fake_fetch():
            return _fake_registry

        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.commands._fetch_registry",
            _fake_fetch,
        )

        # Intercept save_profile_to_acp_config via the config module
        _saved_profiles: dict[str, dict] = {}

        def _fake_save(profile_name: str, profile_data: dict) -> None:
            _saved_profiles[profile_name] = profile_data

        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.config.save_profile_to_acp_config",
            _fake_save,
        )

        ctx = CommandContext(
            session_key="s",
            channel="cli",
            reply=lambda text: None,
            send_progress=lambda *a, **kw: __import__("asyncio").sleep(0),
            plugin_config={},
        )

        result = await _cmd_install(ctx, "gemini")
        assert "Installed Gemini CLI" in result
        assert "npx" in result
        assert "/acp connect gemini" in result

        # Verify profile was saved
        assert "gemini" in _saved_profiles
        assert _saved_profiles["gemini"]["command"] == [
            "npx",
            "@google/gemini-cli@0.37.0",
            "--acp",
        ]
        assert _saved_profiles["gemini"]["env"] == {"GEMINI_KEY": "x"}

    @pytest.mark.asyncio
    async def test_install_unknown_agent(self, monkeypatch) -> None:
        from aeloon.plugins._sdk.types import CommandContext
        from aeloon.plugins.acp_bridge.commands import _cmd_install

        async def _fake_fetch():
            return {"version": "1.0.0", "agents": []}

        monkeypatch.setattr(
            "aeloon.plugins.acp_bridge.commands._fetch_registry",
            _fake_fetch,
        )

        ctx = CommandContext(
            session_key="s",
            channel="cli",
            reply=lambda text: None,
            send_progress=lambda *a, **kw: __import__("asyncio").sleep(0),
            plugin_config={},
        )

        result = await _cmd_install(ctx, "nonexistent")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_install_no_args(self) -> None:
        from aeloon.plugins._sdk.types import CommandContext
        from aeloon.plugins.acp_bridge.commands import _cmd_install

        ctx = CommandContext(
            session_key="s",
            channel="cli",
            reply=lambda text: None,
            send_progress=lambda *a, **kw: __import__("asyncio").sleep(0),
            plugin_config={},
        )

        result = await _cmd_install(ctx, "")
        assert "Usage" in result


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestRegistryHelpers:
    def test_resolve_by_id(self) -> None:
        from aeloon.plugins.acp_bridge.commands import _resolve_registry_agent

        registry = {
            "agents": [
                {"id": "foo", "name": "Foo Agent"},
                {"id": "bar", "name": "Bar Agent"},
            ]
        }
        assert _resolve_registry_agent(registry, "foo") is not None
        assert _resolve_registry_agent(registry, "foo")["name"] == "Foo Agent"

    def test_resolve_by_name_case_insensitive(self) -> None:
        from aeloon.plugins.acp_bridge.commands import _resolve_registry_agent

        registry = {"agents": [{"id": "gemini", "name": "Gemini CLI"}]}
        assert _resolve_registry_agent(registry, "gemini cli") is not None

    def test_resolve_not_found(self) -> None:
        from aeloon.plugins.acp_bridge.commands import _resolve_registry_agent

        assert _resolve_registry_agent({"agents": []}, "nope") is None

    def test_build_npx_profile(self) -> None:
        from aeloon.plugins.acp_bridge.commands import _build_npx_profile

        agent = {
            "distribution": {
                "npx": {
                    "package": "@google/gemini-cli",
                    "args": ["--acp"],
                    "env": {"KEY": "val"},
                }
            }
        }
        profile = _build_npx_profile(agent)
        assert profile["command"] == ["npx", "@google/gemini-cli", "--acp"]
        assert profile["env"] == {"KEY": "val"}

    def test_current_platform_key(self) -> None:
        from aeloon.plugins.acp_bridge.commands import _current_platform_key

        key = _current_platform_key()
        # Must be None or a valid key
        if key is not None:
            assert "-" in key
            parts = key.split("-")
            assert parts[0] in ("darwin", "linux", "windows")
            assert parts[1] in ("x86_64", "aarch64")
