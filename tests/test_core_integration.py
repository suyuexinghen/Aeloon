"""Core runtime integration tests (SP4 — P3-4, SP7 — P7-3/P7-4).

Validates that the plugin-driven runtime works correctly:
config schema, dispatcher routing, boot sequence, shutdown, middleware merge,
and dynamic /help rendering.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aeloon.plugins._sdk.registry import PluginRegistry
from aeloon.plugins._sdk.types import CLICommandSpec, CLIRecord, CommandRecord

# ---------------------------------------------------------------------------
# TestPluginConfig
# ---------------------------------------------------------------------------


class TestPluginConfig:
    """Test Config.plugins field."""

    def test_plugins_defaults_empty(self) -> None:
        from aeloon.core.config.schema import Config

        cfg = Config()
        assert cfg.plugins == {}

    def test_plugins_accepts_dict(self) -> None:
        from aeloon.core.config.schema import Config

        cfg = Config.model_validate({"plugins": {"aeloon.science": {"enabled": True}}})
        assert cfg.plugins["aeloon.science"]["enabled"] is True

    def test_no_science_top_level_field(self) -> None:
        from aeloon.core.config.schema import Config

        assert "science" not in Config.model_fields

    def test_backward_compat_science_config_importable(self) -> None:
        """ScienceConfig importable from plugins.science.config."""
        from aeloon.plugins.ScienceResearch.config import ScienceConfig

        sc = ScienceConfig()
        assert sc.enabled is False


# ---------------------------------------------------------------------------
# TestDispatcherPluginRouting
# ---------------------------------------------------------------------------


def _make_dispatcher_with_plugins(plugin_commands: dict | None = None):
    """Build a Dispatcher with optional plugin commands in the registry."""
    from aeloon.core.agent.loop import AgentLoop
    from aeloon.core.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with (
        patch("aeloon.core.agent.loop.ContextBuilder"),
        patch("aeloon.core.agent.loop.SessionManager") as mock_sm,
        patch("aeloon.core.agent.loop.SubagentManager") as mock_sub,
    ):
        mock_sub.return_value.cancel_by_session = AsyncMock(return_value=0)
        session = MagicMock()
        session.messages = []
        session.last_consolidated = 0
        mock_sm.return_value.get_or_create = MagicMock(return_value=session)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

    if plugin_commands:
        pm = MagicMock()
        registry = PluginRegistry()
        for name, handler in plugin_commands.items():
            registry.commit_plugin(
                f"test.{name}",
                commands=[CommandRecord(plugin_id=f"test.{name}", name=name, handler=handler)],
            )
        pm.registry = registry
        pm._plugin_config = {}
        loop.plugin_manager = pm

    from aeloon.core.agent.dispatcher import Dispatcher

    return Dispatcher(loop)


class TestDispatcherPluginRouting:
    """Test generic plugin command routing."""

    def test_known_commands_includes_plugin_commands(self) -> None:
        """Plugin commands appear in _known_slash_commands()."""
        handler = AsyncMock(return_value="ok")
        dispatcher = _make_dispatcher_with_plugins({"mycommand": handler})
        commands = dispatcher._known_slash_commands()
        assert "/mycommand" in commands

    def test_known_commands_without_plugin_manager(self) -> None:
        """Without plugin_manager, only base commands listed."""
        dispatcher = _make_dispatcher_with_plugins()
        commands = dispatcher._known_slash_commands()
        assert "/help" in commands
        assert "/channel" in commands
        assert "/channel wechat status" in commands
        assert "/plugin" in commands
        assert "/feishu" in commands
        assert "/mycommand" not in commands

    def test_known_commands_include_nested_plugin_commands(self) -> None:
        """Declarative plugin subcommands appear in _known_slash_commands()."""
        dispatcher = _make_dispatcher_with_plugins()
        registry = PluginRegistry()
        registry.commit_plugin(
            "test.pc",
            commands=[
                CommandRecord(
                    plugin_id="test.pc",
                    name="pc",
                    handler=AsyncMock(return_value="ok"),
                    description="Create and manage plugin plans",
                )
            ],
            cli=[
                CLIRecord(
                    plugin_id="test.pc",
                    name="pc",
                    builder=lambda _app: None,
                    commands=(
                        CLICommandSpec(
                            group_name="pc",
                            command_name="plan",
                            help="Create a plugin plan from a requirement description.",
                            plugin_command="pc",
                        ),
                    ),
                )
            ],
        )
        dispatcher._agent_loop.plugin_manager = MagicMock(registry=registry, _plugin_config={})

        commands = dispatcher._known_slash_commands()

        assert "/pc" in commands
        assert "/pc plan" in commands

    @pytest.mark.asyncio
    async def test_plugin_command_dispatched(self) -> None:
        """Plugin command routes to handler and returns result."""
        from aeloon.core.bus.events import InboundMessage

        handler = AsyncMock(return_value="plugin response")
        dispatcher = _make_dispatcher_with_plugins({"mycommand": handler})

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/mycommand arg1")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert response.content == "plugin response"
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_help_still_works_with_plugins(self) -> None:
        """Core /help command still works when plugin_manager is set."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher = _make_dispatcher_with_plugins({"science": AsyncMock()})
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)
        assert response is not None

    @pytest.mark.asyncio
    async def test_unknown_command_returns_direct_error(self) -> None:
        """Unknown /xyz returns a direct error without fuzzy suggestions."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher = _make_dispatcher_with_plugins()
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/xyznotacommand")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert (
            response.content
            == "Unknown command: /xyznotacommand. Use /help to see available commands."
        )


# ---------------------------------------------------------------------------
# TestBootPlugins
# ---------------------------------------------------------------------------


class TestBootPlugins:
    """Test _boot_plugins() helper."""

    @pytest.mark.asyncio
    async def test_boot_returns_manager(self) -> None:
        """_boot_plugins creates and returns a PluginManager."""
        from aeloon.core.config.schema import Config

        loop = MagicMock()
        loop.tools = MagicMock()
        config = Config()

        with (
            patch("aeloon.plugins._sdk.discovery.PluginDiscovery") as mock_disc,
            patch("aeloon.plugins._sdk.manager.PluginManager") as mock_pm_cls,
        ):
            mock_disc.return_value.discover_all.return_value = []
            mock_pm = MagicMock()
            mock_pm.boot = AsyncMock(return_value=MagicMock(loaded=[], failed=[]))
            mock_pm.registry = PluginRegistry()
            mock_pm_cls.return_value = mock_pm

            from aeloon.cli.flows.helpers import boot_plugins

            result = await boot_plugins(loop, config)
            assert result is not None


# ---------------------------------------------------------------------------
# TestPluginShutdown
# ---------------------------------------------------------------------------


class TestPluginShutdown:
    """Test plugin manager shutdown integration."""

    @pytest.mark.asyncio
    async def test_shutdown_deactivates_in_reverse(self) -> None:
        """Plugins deactivated in reverse activation order."""
        from aeloon.plugins._sdk.hooks import HookDispatcher
        from aeloon.plugins._sdk.loader import PluginLoader
        from aeloon.plugins._sdk.manager import PluginManager

        registry = PluginRegistry()
        manager = PluginManager(
            registry=registry,
            discovery=MagicMock(),
            loader=PluginLoader(),
            hook_dispatcher=HookDispatcher(),
            agent_loop=MagicMock(),
            plugin_config={},
            storage_base=Path("/tmp"),
        )

        # Simulate two activated plugins
        manager._activation_order = ["p1", "p2"]
        p1_inst = AsyncMock()
        p2_inst = AsyncMock()
        registry.add_plugin(MagicMock(plugin_id="p1", instance=p1_inst))
        registry.add_plugin(MagicMock(plugin_id="p2", instance=p2_inst))

        deactivation_order: list[str] = []
        p1_inst.deactivate = AsyncMock(side_effect=lambda: deactivation_order.append("p1"))
        p2_inst.deactivate = AsyncMock(side_effect=lambda: deactivation_order.append("p2"))

        await manager.shutdown()
        assert deactivation_order == ["p2", "p1"]


# ---------------------------------------------------------------------------
# TestMiddlewareMerge
# ---------------------------------------------------------------------------


class TestMiddlewareMerge:
    """Test plugin middleware merge into agent kernel."""

    def test_no_plugin_manager_only_profiler(self) -> None:
        """Without plugin_manager, only ProfilerMiddleware."""
        from aeloon.core.agent.loop import AgentLoop
        from aeloon.core.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        workspace = MagicMock()
        workspace.__truediv__ = MagicMock(return_value=MagicMock())

        with (
            patch("aeloon.core.agent.loop.ContextBuilder"),
            patch("aeloon.core.agent.loop.SessionManager") as mock_sm,
            patch("aeloon.core.agent.loop.SubagentManager"),
        ):
            session = MagicMock()
            session.messages = []
            session.last_consolidated = 0
            mock_sm.return_value.get_or_create = MagicMock(return_value=session)
            loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

        assert loop.plugin_manager is None

    def test_plugin_manager_attribute_settable(self) -> None:
        """AgentLoop.plugin_manager can be set externally."""
        from aeloon.core.agent.loop import AgentLoop
        from aeloon.core.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        workspace = MagicMock()
        workspace.__truediv__ = MagicMock(return_value=MagicMock())

        with (
            patch("aeloon.core.agent.loop.ContextBuilder"),
            patch("aeloon.core.agent.loop.SessionManager") as mock_sm,
            patch("aeloon.core.agent.loop.SubagentManager"),
        ):
            session = MagicMock()
            session.messages = []
            session.last_consolidated = 0
            mock_sm.return_value.get_or_create = MagicMock(return_value=session)
            loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

        pm = MagicMock()
        loop.plugin_manager = pm
        assert loop.plugin_manager is pm


# ---------------------------------------------------------------------------
# TestDynamicHelp (SP7 — P7-1, P7-3, P7-4, P7-5)
# ---------------------------------------------------------------------------


def _make_dispatcher_with_plugin_commands(
    commands: dict[str, str],
) -> "tuple":
    """Build a Dispatcher with plugin commands carrying descriptions.

    Returns (dispatcher, loop) for assertion convenience.
    """
    from aeloon.core.agent.loop import AgentLoop
    from aeloon.core.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with (
        patch("aeloon.core.agent.loop.ContextBuilder"),
        patch("aeloon.core.agent.loop.SessionManager") as mock_sm,
        patch("aeloon.core.agent.loop.SubagentManager") as mock_sub,
    ):
        mock_sub.return_value.cancel_by_session = AsyncMock(return_value=0)
        session = MagicMock()
        session.messages = []
        session.last_consolidated = 0
        mock_sm.return_value.get_or_create = MagicMock(return_value=session)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

    registry = PluginRegistry()
    for name, desc in commands.items():
        registry.commit_plugin(
            f"test.{name}",
            commands=[
                CommandRecord(
                    plugin_id=f"test.{name}",
                    name=name,
                    handler=AsyncMock(return_value="ok"),
                    description=desc,
                )
            ],
        )
    pm = MagicMock()
    pm.registry = registry
    pm._plugin_config = {}
    loop.plugin_manager = pm

    from aeloon.core.agent.dispatcher import Dispatcher

    return Dispatcher(loop), loop


class TestDynamicHelp:
    """SP7: Dynamic /help lists built-in + plugin commands with descriptions."""

    @pytest.mark.asyncio
    async def test_help_lists_builtin_commands(self) -> None:
        """/help includes built-in commands like /new, /stop, /help."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher, _ = _make_dispatcher_with_plugin_commands({})
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "/new" in response.content
        assert "/stop" in response.content
        assert "/channel" in response.content
        assert "/plugin" in response.content
        assert "/feishu" in response.content
        assert "/help" in response.content

    @pytest.mark.asyncio
    async def test_help_renders_builtin_commands_as_tree(self) -> None:
        """Nested built-in commands render under their parent command."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher, _ = _make_dispatcher_with_plugin_commands({})
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)

        assert response is not None
        assert "- `/channel` — Manage one channel." in response.content
        assert "  - `wechat`" in response.content
        assert "    - `status`" in response.content
        assert "  - `status`" in response.content

    @pytest.mark.asyncio
    async def test_help_lists_plugin_commands_with_descriptions(self) -> None:
        """/help dynamically lists registered plugin commands with descriptions."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher, _ = _make_dispatcher_with_plugin_commands(
            {"science": "Run an AI4S science research task"}
        )
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "/science" in response.content
        assert "AI4S" in response.content

    @pytest.mark.asyncio
    async def test_help_renders_nested_plugin_commands_as_tree(self) -> None:
        """Declarative plugin subcommands render beneath the plugin command."""
        from aeloon.core.agent.loop import AgentLoop
        from aeloon.core.bus.events import InboundMessage
        from aeloon.core.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        workspace = MagicMock()
        workspace.__truediv__ = MagicMock(return_value=MagicMock())

        with (
            patch("aeloon.core.agent.loop.ContextBuilder"),
            patch("aeloon.core.agent.loop.SessionManager") as mock_sm,
            patch("aeloon.core.agent.loop.SubagentManager") as mock_sub,
        ):
            mock_sub.return_value.cancel_by_session = AsyncMock(return_value=0)
            session = MagicMock()
            session.messages = []
            session.last_consolidated = 0
            mock_sm.return_value.get_or_create = MagicMock(return_value=session)
            loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

        registry = PluginRegistry()
        registry.commit_plugin(
            "test.pc",
            commands=[
                CommandRecord(
                    plugin_id="test.pc",
                    name="pc",
                    handler=AsyncMock(return_value="ok"),
                    description="Create and manage plugin plans",
                )
            ],
            cli=[
                CLIRecord(
                    plugin_id="test.pc",
                    name="pc",
                    builder=lambda _app: None,
                    commands=(
                        CLICommandSpec(
                            group_name="pc",
                            command_name="plan",
                            help="Create a plugin plan from a requirement description.",
                            plugin_command="pc",
                        ),
                        CLICommandSpec(
                            group_name="pc",
                            command_name="status",
                            help="Show the latest plugin planning status.",
                            plugin_command="pc",
                        ),
                    ),
                )
            ],
        )
        loop.plugin_manager = MagicMock(registry=registry, _plugin_config={})

        from aeloon.core.agent.dispatcher import Dispatcher

        dispatcher = Dispatcher(loop)
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)

        assert response is not None
        assert "## Plugins" in response.content
        assert "- `/pc` — Create and manage plugin plans" in response.content
        assert (
            "  - `plan` — Create a plugin plan from a requirement description." in response.content
        )
        assert "  - `status` — Show the latest plugin planning status." in response.content

    @pytest.mark.asyncio
    async def test_help_shows_market_and_fs(self) -> None:
        """/help lists /market and /fs alongside /science."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher, _ = _make_dispatcher_with_plugin_commands(
            {
                "science": "Run an AI4S science research task",
                "market": "Market/news intelligence workflows",
                "fs": "Filesystem snapshots, audit, and diagnostics",
            }
        )
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "/science" in response.content
        assert "/market" in response.content
        assert "/fs" in response.content

    @pytest.mark.asyncio
    async def test_help_includes_plugins_section(self) -> None:
        """/help has a Plugins section when plugins are registered."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher, _ = _make_dispatcher_with_plugin_commands({"custom": "A custom plugin"})
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "## Plugins" in response.content
        assert "/custom" in response.content

    @pytest.mark.asyncio
    async def test_help_no_plugins_section_without_plugins(self) -> None:
        """/help has no Plugins section when no plugins are registered."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher, _ = _make_dispatcher_with_plugin_commands({})
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "## Plugins" not in response.content

    @pytest.mark.asyncio
    async def test_help_no_hardcoded_science(self) -> None:
        """/help does not have a hardcoded /science entry separate from the plugin registry."""
        from aeloon.core.bus.events import InboundMessage

        # No plugins registered — /science should NOT appear
        dispatcher, _ = _make_dispatcher_with_plugin_commands({})
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "/science" not in response.content

    @pytest.mark.asyncio
    async def test_new_plugin_auto_appears_in_help(self) -> None:
        """Adding a new plugin command automatically makes it appear in /help."""
        from aeloon.core.bus.events import InboundMessage

        dispatcher, _ = _make_dispatcher_with_plugin_commands(
            {"myplugin": "My custom plugin command"}
        )
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert "/myplugin" in response.content
        assert "My custom plugin command" in response.content

    @pytest.mark.asyncio
    async def test_market_command_routes_correctly(self) -> None:
        """/market routes to the plugin handler via the registry."""
        from aeloon.core.bus.events import InboundMessage

        handler = AsyncMock(return_value="market response")
        registry = PluginRegistry()
        registry.commit_plugin(
            "test.market",
            commands=[
                CommandRecord(
                    plugin_id="test.market",
                    name="market",
                    handler=handler,
                    description="Market plugin",
                )
            ],
        )

        from aeloon.core.agent.loop import AgentLoop
        from aeloon.core.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        workspace = MagicMock()
        workspace.__truediv__ = MagicMock(return_value=MagicMock())

        with (
            patch("aeloon.core.agent.loop.ContextBuilder"),
            patch("aeloon.core.agent.loop.SessionManager") as mock_sm,
            patch("aeloon.core.agent.loop.SubagentManager") as mock_sub,
        ):
            mock_sub.return_value.cancel_by_session = AsyncMock(return_value=0)
            session = MagicMock()
            session.messages = []
            session.last_consolidated = 0
            mock_sm.return_value.get_or_create = MagicMock(return_value=session)
            loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

        pm = MagicMock()
        pm.registry = registry
        pm._plugin_config = {}
        loop.plugin_manager = pm

        from aeloon.core.agent.dispatcher import Dispatcher

        dispatcher = Dispatcher(loop)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/market status")
        response = await dispatcher.process_message(msg)
        assert response is not None
        assert response.content == "market response"
        handler.assert_called_once()
