"""Tests for dispatcher-level command middleware execution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.plugins._sdk.registry import PluginRegistry
from aeloon.plugins._sdk.types import CommandMiddlewareRecord, CommandRecord


class _TraceCommandMiddleware:
    def __init__(self, events: list[tuple[str, str, str]]) -> None:
        self._events = events

    async def before(self, cmd: str, args: str, ctx) -> None:
        self._events.append(("before", cmd, ctx.plugin_id or "builtin"))

    async def after(self, cmd: str, result, ctx) -> None:
        self._events.append(("after", cmd, ctx.plugin_id or "builtin"))


def _make_dispatcher() -> object:
    from aeloon.core.agent.dispatcher import Dispatcher

    agent_loop = MagicMock()
    agent_loop.plugin_manager = None
    agent_loop.sessions = MagicMock()
    session = MagicMock()
    session.messages = []
    session.last_consolidated = 0
    agent_loop.sessions.get_or_create.return_value = session
    agent_loop.runtime_settings = MagicMock()
    agent_loop.runtime_settings.show_detail = False
    agent_loop.runtime_settings.show_debug = False
    agent_loop.runtime_settings.show_profile = False
    agent_loop.runtime_settings.show_deep_profile = False
    agent_loop.profiler = MagicMock()
    agent_loop.profiler.enabled = False
    agent_loop.process_turn = AsyncMock(return_value="test response")
    agent_loop.bus = MagicMock()
    agent_loop.bus.publish_outbound = AsyncMock()

    return Dispatcher(agent_loop)


@pytest.mark.asyncio
async def test_builtin_command_middleware_runs_before_and_after() -> None:
    from aeloon.core.bus.events import InboundMessage

    dispatcher = _make_dispatcher()
    events: list[tuple[str, str, str]] = []
    dispatcher.add_middleware(_TraceCommandMiddleware(events))

    response = await dispatcher.process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
    )

    assert response is not None
    assert events == [("before", "/help", "builtin"), ("after", "/help", "builtin")]


@pytest.mark.asyncio
async def test_plugin_command_middlewares_run_after_local_middlewares() -> None:
    from aeloon.core.bus.events import InboundMessage

    dispatcher = _make_dispatcher()
    local_events: list[tuple[str, str, str]] = []
    plugin_events: list[tuple[str, str, str]] = []
    dispatcher.add_middleware(_TraceCommandMiddleware(local_events))

    registry = PluginRegistry()
    registry.commit_plugin(
        "test.echo",
        commands=[
            CommandRecord(
                plugin_id="test.echo",
                name="echo",
                handler=AsyncMock(return_value="echo ok"),
                description="Echo plugin command",
            )
        ],
        command_middlewares=[
            CommandMiddlewareRecord(
                plugin_id="test.echo",
                name="trace",
                middleware=_TraceCommandMiddleware(plugin_events),
            )
        ],
    )
    dispatcher._agent_loop.plugin_manager = MagicMock(
        registry=registry,
        _plugin_config={"test.echo": {}},
    )
    dispatcher._agent_loop.plugin_manager._hooks = MagicMock()
    dispatcher._agent_loop.plugin_manager._hooks.dispatch_notify = AsyncMock()
    dispatcher._agent_loop._profiled_turn = lambda: MagicMock()

    response = await dispatcher.process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/echo hi")
    )

    assert response is not None
    assert response.content == "echo ok"
    assert local_events == [("before", "/echo", "test.echo"), ("after", "/echo", "test.echo")]
    assert plugin_events == [("before", "/echo", "test.echo"), ("after", "/echo", "test.echo")]
