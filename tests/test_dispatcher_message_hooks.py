"""Tests for dispatcher message hook payloads."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_message_received_hook_includes_content_and_media() -> None:
    from aeloon.core.agent.dispatcher import Dispatcher
    from aeloon.core.bus.events import InboundMessage
    from aeloon.plugins._sdk.hooks import HookEvent

    agent_loop = MagicMock()
    agent_loop.plugin_manager = MagicMock()
    agent_loop.plugin_manager._hooks = MagicMock()
    agent_loop.plugin_manager._hooks.dispatch_notify = AsyncMock()
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
        sender_id="user-1",
        chat_id="chat-1",
        content="hello https://example.com",
        media=["/tmp/example.pdf"],
    )

    await dispatcher.process_message(msg)

    agent_loop.plugin_manager._hooks.dispatch_notify.assert_called()
    args, kwargs = agent_loop.plugin_manager._hooks.dispatch_notify.call_args_list[0]
    assert args[0] == HookEvent.MESSAGE_RECEIVED
    assert kwargs["channel"] == "cli"
    assert kwargs["sender_id"] == "user-1"
    assert kwargs["chat_id"] == "chat-1"
    assert kwargs["session_key"] == "cli:chat-1"
    assert kwargs["content"] == "hello https://example.com"
    assert kwargs["media"] == ["/tmp/example.pdf"]
    assert kwargs["content_preview"] == "hello https://example.com"
