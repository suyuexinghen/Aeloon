"""Tests for channel state tracking and /status slash command."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aeloon.channels.base import BaseChannel
from aeloon.channels.manager import ChannelManager, ChannelState
from aeloon.core.bus.events import InboundMessage, OutboundMessage
from aeloon.core.bus.queue import MessageBus

# ---------------------------------------------------------------------------
# Dummy channel implementations
# ---------------------------------------------------------------------------


class _LongRunningChannel(BaseChannel):
    """Simulates a healthy channel that runs until stopped."""

    name = "longrunning"
    display_name = "Long Running"

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        self._running = True
        await self._stop_event.wait()
        self._running = False

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()

    async def send(self, msg: OutboundMessage) -> None:
        pass


class _SilentFailChannel(BaseChannel):
    """Simulates a channel that returns early without raising (silent failure)."""

    name = "silentfail"
    display_name = "Silent Fail"

    async def start(self) -> None:
        # Mimics telegram.py / email.py pattern: log error, return
        return

    async def stop(self) -> None:
        pass

    async def send(self, msg: OutboundMessage) -> None:
        pass


class _ExceptionChannel(BaseChannel):
    """Simulates a channel that raises on start."""

    name = "exception"
    display_name = "Exception"

    async def start(self) -> None:
        raise ConnectionError("connection refused")

    async def stop(self) -> None:
        pass

    async def send(self, msg: OutboundMessage) -> None:
        pass


class _CrashAfterRunChannel(BaseChannel):
    """Simulates a channel that starts successfully then crashes."""

    name = "crashafter"
    display_name = "Crash After"

    async def start(self) -> None:
        self._running = True
        await asyncio.sleep(0.2)
        raise RuntimeError("unexpected disconnect")

    async def stop(self) -> None:
        self._running = False

    async def send(self, msg: OutboundMessage) -> None:
        pass


class _ExitWithoutErrorChannel(BaseChannel):
    """Simulates a channel whose loop exits while is_running is still True."""

    name = "stalerunning"
    display_name = "Stale Running"

    async def start(self) -> None:
        self._running = True
        await asyncio.sleep(0.2)
        # Exits without setting _running = False (like DingTalk outer-except path)

    async def stop(self) -> None:
        self._running = False

    async def send(self, msg: OutboundMessage) -> None:
        pass


# ---------------------------------------------------------------------------
# Helper to build a ChannelManager without auto-discovery
# ---------------------------------------------------------------------------


def _make_manager(channels: dict[str, BaseChannel]) -> ChannelManager:
    """Build a ChannelManager with pre-built channel instances (skip discovery)."""
    bus = MessageBus()
    config = MagicMock()
    config.channels.send_progress = False
    config.channels.send_tool_hints = False

    with patch.object(ChannelManager, "_init_channels"):
        mgr = ChannelManager(config, bus)

    mgr.channels = dict(channels)
    for name in channels:
        mgr._channel_states[name] = ChannelState.PENDING
    return mgr


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


class TestChannelStateTransitions:
    @pytest.mark.asyncio
    async def test_long_running_channel_becomes_running(self):
        ch = _LongRunningChannel(SimpleNamespace(allow_from=["*"]), MessageBus())
        mgr = _make_manager({"lr": ch})

        task = asyncio.create_task(mgr.start_all())
        # Wait for the channel to be marked running
        for _ in range(20):
            if mgr._channel_states.get("lr") == ChannelState.RUNNING:
                break
            await asyncio.sleep(0.1)

        assert mgr._channel_states["lr"] == ChannelState.RUNNING
        assert "lr" not in mgr._channel_errors

        # Cleanup
        await mgr.stop_all()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_silent_fail_detected_as_failed(self):
        ch = _SilentFailChannel(SimpleNamespace(allow_from=["*"]), MessageBus())
        mgr = _make_manager({"sf": ch})

        task = asyncio.create_task(mgr.start_all())
        # Wait for the task to settle
        for _ in range(20):
            if mgr._channel_states.get("sf") == ChannelState.FAILED:
                break
            await asyncio.sleep(0.1)

        assert mgr._channel_states["sf"] == ChannelState.FAILED
        assert "did not start" in mgr._channel_errors["sf"]

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_exception_on_start_becomes_failed(self):
        ch = _ExceptionChannel(SimpleNamespace(allow_from=["*"]), MessageBus())
        mgr = _make_manager({"exc": ch})

        task = asyncio.create_task(mgr.start_all())
        for _ in range(20):
            if mgr._channel_states.get("exc") == ChannelState.FAILED:
                break
            await asyncio.sleep(0.1)

        assert mgr._channel_states["exc"] == ChannelState.FAILED
        assert "connection refused" in mgr._channel_errors["exc"]

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_crash_after_running_becomes_failed(self):
        ch = _CrashAfterRunChannel(SimpleNamespace(allow_from=["*"]), MessageBus())
        mgr = _make_manager({"crash": ch})

        task = asyncio.create_task(mgr.start_all())
        # First it should become RUNNING, then FAILED
        for _ in range(40):
            if mgr._channel_states.get("crash") == ChannelState.FAILED:
                break
            await asyncio.sleep(0.1)

        assert mgr._channel_states["crash"] == ChannelState.FAILED
        assert "unexpected disconnect" in mgr._channel_errors["crash"]

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_exit_with_stale_running_flag_becomes_failed(self):
        """Channel exits normally but forgets to clear is_running → FAILED."""
        ch = _ExitWithoutErrorChannel(SimpleNamespace(allow_from=["*"]), MessageBus())
        mgr = _make_manager({"stale": ch})

        task = asyncio.create_task(mgr.start_all())
        for _ in range(40):
            if mgr._channel_states.get("stale") == ChannelState.FAILED:
                break
            await asyncio.sleep(0.1)

        assert mgr._channel_states["stale"] == ChannelState.FAILED
        assert "unexpectedly" in mgr._channel_errors["stale"]

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_graceful_stop_becomes_stopped(self):
        ch = _LongRunningChannel(SimpleNamespace(allow_from=["*"]), MessageBus())
        mgr = _make_manager({"lr": ch})

        task = asyncio.create_task(mgr.start_all())
        for _ in range(20):
            if mgr._channel_states.get("lr") == ChannelState.RUNNING:
                break
            await asyncio.sleep(0.1)

        await mgr.stop_all()
        assert mgr._channel_states["lr"] == ChannelState.STOPPED

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_get_status_includes_error_for_failed(self):
        ch = _ExceptionChannel(SimpleNamespace(allow_from=["*"]), MessageBus())
        mgr = _make_manager({"exc": ch})

        task = asyncio.create_task(mgr.start_all())
        for _ in range(20):
            if mgr._channel_states.get("exc") == ChannelState.FAILED:
                break
            await asyncio.sleep(0.1)

        status = mgr.get_status()
        assert status["exc"]["state"] == "failed"
        assert "error" in status["exc"]
        assert status["exc"]["display_name"] == "Exception"

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# /status slash command tests
# ---------------------------------------------------------------------------


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from aeloon.core.agent.loop import AgentLoop

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with (
        patch("aeloon.core.agent.loop.ContextBuilder"),
        patch("aeloon.core.agent.loop.SessionManager"),
        patch("aeloon.core.agent.loop.SubagentManager"),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestStatusCommand:
    @pytest.mark.asyncio
    async def test_status_no_channel_manager(self):
        loop, _ = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/status")
        result = await loop.dispatcher.process_message(msg)
        assert result is not None
        assert "not available" in result.content

    @pytest.mark.asyncio
    async def test_status_no_channels(self):
        loop, _ = _make_loop()
        mgr = MagicMock()
        mgr.get_status.return_value = {}
        loop.dispatcher.channel_manager = mgr

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/status")
        result = await loop.dispatcher.process_message(msg)
        assert result is not None
        assert "No channels configured" in result.content

    @pytest.mark.asyncio
    async def test_status_shows_running_and_failed(self):
        loop, _ = _make_loop()
        mgr = MagicMock()
        mgr.get_status.return_value = {
            "telegram": {
                "display_name": "Telegram",
                "state": "running",
            },
            "email": {
                "display_name": "Email",
                "state": "failed",
                "error": "consent not granted",
            },
        }
        loop.dispatcher.channel_manager = mgr

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/status")
        result = await loop.dispatcher.process_message(msg)
        assert result is not None
        assert "Telegram" in result.content
        assert "running" in result.content
        assert "Email" in result.content
        assert "failed" in result.content
        assert "consent not granted" in result.content

    @pytest.mark.asyncio
    async def test_status_routed_in_process_message(self):
        """Verify /status is handled inside process_message (not dispatched to agent)."""
        loop, bus = _make_loop()
        mgr = MagicMock()
        mgr.get_status.return_value = {
            "slack": {"display_name": "Slack", "state": "running"},
        }
        loop.dispatcher.channel_manager = mgr

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/status")
        result = await loop.dispatcher.process_message(msg)
        assert result is not None
        assert "Slack" in result.content

    @pytest.mark.asyncio
    async def test_help_includes_status(self):
        loop, _ = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/help")
        result = await loop.dispatcher.process_message(msg)
        assert result is not None
        assert "/status" in result.content
