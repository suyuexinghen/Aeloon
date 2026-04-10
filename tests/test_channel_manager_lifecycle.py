import asyncio
from types import SimpleNamespace

import pytest

from aeloon.channels.base import BaseChannel
from aeloon.channels.manager import ChannelManager
from aeloon.core.bus.queue import MessageBus


class _SlowChannel(BaseChannel):
    name = "slow"
    display_name = "Slow"

    async def start(self) -> None:
        self._running = True
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()

    async def send(self, msg) -> None:
        return None


@pytest.mark.asyncio
async def test_start_all_tracks_channel_tasks(monkeypatch) -> None:
    mgr = ChannelManager.__new__(ChannelManager)
    mgr.config = SimpleNamespace(
        channels=SimpleNamespace(send_tool_hints=False, send_progress=True),
        providers=SimpleNamespace(groq=SimpleNamespace(api_key="")),
    )
    mgr.bus = MessageBus()
    mgr.channels = {"slow": _SlowChannel(SimpleNamespace(allow_from=["*"]), mgr.bus)}
    mgr._dispatch_task = None
    mgr._channel_tasks = {}

    start_all = asyncio.create_task(ChannelManager.start_all(mgr))
    await asyncio.sleep(0)

    assert "slow" in mgr._channel_tasks

    await mgr.stop_all()
    await start_all
