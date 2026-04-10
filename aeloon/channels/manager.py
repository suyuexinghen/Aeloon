"""Start channels and route outbound messages."""

from __future__ import annotations

import asyncio
import enum
from typing import Any

from loguru import logger

from aeloon.channels.base import BaseChannel
from aeloon.core.bus.queue import MessageBus
from aeloon.core.config.schema import Config


class ChannelState(enum.Enum):
    """Lifecycle state of a channel."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


class ChannelManager:
    """Track enabled channels, their tasks, and their lifecycle state."""

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._channel_tasks: dict[str, asyncio.Task] = {}
        self._channel_states: dict[str, ChannelState] = {}
        self._channel_errors: dict[str, str] = {}

        self._init_channels()

    def _init_channels(self) -> None:
        """Build enabled channels from config and the registry."""
        from aeloon.channels.registry import discover_all

        groq_key = self.config.providers.groq.api_key

        for name, cls in discover_all().items():
            section = getattr(self.config.channels, name, None)
            if section is None:
                # Fall back to the channel's default config.
                section = cls.default_config()
            enabled = (
                section.get("enabled", False)
                if isinstance(section, dict)
                else getattr(section, "enabled", False)
            )
            if not enabled:
                continue
            try:
                channel = cls(section, self.bus)
                channel.transcription_api_key = groq_key
                self.channels[name] = channel
                self._channel_states[name] = ChannelState.PENDING
                logger.info("{} channel enabled", cls.display_name)
            except Exception as e:
                logger.warning("{} channel not available: {}", name, e)

        self._validate_allow_from()

    def _validate_allow_from(self) -> None:
        """Placeholder for channel allow-list validation."""
        pass

    def _resolve_finished_state(self, name: str, channel: BaseChannel) -> None:
        """Update state after a channel task exits cleanly."""
        if channel.is_running:
            # The loop returned without clearing its running flag.
            self._channel_states[name] = ChannelState.FAILED
            self._channel_errors[name] = "channel exited unexpectedly"
        elif self._channel_states[name] == ChannelState.STARTING:
            # The channel never reported itself as ready.
            self._channel_states[name] = ChannelState.FAILED
            self._channel_errors[name] = "channel did not start (check logs)"
        else:
            self._channel_states[name] = ChannelState.STOPPED

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start one channel and keep its lifecycle state in sync."""
        self._channel_states[name] = ChannelState.STARTING
        try:
            start_task = asyncio.create_task(channel.start())

            # Wait for the channel to report ready or exit.
            while not channel.is_running and not start_task.done():
                await asyncio.sleep(0.5)

            if start_task.done():
                start_task.result()
                self._resolve_finished_state(name, channel)
                return

            self._channel_states[name] = ChannelState.RUNNING

            await start_task
            self._resolve_finished_state(name, channel)

        except asyncio.CancelledError:
            self._channel_states[name] = ChannelState.STOPPED
            raise
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)
            self._channel_states[name] = ChannelState.FAILED
            self._channel_errors[name] = str(e)

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            self._channel_tasks[name] = asyncio.create_task(self._start_channel(name, channel))

        await asyncio.gather(*self._channel_tasks.values(), return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for name, channel in self.channels.items():
            try:
                await channel.stop()
                self._channel_states[name] = ChannelState.STOPPED
                logger.info("Stopped {} channel", name)
            except Exception as e:
                logger.error("Error stopping {}: {}", name, e)

        for name, task in list(self._channel_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._channel_tasks.clear()

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")

        while True:
            try:
                msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)

                if msg.metadata.get("_progress"):
                    if msg.metadata.get("_tool_hint") and not self.config.channels.send_tool_hints:
                        continue
                    if (
                        not msg.metadata.get("_tool_hint")
                        and not self.config.channels.send_progress
                    ):
                        continue

                channel = self.channels.get(msg.channel)
                if channel:
                    try:
                        await channel.send(msg)
                    except Exception as e:
                        logger.error("Error sending to {}: {}", msg.channel, e)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Return status for each enabled channel."""
        result: dict[str, Any] = {}
        for name, channel in self.channels.items():
            state = self._channel_states.get(name, ChannelState.PENDING)
            entry: dict[str, Any] = {
                "display_name": channel.display_name,
                "state": state.value,
            }
            if state == ChannelState.FAILED:
                entry["error"] = self._channel_errors.get(name, "unknown")
            result[name] = entry
        return result

    @property
    def enabled_channels(self) -> list[str]:
        """Return the names of enabled channels."""
        return list(self.channels.keys())

    def _get_channel_config(self, name: str) -> Any:
        """Return the enabled config block for one channel."""
        section = getattr(self.config.channels, name, None)
        if section is None:
            return None

        if isinstance(section, dict):
            return section if section.get("enabled", False) else None
        return section if getattr(section, "enabled", False) else None

    async def ensure_channel(self, name: str) -> BaseChannel | None:
        """Return an existing channel or create it from config."""
        if name in self.channels:
            return self.channels[name]

        config = self._get_channel_config(name)
        if config is None:
            logger.warning("Channel {} is not enabled in config", name)
            return None

        from aeloon.channels.registry import discover_all

        all_channels = discover_all()
        if name not in all_channels:
            logger.error("Channel {} not found in registry", name)
            return None

        try:
            cls = all_channels[name]
            channel = cls(config, self.bus)
            channel.transcription_api_key = self.config.providers.groq.api_key
            self.channels[name] = channel
            self._channel_states[name] = ChannelState.PENDING
            logger.info("{} channel instantiated", cls.display_name)
            return channel
        except Exception as e:
            logger.error("Failed to instantiate channel {}: {}", name, e)
            return None

    async def start_channel(self, name: str) -> bool:
        """Start one channel by name."""
        channel = await self.ensure_channel(name)
        if channel is None:
            return False

        if channel.is_running:
            logger.info("Channel {} is already running", name)
            return True

        task = asyncio.create_task(self._start_channel(name, channel))
        self._channel_tasks[name] = task

        logger.info("Started channel {} in background task", name)
        return True

    async def stop_channel(self, name: str) -> bool:
        """Stop one channel by name."""
        channel = self.channels.get(name)
        if channel is None:
            logger.warning("Channel {} not found", name)
            return False

        try:
            await channel.stop()
            self._channel_states[name] = ChannelState.STOPPED
            logger.info("Stopped channel {}", name)
        except Exception as e:
            logger.error("Error stopping channel {}: {}", name, e)

        task = self._channel_tasks.pop(name, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.channels.pop(name, None)

        return True

    async def reload_channel(self, name: str) -> bool:
        """Recreate and restart one channel."""
        logger.info("Reloading channel {}...", name)

        await self.stop_channel(name)

        self.channels.pop(name, None)
        self._channel_states.pop(name, None)
        self._channel_errors.pop(name, None)

        return await self.start_channel(name)
