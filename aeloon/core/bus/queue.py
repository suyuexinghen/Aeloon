"""Message queues used by Aeloon."""

import asyncio

from aeloon.core.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """Async queues between channels and the agent."""

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Add an inbound message."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Get the next inbound message."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Add an outbound message."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Get the next outbound message."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Return the inbound queue size."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Return the outbound queue size."""
        return self.outbound.qsize()
