"""Message bus payloads used by Aeloon."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from a channel."""

    channel: str  # Source channel name.
    sender_id: str  # Sender ID from the channel.
    chat_id: str  # Chat or conversation ID.
    content: str  # Message text.
    timestamp: datetime = field(default_factory=datetime.now)  # Receive time.
    media: list[str] = field(default_factory=list)  # Attached media paths or URLs.
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data.
    session_key_override: str | None = None  # Optional session key override.

    @property
    def session_key(self) -> str:
        """Return the session key for this message."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
