"""Turn-level routing context for agent processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnContext:
    """Per-turn routing and session metadata."""

    channel: str
    chat_id: str
    message_id: str | None = None
    session_key: str = ""
    sender_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
