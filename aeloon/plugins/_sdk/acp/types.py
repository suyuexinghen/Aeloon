"""Local ACP-facing types for the internal client layer.

These types are internal to Aeloon and do not depend on the ACP SDK schema
directly — they represent the bridge's own view of the ACP world.
"""

from __future__ import annotations

import dataclasses
import enum
from datetime import datetime
from typing import Any


class ConnectionState(enum.Enum):
    """Lifecycle state of an ACP connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class ACPLayer(enum.Enum):
    """Which layer produced an error."""

    TRANSPORT = "transport"
    HANDSHAKE = "handshake"
    SESSION = "session"
    EXECUTION = "execution"


@dataclasses.dataclass
class ACPError:
    """Structured error from the ACP bridge."""

    layer: ACPLayer
    message: str
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.layer.value}] {self.message}"


@dataclasses.dataclass
class SessionInfo:
    """Tracked state for an ACP session."""

    acp_session_id: str
    aeloon_session_key: str
    created_at: datetime = dataclasses.field(default_factory=datetime.now)
    last_active: datetime = dataclasses.field(default_factory=datetime.now)


@dataclasses.dataclass
class BackendProfile:
    """Resolved configuration for connecting to one ACP backend."""

    name: str
    command: list[str]
    cwd: str = "~"
    timeout_seconds: float = 30.0
    env: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DelegateResult:
    """Normalized result from a delegated prompt execution."""

    content: str
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    execution_meta: dict[str, Any] = dataclasses.field(default_factory=dict)
