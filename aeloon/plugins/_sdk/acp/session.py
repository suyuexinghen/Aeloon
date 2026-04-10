"""Session map — Aeloon session key to ACP session id mapping.

V1 uses an in-memory dict. Persistence will be added in E3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aeloon.plugins._sdk.acp.types import SessionInfo

logger = logging.getLogger(__name__)


class SessionMap:
    """Bidirectional lookup between Aeloon session keys and ACP session ids."""

    def __init__(self) -> None:
        self._by_aeloon: dict[str, SessionInfo] = {}

    def get(self, aeloon_session_key: str) -> SessionInfo | None:
        return self._by_aeloon.get(aeloon_session_key)

    def set(self, info: SessionInfo) -> None:
        self._by_aeloon[info.aeloon_session_key] = info
        logger.debug(
            "session mapped: aeloon=%s -> acp=%s",
            info.aeloon_session_key,
            info.acp_session_id,
        )

    def remove(self, aeloon_session_key: str) -> SessionInfo | None:
        info = self._by_aeloon.pop(aeloon_session_key, None)
        if info is not None:
            logger.debug(
                "session unmapped: aeloon=%s (was acp=%s)",
                aeloon_session_key,
                info.acp_session_id,
            )
        return info

    def has(self, aeloon_session_key: str) -> bool:
        return aeloon_session_key in self._by_aeloon

    def clear(self) -> None:
        self._by_aeloon.clear()

    def all_sessions(self) -> list[SessionInfo]:
        return list(self._by_aeloon.values())
