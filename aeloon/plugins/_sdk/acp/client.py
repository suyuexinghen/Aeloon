"""ACP client — high-level interface for connecting to an ACP agent.

Provides connect/session/prompt/disconnect lifecycle using the
``agent-client-protocol`` SDK under the hood.

Content model
-------------
ACP agents send content through ``session_update`` notifications, not in
the ``PromptResponse`` itself.  The client accumulates text chunks from
agent message updates and returns them in :class:`DelegateResult`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from aeloon.plugins._sdk.acp.session import SessionMap
from aeloon.plugins._sdk.acp.transport import ACPTransport
from aeloon.plugins._sdk.acp.types import (
    ACPError,
    ACPLayer,
    BackendProfile,
    ConnectionState,
    DelegateResult,
    SessionInfo,
)

logger = logging.getLogger(__name__)

# Callback types
UpdateCallback = Callable[[str], Awaitable[None]]
"""Called with short text progress updates during prompt execution."""


def _extract_text_parts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_text_parts(item))
        return parts
    text = getattr(value, "text", None)
    if isinstance(text, str) and text:
        return [text]
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _extract_text_parts(content)
    message = getattr(value, "message", None)
    if isinstance(message, str) and message:
        return [message]
    delta = getattr(value, "delta", None)
    if delta is not None and delta is not value:
        return _extract_text_parts(delta)
    return []


def _extract_text_from_update(update: Any) -> str | None:
    """Extract plain text from an ACP session_update payload."""
    update_type = type(update).__name__

    if "ToolCall" in update_type:
        title = getattr(update, "title", None) or getattr(update, "tool_name", None) or ""
        status = getattr(update, "status", None) or ""
        if title:
            return f"[tool: {title} {status}]".strip()

    parts = _extract_text_parts(update)
    if parts:
        return "".join(parts)

    return None


class _StreamingCollector:
    """Collects text from ``session_update`` notifications.

    The handler runs inside the SDK's async dispatch — the
    ``_ACPSessionUpdateHandler.session_update`` is ``async`` and calls
    this collector synchronously from within its body.

    If an *on_progress* callback is provided, the collector will fire it
    asynchronously for each text snippet so the user sees output in
    real time.
    """

    def __init__(self, on_progress: UpdateCallback | None = None) -> None:
        self.chunks: list[str] = []
        self._on_progress = on_progress
        self.update_types: list[str] = []
        self.unknown_update_types: list[str] = []
        self._buffer: list[str] = []
        self._last_flush: float = 0.0

    _FLUSH_INTERVAL = 0.15  # seconds between progress flushes
    _FLUSH_MIN_CHARS = 12  # minimum chars before flushing

    def _try_flush(self, text: str) -> None:
        """Buffer text and flush when enough content or time has elapsed."""
        self._buffer.append(text)
        joined = "".join(self._buffer)
        now = time.monotonic()
        elapsed = now - self._last_flush
        if len(joined) >= self._FLUSH_MIN_CHARS or elapsed >= self._FLUSH_INTERVAL:
            self._buffer.clear()
            self._last_flush = now
            if self._on_progress is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._on_progress(joined))
                except RuntimeError:
                    pass

    def _final_flush(self) -> None:
        """Flush any remaining buffered content."""
        if self._buffer and self._on_progress is not None:
            joined = "".join(self._buffer)
            self._buffer.clear()
            self._last_flush = time.monotonic()
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._on_progress(joined))
            except RuntimeError:
                pass

    def __call__(self, session_id: str, update: Any, **kwargs: Any) -> None:
        update_type = type(update).__name__
        self.update_types.append(update_type)
        logger.debug("ACP session_update: type=%s", update_type)
        text = _extract_text_from_update(update)
        if text:
            self.chunks.append(text)
            self._try_flush(text)
        else:
            self.unknown_update_types.append(update_type)
            logger.info("ACP session_update produced no text: type=%s", update_type)


class ACPClient:
    """High-level ACP client managing one backend connection.

    Usage::

        client = ACPClient()
        await client.connect(profile)
        result = await client.prompt("aeloon-session-1", "do something")
        await client.disconnect()
    """

    def __init__(self) -> None:
        self._transport = ACPTransport()
        self._session_map = SessionMap()
        self._on_update: UpdateCallback | None = None

    @property
    def state(self) -> ConnectionState:
        return self._transport.state

    @property
    def last_error(self) -> ACPError | None:
        return self._transport.last_error

    @property
    def session_map(self) -> SessionMap:
        return self._session_map

    @property
    def is_connected(self) -> bool:
        return self._transport.state == ConnectionState.CONNECTED

    def set_update_callback(self, cb: UpdateCallback | None) -> None:
        self._on_update = cb

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, profile: BackendProfile) -> None:
        """Spawn the backend process, perform ACP handshake."""
        if self.is_connected:
            logger.warning("ACP client already connected, disconnecting first")
            await self.disconnect()

        await self._transport.start(
            command=profile.command,
            cwd=profile.cwd,
            env=profile.env,
        )

        conn = self._transport.connection
        if conn is None:
            error = ACPError(
                layer=ACPLayer.TRANSPORT,
                message="Transport started but no connection available",
            )
            raise RuntimeError(str(error))

        try:
            await conn.initialize(protocol_version=1)
            logger.info("ACP handshake complete for profile '%s'", profile.name)
        except Exception as exc:
            error = ACPError(
                layer=ACPLayer.HANDSHAKE,
                message=f"Handshake failed: {exc}",
            )
            logger.error("ACP handshake error: %s", exc)
            await self.disconnect()
            raise RuntimeError(str(error)) from exc

    async def disconnect(self) -> None:
        """Clean up all sessions and shut down transport."""
        self._session_map.clear()
        await self._transport.stop()
        logger.info("ACP client disconnected")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def create_session(
        self,
        aeloon_session_key: str,
        cwd: str | None = None,
    ) -> str:
        """Create a new ACP session mapped to the Aeloon session key."""
        conn = self._require_connection()

        try:
            response = await conn.new_session(cwd=cwd or ".", mcp_servers=[])
            acp_session_id = response.session_id
        except Exception as exc:
            error = ACPError(layer=ACPLayer.SESSION, message=f"Failed to create session: {exc}")
            logger.error("ACP session create error: %s", exc)
            raise RuntimeError(str(error)) from exc

        info = SessionInfo(
            acp_session_id=acp_session_id,
            aeloon_session_key=aeloon_session_key,
        )
        self._session_map.set(info)
        logger.info(
            "ACP session created: aeloon=%s -> acp=%s",
            aeloon_session_key,
            acp_session_id,
        )
        return acp_session_id

    async def load_session(
        self,
        aeloon_session_key: str,
        acp_session_id: str,
        cwd: str | None = None,
    ) -> str:
        """Load an existing ACP session and map it."""
        conn = self._require_connection()

        try:
            await conn.load_session(session_id=acp_session_id, cwd=cwd or ".")
        except Exception as exc:
            error = ACPError(
                layer=ACPLayer.SESSION,
                message=f"Failed to load session {acp_session_id}: {exc}",
            )
            logger.error("ACP session load error: %s", exc)
            raise RuntimeError(str(error)) from exc

        info = SessionInfo(
            acp_session_id=acp_session_id,
            aeloon_session_key=aeloon_session_key,
        )
        self._session_map.set(info)
        return acp_session_id

    def get_session(self, aeloon_session_key: str) -> SessionInfo | None:
        return self._session_map.get(aeloon_session_key)

    def remove_session(self, aeloon_session_key: str) -> SessionInfo | None:
        return self._session_map.remove(aeloon_session_key)

    # ------------------------------------------------------------------
    # Prompt execution
    # ------------------------------------------------------------------

    async def prompt(
        self,
        aeloon_session_key: str,
        text: str,
    ) -> DelegateResult:
        """Send a prompt to the ACP backend for the given Aeloon session.

        Content from the agent arrives via ``session_update`` notifications.
        """
        conn = self._require_connection()

        # Ensure session exists
        session_info = self._session_map.get(aeloon_session_key)
        if session_info is None:
            acp_session_id = await self.create_session(aeloon_session_key)
        else:
            acp_session_id = session_info.acp_session_id

        # Set up streaming collector
        collector = _StreamingCollector(on_progress=self._on_update)
        self._transport.client_handler.set_handler(collector)

        try:
            from acp import text_block

            response = await conn.prompt(
                session_id=acp_session_id,
                prompt=[text_block(text)],
            )
        except Exception as exc:
            exc_msg = str(exc)
            layer = ACPLayer.EXECUTION
            if "Authentication required" in exc_msg or "auth" in exc_msg.lower():
                layer = ACPLayer.HANDSHAKE
                exc_msg = (
                    "Authentication required. Run 'claude login' first, or set ANTHROPIC_API_KEY."
                )
            elif "session" in exc_msg.lower():
                layer = ACPLayer.SESSION

            error = ACPError(
                layer=layer,
                message=f"Prompt execution failed: {exc_msg}",
                details={"session_id": acp_session_id},
            )
            logger.error("ACP prompt error: %s", exc)
            raise RuntimeError(str(error)) from exc
        finally:
            collector._final_flush()
            self._transport.client_handler.set_handler(None)

        # Update last-active timestamp
        if session_info is not None:
            session_info.last_active = datetime.now()

        content = "".join(collector.chunks) if collector.chunks else ""
        logger.info(
            "ACP prompt completed: %d chunks, %d chars, stop=%s",
            len(collector.chunks),
            len(content),
            getattr(response, "stop_reason", None),
        )

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else {}

        return DelegateResult(
            content=content,
            usage=usage,
            execution_meta={
                "acp_session_id": acp_session_id,
                "stop_reason": getattr(response, "stop_reason", None),
                "update_types": list(collector.update_types),
                "unknown_update_types": list(collector.unknown_update_types),
            },
        )

    async def cancel(self, aeloon_session_key: str) -> None:
        """Cancel in-progress prompt for the given session."""
        conn = self._require_connection()
        session_info = self._session_map.get(aeloon_session_key)
        if session_info is None:
            logger.warning("cancel requested but no session mapped for %s", aeloon_session_key)
            return

        try:
            await conn.cancel(session_id=session_info.acp_session_id)
            logger.info("ACP session cancelled: acp=%s", session_info.acp_session_id)
        except Exception as exc:
            logger.warning("ACP cancel failed (may be unsupported): %s", exc)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        sessions = self._session_map.all_sessions()
        return {
            "state": self.state.value,
            "connected": self.is_connected,
            "sessions": len(sessions),
            "last_error": str(self.last_error) if self.last_error else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connection(self) -> Any:
        conn = self._transport.connection
        if conn is None or not self.is_connected:
            raise RuntimeError("ACP client is not connected. Use /acp connect first.")
        return conn
