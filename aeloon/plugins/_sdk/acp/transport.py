"""ACP transport — spawns and manages an ACP agent process over stdio.

Wraps the ``agent-client-protocol`` SDK's ``spawn_agent_process`` to
provide start/stop lifecycle and connection reference tracking.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from aeloon.plugins._sdk.acp.types import ACPError, ACPLayer, ConnectionState

logger = logging.getLogger(__name__)


class _ACPSessionUpdateHandler:
    """Callback target for ACP session_update notifications.

    Installed as the ``Client`` implementation when spawning the agent process.
    The SDK dispatches notifications **asynchronously** — all handler methods
    MUST be ``async def`` or the SDK will silently suppress the ``TypeError``
    from ``await None``.
    """

    def __init__(self) -> None:
        self._handler: Any = None

    def set_handler(self, handler: Any) -> None:
        self._handler = handler

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Called by the SDK for every agent notification (text, tool, thought, etc.)."""
        if self._handler is not None:
            self._handler(session_id, update, **kwargs)

    async def request_permission(
        self, options: Any, session_id: str, tool_call: Any, **kwargs: Any
    ) -> Any:
        """Called by the SDK when the agent requests permission for a risky action.

        ``options`` is a list of ``PermissionOption`` objects with ``kind`` and
        ``option_id``.  We auto-approve by selecting the first ``allow_once``
        option.  If none exists, we cancel.
        """
        from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

        # Find an allow option — prefer allow_once
        for opt in options or []:
            kind = getattr(opt, "kind", "")
            if kind in ("allow_once", "allow_always"):
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(option_id=opt.option_id, outcome="selected"),
                )

        # No allow option available — cancel
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    def on_connect(self, conn: Any) -> None:
        """Called by SDK after connection is established (sync — SDK calls this synchronously)."""
        pass

    async def read_text_file(self, path: str, session_id: str, **kwargs: Any) -> Any:
        """Agent requests to read a file — deny for now."""
        from acp import ReadTextFileResponse

        return ReadTextFileResponse(content="")

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> None:
        """Agent requests to write a file — deny for now."""
        return None

    async def create_terminal(self, command: str, session_id: str, **kwargs: Any) -> Any:
        """Agent requests to create a terminal — deny for now."""
        return None

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        """Agent requests terminal output — not supported."""
        return None

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        """Agent requests to wait for terminal exit — not supported."""
        return None

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        """Agent requests to kill terminal — not supported."""
        return None

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        """Agent requests to release terminal — not supported."""
        return None


class ACPTransport:
    """Manages a single ACP agent subprocess connected over stdio.

    The transport holds the ``spawn_agent_process`` context manager open
    for the duration of the connection.  Call :meth:`start` to spawn and
    :meth:`stop` to clean up.
    """

    def __init__(self) -> None:
        self._state: ConnectionState = ConnectionState.DISCONNECTED
        self._conn: Any = None
        self._process: Any = None
        self._cm: Any = None  # the async context manager from spawn_agent_process
        self._last_error: ACPError | None = None
        self._client_handler = _ACPSessionUpdateHandler()

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def last_error(self) -> ACPError | None:
        return self._last_error

    @property
    def connection(self) -> Any:
        return self._conn

    @property
    def client_handler(self) -> _ACPSessionUpdateHandler:
        return self._client_handler

    @property
    def is_alive(self) -> bool:
        """Check if the managed process is still running."""
        return self._process is not None and self._process.returncode is None

    async def start(
        self,
        command: list[str],
        cwd: str = "~",
        env: dict[str, str] | None = None,
    ) -> None:
        """Spawn the ACP agent process and establish stdio connection."""
        if self._state == ConnectionState.CONNECTED:
            return

        self._state = ConnectionState.CONNECTING
        self._last_error = None

        try:
            from acp import spawn_agent_process

            cmd, *args = command

            # Expand ~ in cwd to actual home directory
            resolved_cwd = str(Path(cwd).expanduser())

            # Merge profile env with current process env so variables like
            # ANTHROPIC_API_KEY are inherited by the child process.
            merged_env: dict[str, str] = dict(os.environ)
            if env:
                merged_env.update(env)

            # Pass our handler as the Client — the SDK will call its
            # async session_update and request_permission methods.
            self._cm = spawn_agent_process(
                lambda _agent: self._client_handler,
                cmd,
                *args,
                cwd=resolved_cwd,
                env=merged_env,
            )
            conn, proc = await self._cm.__aenter__()
            self._conn = conn
            self._process = proc
            self._state = ConnectionState.CONNECTED
            logger.info("ACP transport connected: %s", " ".join(command))

        except FileNotFoundError:
            error = ACPError(
                layer=ACPLayer.TRANSPORT,
                message=f"Command not found: {command[0]}",
                details={"command": command},
            )
            self._last_error = error
            self._state = ConnectionState.ERROR
            raise

        except Exception as exc:
            error = ACPError(
                layer=ACPLayer.TRANSPORT,
                message=f"Failed to start transport: {exc}",
                details={"command": command},
            )
            self._last_error = error
            self._state = ConnectionState.ERROR
            raise

    async def stop(self) -> None:
        """Shut down the transport and terminate the process."""
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception as exc:
                logger.debug("Error closing ACP connection: %s", exc)
            self._conn = None

        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("Error closing ACP context manager: %s", exc)
            self._cm = None

        self._process = None
        self._state = ConnectionState.DISCONNECTED
        logger.info("ACP transport stopped")
