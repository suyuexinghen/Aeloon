"""Config-driven hook handlers — register hooks from ``~/.aeloon/config.json``.

This module implements Phase 4 of the hook roadmap: users can declare hooks
in the ``hooks`` section of their configuration file without writing Python
plugins. Two handler backends are supported:

* **command** — run a shell command; parse exit code + JSON stdout for
  :class:`HookDecision` results (Claude Code's exit-code protocol).
* **http** — POST the event payload to a URL; parse the JSON response body.

:class:`ConfigHookAdapter` bridges config declarations into the existing
:class:`~aeloon.plugins._sdk.registry.PluginRegistry` as virtual plugin hooks
under the ``aeloon.config_hooks`` plugin ID.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from loguru import logger

from aeloon.plugins._sdk.hooks import HookEvent
from aeloon.plugins._sdk.types import HookDecision, HookRecord

if TYPE_CHECKING:
    from aeloon.plugins._sdk.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIRTUAL_PLUGIN_ID = "aeloon.config_hooks"

# ---------------------------------------------------------------------------
# HookHandler ABC
# ---------------------------------------------------------------------------


class HookHandler(ABC):
    """Base class for config-driven hook execution backends."""

    @abstractmethod
    async def execute(self, event: str, payload: dict[str, Any]) -> HookDecision | None:
        """Execute the hook.

        Returns:
            A :class:`HookDecision` for guard hooks, or ``None`` for
            fire-and-forget (no opinion / allow).
        """
        ...


# ---------------------------------------------------------------------------
# CommandHookHandler
# ---------------------------------------------------------------------------


class CommandHookHandler(HookHandler):
    """Execute a shell command and parse the exit-code protocol.

    Protocol (borrowed from Claude Code):

    * **Exit 0 + JSON stdout** with ``{"allow": false, "reason": "..."}`` → deny.
    * **Exit 0 + JSON stdout** with ``{"modified_value": ...}`` → allow + modify.
    * **Exit 0 + non-JSON stdout** → ``None`` (no opinion).
    * **Exit 2** → deny with reason from stderr.
    * **Timeout / other exit** → ``None`` (log warning).
    * **async_exec=True** → fire-and-forget, always returns ``None``.
    """

    def __init__(self, command: str, *, timeout: int = 600, async_exec: bool = False) -> None:
        self._command = command
        self._timeout = timeout
        self._async_exec = async_exec

    async def execute(self, event: str, payload: dict[str, Any]) -> HookDecision | None:
        # Template-substitute the command with payload values.
        mapping = defaultdict(str, payload)
        rendered = self._command.format_map(mapping)

        if self._async_exec:
            asyncio.create_task(self._run_command(rendered, payload))
            return None

        return await self._run_command(rendered, payload)

    async def _run_command(self, command: str, payload: dict[str, Any]) -> HookDecision | None:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdin_data = json.dumps(payload).encode()
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin_data),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Command hook timed out after {}s: {}", self._timeout, command)
                return None

            if proc.returncode == 2:
                return HookDecision(allow=False, reason=stderr.decode(errors="replace"))

            if proc.returncode == 0:
                stdout_text = stdout.decode(errors="replace").strip()
                if stdout_text:
                    try:
                        data = json.loads(stdout_text)
                        if isinstance(data, dict):
                            allow = data.get("allow", True)
                            reason = data.get("reason", "")
                            modified = data.get("modified_value")
                            return HookDecision(
                                allow=bool(allow),
                                reason=str(reason),
                                modified_value=modified,
                            )
                    except json.JSONDecodeError:
                        pass
                return None

            logger.warning(
                "Command hook exited with code {}: {}",
                proc.returncode,
                command,
            )
            return None

        except Exception:
            logger.opt(exception=True).warning("Command hook execution failed: {}", command)
            return None


# ---------------------------------------------------------------------------
# HttpHookHandler
# ---------------------------------------------------------------------------


class HttpHookHandler(HookHandler):
    """POST the event payload to a URL and parse the JSON response.

    Protocol:

    * **200 + JSON body** with decision fields → :class:`HookDecision`.
    * **200 + non-JSON / empty** → ``None`` (no opinion).
    * **Non-200 / timeout / error** → log warning, ``None``.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        method: str = "POST",
        timeout: int = 30,
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._method = method.upper()
        self._timeout = timeout

    async def execute(self, event: str, payload: dict[str, Any]) -> HookDecision | None:
        # Expand env vars in header values at execution time.
        resolved_headers = {k: os.path.expandvars(v) for k, v in self._headers.items()}

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(
                    method=self._method,
                    url=self._url,
                    json=payload,
                    headers=resolved_headers,
                )

            if response.status_code != 200:
                logger.warning(
                    "HTTP hook returned status {}: {}",
                    response.status_code,
                    self._url,
                )
                return None

            try:
                data = response.json()
                if isinstance(data, dict):
                    allow = data.get("allow", True)
                    reason = data.get("reason", "")
                    modified = data.get("modified_value")
                    return HookDecision(
                        allow=bool(allow),
                        reason=str(reason),
                        modified_value=modified,
                    )
            except (json.JSONDecodeError, ValueError):
                pass
            return None

        except Exception:
            logger.opt(exception=True).warning("HTTP hook execution failed: {}", self._url)
            return None


# ---------------------------------------------------------------------------
# ConfigHookAdapter
# ---------------------------------------------------------------------------


class ConfigHookAdapter:
    """Load hooks from config and register them as a virtual plugin.

    Usage::

        adapter = ConfigHookAdapter(registry)
        adapter.load_from_config(config.hooks)
    """

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    def load_from_config(self, hooks_config: dict[str, list[Any]]) -> None:
        """Parse the ``hooks`` section from config and register handlers.

        Each entry produces a :class:`HookRecord` with ``plugin_id`` of
        ``"aeloon.config_hooks"``.

        Invalid event names, bad matcher regex, and malformed entries are
        logged and skipped — they never crash the agent.
        """
        # Rollback any previous config hooks (supports future reload).
        self._registry.rollback_plugin(_VIRTUAL_PLUGIN_ID)

        known_events = {e.value for e in HookEvent}
        records: list[HookRecord] = []

        for event_name, entries in hooks_config.items():
            if event_name not in known_events:
                logger.warning("Config hook: unknown event '{}', skipping", event_name)
                continue

            for idx, entry in enumerate(entries):
                try:
                    # Support both Pydantic model and raw dict.
                    if hasattr(entry, "handler"):
                        handler_cfg = entry.handler
                        matcher = getattr(entry, "matcher", None)
                        priority = getattr(entry, "priority", 0)
                    else:
                        handler_cfg = entry["handler"]
                        matcher = entry.get("matcher")
                        priority = entry.get("priority", 0)

                    hook_handler = self._build_handler(handler_cfg)
                    wrapper = self._build_wrapper(event_name, hook_handler)

                    # Validate matcher regex.
                    if matcher is not None:
                        re.compile(matcher)

                    records.append(
                        HookRecord(
                            plugin_id=_VIRTUAL_PLUGIN_ID,
                            event=event_name,
                            kind="notify",
                            priority=priority,
                            handler=wrapper,
                            matcher=matcher,
                        )
                    )
                except Exception:
                    logger.opt(exception=True).warning(
                        "Config hook: failed to parse entry {} for event '{}'",
                        idx,
                        event_name,
                    )

        if records:
            self._registry.commit_plugin(_VIRTUAL_PLUGIN_ID, hooks=records)
            logger.info("Config hooks: registered {} handler(s)", len(records))

    def _build_handler(self, config: Any) -> HookHandler:
        """Factory: create a :class:`HookHandler` from config."""
        if hasattr(config, "type"):
            # Pydantic model.
            cfg_type = config.type
            if cfg_type == "command":
                return CommandHookHandler(
                    command=config.command,
                    timeout=config.timeout,
                    async_exec=config.async_exec,
                )
            if cfg_type == "http":
                return HttpHookHandler(
                    url=config.url,
                    headers=config.headers,
                    method=config.method,
                    timeout=config.http_timeout,
                )
        else:
            # Raw dict fallback.
            cfg_type = config.get("type", "command") if isinstance(config, dict) else "command"
            if cfg_type == "command":
                return CommandHookHandler(
                    command=config.get("command", ""),
                    timeout=config.get("timeout", 600),
                    async_exec=config.get("async_exec", False),
                )
            if cfg_type == "http":
                return HttpHookHandler(
                    url=config.get("url", ""),
                    headers=config.get("headers"),
                    method=config.get("method", "POST"),
                    timeout=config.get("http_timeout", 30),
                )
        raise ValueError(f"Unknown hook handler type: {config}")

    @staticmethod
    def _build_wrapper(event: str, handler: HookHandler) -> Any:
        """Wrap a :class:`HookHandler` into a ``Callable[..., Any]`` for :class:`HookRecord`.

        The wrapper is an ``async def`` so the dispatcher's
        ``inspect.isawaitable`` check will await it.
        """

        async def _wrapper(*args: Any, **kwargs: Any) -> HookDecision | None:
            value = args[0] if args else None
            payload = dict(kwargs)
            if value is not None:
                payload["value"] = value
            payload["event"] = event
            return await handler.execute(event, payload)

        return _wrapper
