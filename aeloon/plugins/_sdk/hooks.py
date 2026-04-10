"""Hook type definitions and dispatcher.

The hook system allows plugins to participate in lifecycle events and
data transformations without coupling to core internals.  Four dispatch
modes are supported:

* **NOTIFY** — fire-and-forget; handler errors are logged but do not
  propagate.
* **MUTATE** — value is piped through a chain of handlers; each handler
  receives the previous handler's return value.
* **REDUCE** — all handler return values are collected into a list.
* **GUARD** — handlers can allow, deny, or modify the proceeding action;
  the first ``deny`` wins.
"""

from __future__ import annotations

import inspect
import re
from enum import Enum
from typing import Any

from loguru import logger

from aeloon.plugins._sdk.types import HookDecision, HookRecord


class HookType(str, Enum):
    """Dispatch mode for a hook handler."""

    NOTIFY = "notify"
    MUTATE = "mutate"
    REDUCE = "reduce"
    GUARD = "guard"


class HookEvent(str, Enum):
    """Well-known lifecycle events that plugins may hook into."""

    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"
    PLUGIN_REGISTERED = "plugin_registered"
    PLUGIN_ACTIVATED = "plugin_activated"
    PLUGIN_DEACTIVATED = "plugin_deactivated"
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    CONFIG_RELOADED = "config_reloaded"
    STATUS_INVALIDATE = "status_invalidate"

    # --- Tool call failures ---
    TOOL_CALL_FAILURE = "tool_call_failure"

    # --- Permission control (dispatch points TBD) ---
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_DENIED = "permission_denied"

    # --- Context compaction (dispatch points TBD) ---
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"

    # --- Sub-agent lifecycle (dispatch points TBD) ---
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"

    # --- File watching (dispatch points TBD) ---
    FILE_CHANGED = "file_changed"


class HookDispatcher:
    """Dispatches hook events to registered handlers.

    Handlers are stored externally in
    :class:`~aeloon.plugins._sdk.registry.PluginRegistry`; the dispatcher
    receives a reference to the registry's ``hooks_for_event`` method at
    construction time so it can look up handlers lazily.
    """

    def __init__(self, hooks_for_event: Any = None) -> None:
        self._hooks_for_event = hooks_for_event
        # Fallback local storage when no registry is wired.
        self._local_hooks: dict[str, list[HookRecord]] = {}

    def set_hooks_source(self, hooks_for_event: Any) -> None:
        """Wire the dispatcher to a registry lookup function."""
        self._hooks_for_event = hooks_for_event

    def _get_handlers(self, event: str, match_value: str | None = None) -> list[HookRecord]:
        if self._hooks_for_event is not None:
            records = self._hooks_for_event(event)
        else:
            records = self._local_hooks.get(event, [])
        if match_value is None:
            return records
        return [r for r in records if r.matcher is None or re.fullmatch(r.matcher, match_value)]

    # ------------------------------------------------------------------
    # Dispatch modes
    # ------------------------------------------------------------------

    async def dispatch_notify(
        self, event: str | HookEvent, *, match_value: str | None = None, **kwargs: Any
    ) -> None:
        """Fire-and-forget: call all handlers, log errors, never propagate."""
        event_value = event.value if isinstance(event, HookEvent) else event
        for record in self._get_handlers(event_value, match_value):
            try:
                result = record.handler(**kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.opt(exception=True).warning(
                    "Hook handler error for event '{}' from plugin '{}'",
                    event_value,
                    record.plugin_id,
                )

    async def dispatch_mutate(
        self,
        event: str | HookEvent,
        value: Any,
        *,
        match_value: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Chain: pipe *value* through handlers in priority order.

        Each handler receives the return value of the previous one.
        If a handler returns ``None``, the chain stops and the current
        value is returned.
        """
        event_value = event.value if isinstance(event, HookEvent) else event
        for record in self._get_handlers(event_value, match_value):
            try:
                result = record.handler(value, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                if result is None:
                    break
                value = result
            except Exception:
                logger.opt(exception=True).warning(
                    "Hook mutate handler error for event '{}' from plugin '{}'",
                    event_value,
                    record.plugin_id,
                )
        return value

    async def dispatch_reduce(
        self, event: str | HookEvent, *, match_value: str | None = None, **kwargs: Any
    ) -> list[Any]:
        """Collect: gather all handler return values into a list."""
        event_value = event.value if isinstance(event, HookEvent) else event
        results: list[Any] = []
        for record in self._get_handlers(event_value, match_value):
            try:
                result = record.handler(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
                results.append(result)
            except Exception:
                logger.opt(exception=True).warning(
                    "Hook reduce handler error for event '{}' from plugin '{}'",
                    event_value,
                    record.plugin_id,
                )
        return results

    async def dispatch_guard(
        self,
        event: str | HookEvent,
        value: Any = None,
        *,
        match_value: str | None = None,
        **kwargs: Any,
    ) -> HookDecision:
        """Guard: handlers can allow, deny, or modify the proceeding action.

        Returns the aggregated :class:`HookDecision`.  If any handler
        denies (``allow=False``), the action is blocked immediately
        (deny wins over allow, matching Claude Code's precedence model).

        Handler return types:
        * :class:`HookDecision` — full control.
        * ``bool`` — ``False`` means deny with a generic reason.
        * ``None`` / anything else — treated as allow (no opinion).

        On handler exception, the default is **deny** (fail-closed).
        """
        event_value = event.value if isinstance(event, HookEvent) else event
        decision = HookDecision()

        for record in self._get_handlers(event_value, match_value):
            try:
                result = (
                    record.handler(value, **kwargs)
                    if value is not None
                    else record.handler(**kwargs)
                )
                if inspect.isawaitable(result):
                    result = await result

                if isinstance(result, HookDecision):
                    if not result.allow:
                        return result  # deny wins immediately
                    if result.modified_value is not None:
                        decision.modified_value = result.modified_value
                        value = result.modified_value
                    if result.reason:
                        decision.reason = result.reason
                elif isinstance(result, bool):
                    if not result:
                        return HookDecision(
                            allow=False, reason=f"Blocked by plugin '{record.plugin_id}'"
                        )
                # None or other → treat as allow (no opinion)
            except Exception:
                logger.opt(exception=True).warning(
                    "Guard hook error for event '{}' from plugin '{}'",
                    event_value,
                    record.plugin_id,
                )
                # Fail-closed: deny on error
                return HookDecision(
                    allow=False, reason=f"Guard hook error in plugin '{record.plugin_id}'"
                )

        return decision
