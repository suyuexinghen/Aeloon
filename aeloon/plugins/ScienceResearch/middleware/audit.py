"""Audit middleware: append-only structured event log for science task tracing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable

from loguru import logger
from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    """Types of structured events recorded in the audit log."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_ERROR = "tool_error"
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    NODE_FAIL = "node_fail"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"


class AuditEvent(BaseModel):
    """A single immutable entry in the audit log."""

    event_type: AuditEventType
    trace_id: str
    task_id: str
    node_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None


class AuditMiddleware:
    """Middleware that records structured events to an append-only JSONL file."""

    def __init__(self, trace_id: str, task_id: str, audit_path: Path) -> None:
        self._trace_id = trace_id
        self._task_id = task_id
        self._audit_path = audit_path

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log_event(self, event: AuditEvent) -> None:
        """Append one JSON line to the audit file (creates file if needed)."""
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
            with self._audit_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            logger.error("AuditMiddleware: failed to write audit event: {}", exc)

    def emit(self, event_type: AuditEventType, **details: Any) -> None:
        """Convenience wrapper: create an AuditEvent and log it."""
        event = AuditEvent(
            event_type=event_type,
            trace_id=self._trace_id,
            task_id=self._task_id,
            details=details,
        )
        self.log_event(event)

    # ------------------------------------------------------------------
    # Middleware hooks
    # ------------------------------------------------------------------

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable[Any]],
    ) -> Any:
        """Record an LLM_CALL event with timing and content length."""
        t0 = perf_counter()
        response = await call_llm(messages, tool_defs)
        duration_ms = (perf_counter() - t0) * 1000

        details: dict[str, Any] = {
            "content_length": len(response.content or ""),
        }
        # Include model name if the response carries it
        model_name = getattr(response, "model", None)
        if model_name:
            details["model"] = model_name

        event = AuditEvent(
            event_type=AuditEventType.LLM_CALL,
            trace_id=self._trace_id,
            task_id=self._task_id,
            duration_ms=duration_ms,
            details=details,
        )
        self.log_event(event)
        return response

    async def around_tool(
        self,
        name: str,
        args: dict | list | None,
        execute: Callable[[], Awaitable[str]],
    ) -> str:
        """Record TOOL_CALL on success, TOOL_ERROR on exception (re-raises)."""
        t0 = perf_counter()
        try:
            result = await execute()
        except Exception as exc:
            duration_ms = (perf_counter() - t0) * 1000
            event = AuditEvent(
                event_type=AuditEventType.TOOL_ERROR,
                trace_id=self._trace_id,
                task_id=self._task_id,
                duration_ms=duration_ms,
                details={
                    "tool": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self.log_event(event)
            raise

        duration_ms = (perf_counter() - t0) * 1000
        event = AuditEvent(
            event_type=AuditEventType.TOOL_CALL,
            trace_id=self._trace_id,
            task_id=self._task_id,
            duration_ms=duration_ms,
            details={
                "tool": name,
                "result_length": len(result) if isinstance(result, str) else None,
            },
        )
        self.log_event(event)
        return result
