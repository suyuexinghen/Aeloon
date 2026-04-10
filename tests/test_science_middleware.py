"""Tests for BudgetMiddleware and AuditMiddleware in aeloon/plugins/science/middleware/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.plugins.ScienceResearch.middleware.audit import AuditEventType, AuditMiddleware
from aeloon.plugins.ScienceResearch.middleware.budget import (
    BudgetExceededError,
    BudgetMiddleware,
    BudgetState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_budget_state(**overrides: Any) -> BudgetState:
    return BudgetState(**overrides)


def _make_budget_middleware(**state_overrides: Any) -> BudgetMiddleware:
    return BudgetMiddleware(budget_state=_make_budget_state(**state_overrides))


def _make_audit_middleware(tmp_path: Path) -> tuple[AuditMiddleware, Path]:
    audit_file = tmp_path / "audit.jsonl"
    return AuditMiddleware(
        trace_id="trace_001", task_id="task_001", audit_path=audit_file
    ), audit_file


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# BudgetState tests
# ---------------------------------------------------------------------------


def test_budget_state_not_over_budget_initially() -> None:
    state = BudgetState()
    assert state.is_over_budget is False


def test_budget_state_over_budget_when_tokens_exceeded() -> None:
    state = BudgetState(max_tokens=100, tokens_used=101)
    assert state.is_over_budget is True


# ---------------------------------------------------------------------------
# BudgetMiddleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_middleware_around_tool_increments_call_count() -> None:
    middleware = _make_budget_middleware()
    execute = AsyncMock(return_value="ok")

    await middleware.around_tool(name="my_tool", args={}, execute=execute)

    assert middleware.snapshot().tool_calls == 1


@pytest.mark.asyncio
async def test_budget_middleware_raises_when_tool_calls_exceeded() -> None:
    # Set tool_calls = max_tool_calls; next call should exceed it
    middleware = _make_budget_middleware(max_tool_calls=5, tool_calls=5)
    execute = AsyncMock(return_value="ok")

    with pytest.raises(BudgetExceededError):
        await middleware.around_tool(name="my_tool", args={}, execute=execute)


@pytest.mark.asyncio
async def test_budget_middleware_around_llm_adds_token_estimate() -> None:
    middleware = _make_budget_middleware()

    mock_response = MagicMock()
    mock_response.content = "hello world"  # 11 chars → 11 // 4 = 2 tokens
    call_llm = AsyncMock(return_value=mock_response)

    await middleware.around_llm(messages=[], tool_defs=[], call_llm=call_llm)

    snapshot = middleware.snapshot()
    assert snapshot.tokens_used == 2  # 11 // 4


def test_budget_middleware_snapshot_returns_copy() -> None:
    middleware = _make_budget_middleware(tokens_used=10)
    snap = middleware.snapshot()

    # Mutating the snapshot should not affect the internal state
    snap.tokens_used = 999
    assert middleware.snapshot().tokens_used == 10


# ---------------------------------------------------------------------------
# AuditMiddleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_middleware_around_tool_writes_event(tmp_path: Path) -> None:
    middleware, audit_file = _make_audit_middleware(tmp_path)
    execute = AsyncMock(return_value="result")

    await middleware.around_tool(name="my_tool", args={}, execute=execute)

    events = _read_jsonl(audit_file)
    assert len(events) == 1
    assert events[0]["event_type"] == AuditEventType.TOOL_CALL.value


@pytest.mark.asyncio
async def test_audit_middleware_around_tool_error_writes_error_event(tmp_path: Path) -> None:
    middleware, audit_file = _make_audit_middleware(tmp_path)
    execute = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await middleware.around_tool(name="my_tool", args={}, execute=execute)

    events = _read_jsonl(audit_file)
    assert len(events) == 1
    assert events[0]["event_type"] == AuditEventType.TOOL_ERROR.value


def test_audit_middleware_log_event_appends_jsonl(tmp_path: Path) -> None:
    from aeloon.plugins.ScienceResearch.middleware.audit import AuditEvent

    middleware, audit_file = _make_audit_middleware(tmp_path)

    event1 = AuditEvent(event_type=AuditEventType.NODE_START, trace_id="t1", task_id="task1")
    event2 = AuditEvent(event_type=AuditEventType.NODE_COMPLETE, trace_id="t1", task_id="task1")
    middleware.log_event(event1)
    middleware.log_event(event2)

    lines = audit_file.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "event_type" in obj


def test_audit_emit_convenience_method_writes_event(tmp_path: Path) -> None:
    middleware, audit_file = _make_audit_middleware(tmp_path)

    middleware.emit(AuditEventType.NODE_START, node_id="n1", detail="test")

    events = _read_jsonl(audit_file)
    assert len(events) == 1
    assert events[0]["event_type"] == AuditEventType.NODE_START.value
