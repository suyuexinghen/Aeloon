"""Tests for benchmark runner utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aeloon.core.agent.profiler import ProfileReport, ProfileSample
from benchmarks.runner import (
    ScenarioDefinition,
    ScenarioRunResult,
    aggregate_results,
    load_scenarios,
    run_scenarios,
)


def _make_report(
    total: float,
    llm: float,
    tools: float,
    context: float,
    session_io: float,
    llm_calls: int = 1,
    tool_calls: int = 1,
) -> ProfileReport:
    return ProfileReport(
        total_ms=total,
        llm_calls=[ProfileSample(label="model", duration_ms=llm / max(llm_calls, 1))] * llm_calls,
        tool_calls=[ProfileSample(label="tool", duration_ms=tools / max(tool_calls, 1))]
        * tool_calls,
        context_build_ms=context,
        session_io_ms=session_io,
    )


@dataclass
class _FakeProfiler:
    enabled: bool
    last_report: ProfileReport | None = None


class _FakeLoop:
    def __init__(self, reports: list[ProfileReport]):
        self.profiler = _FakeProfiler(enabled=False)
        self._reports = iter(reports)
        self.calls: list[tuple[str, str]] = []

    async def process_direct(
        self,
        content: str,
        session_key: str = "",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress=None,
    ) -> str:
        del channel, chat_id, on_progress
        self.calls.append((content, session_key))
        self.profiler.last_report = next(self._reports)
        return "ok"


def test_load_scenarios_supports_group_filter() -> None:
    all_scenarios = load_scenarios()
    only_simple = load_scenarios("simple")

    assert all_scenarios
    assert only_simple
    assert all(s.name.startswith("simple/") for s in only_simple)


def test_aggregate_results_computes_avg_min_max() -> None:
    rows = [
        ScenarioRunResult(
            scenario="simple/math",
            run_index=1,
            total_ms=100.0,
            llm_ms=70.0,
            tools_ms=10.0,
            context_ms=5.0,
            session_io_ms=3.0,
            llm_calls=1,
            tool_calls=1,
        ),
        ScenarioRunResult(
            scenario="simple/math",
            run_index=2,
            total_ms=200.0,
            llm_ms=140.0,
            tools_ms=20.0,
            context_ms=6.0,
            session_io_ms=4.0,
            llm_calls=1,
            tool_calls=1,
        ),
    ]

    agg = aggregate_results(rows)

    assert len(agg) == 1
    assert agg[0].runs == 2
    assert agg[0].total_ms_avg == 150.0
    assert agg[0].total_ms_min == 100.0
    assert agg[0].total_ms_max == 200.0


@pytest.mark.asyncio
async def test_run_scenarios_collects_and_sums_turn_reports() -> None:
    scenario = ScenarioDefinition(name="simple/test", turns=["turn-1", "turn-2"])
    reports = [
        _make_report(total=100, llm=60, tools=10, context=5, session_io=3),
        _make_report(total=120, llm=70, tools=12, context=6, session_io=4),
        _make_report(total=90, llm=50, tools=8, context=4, session_io=2),
        _make_report(total=110, llm=55, tools=9, context=5, session_io=3),
    ]
    loop = _FakeLoop(reports)

    rows = await run_scenarios(loop, [scenario], repeat=2)

    assert len(rows) == 2
    assert rows[0].scenario == "simple/test"
    assert rows[0].run_index == 1
    assert rows[0].total_ms == 220
    assert rows[0].llm_ms == 130
    assert rows[0].tools_ms == 22
    assert rows[0].context_ms == 11
    assert rows[0].session_io_ms == 7
    assert rows[1].total_ms == 200
    assert len(loop.calls) == 4
    assert loop.profiler.enabled is False
