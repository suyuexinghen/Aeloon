"""Minimal benchmark runner used by CLI and tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class ScenarioDefinition:
    """One benchmark scenario."""

    name: str
    turns: list[str]


@dataclass(frozen=True)
class ScenarioRunResult:
    """One benchmark run result."""

    scenario: str
    run_index: int
    total_ms: float
    llm_ms: float
    tools_ms: float
    context_ms: float
    session_io_ms: float
    llm_calls: int
    tool_calls: int


@dataclass(frozen=True)
class ScenarioAggregate:
    """Aggregate metrics across repeated runs."""

    scenario: str
    runs: int
    total_ms_avg: float
    total_ms_min: float
    total_ms_max: float
    llm_ms_avg: float
    tools_ms_avg: float
    context_ms_avg: float
    session_io_ms_avg: float
    llm_calls_avg: float
    tool_calls_avg: float


_SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(name="simple/math", turns=["What is 2 + 2?"]),
    ScenarioDefinition(name="simple/summary", turns=["Summarize the benefits of unit tests."]),
    ScenarioDefinition(
        name="multi/compare",
        turns=["List two Python web frameworks.", "Compare them briefly."],
    ),
)


def load_scenarios(selector: str | None = None) -> list[ScenarioDefinition]:
    """Load bundled benchmark scenarios, optionally filtering by prefix/name."""
    if not selector:
        return list(_SCENARIOS)
    return [scenario for scenario in _SCENARIOS if scenario.name.startswith(selector)]


async def run_scenarios(
    agent_loop,
    scenarios: Iterable[ScenarioDefinition],
    *,
    repeat: int,
) -> list[ScenarioRunResult]:
    """Run scenarios repeatedly and collect profiler metrics."""
    rows: list[ScenarioRunResult] = []
    profiler = agent_loop.profiler
    previous_enabled = bool(profiler.enabled)
    profiler.enabled = True
    try:
        for scenario in scenarios:
            for run_index in range(1, repeat + 1):
                total_ms = llm_ms = tools_ms = context_ms = session_io_ms = 0.0
                llm_calls = tool_calls = 0
                for turn_index, turn in enumerate(scenario.turns, start=1):
                    await agent_loop.process_direct(
                        turn,
                        session_key=f"benchmark:{scenario.name}:{run_index}:{turn_index}",
                    )
                    report = profiler.last_report
                    if report is None:
                        continue
                    total_ms += report.total_ms
                    llm_ms += report.llm_total_ms
                    tools_ms += report.tools_total_ms
                    context_ms += report.context_build_ms
                    session_io_ms += report.session_io_ms
                    llm_calls += len(report.llm_calls)
                    tool_calls += len(report.tool_calls)
                rows.append(
                    ScenarioRunResult(
                        scenario=scenario.name,
                        run_index=run_index,
                        total_ms=total_ms,
                        llm_ms=llm_ms,
                        tools_ms=tools_ms,
                        context_ms=context_ms,
                        session_io_ms=session_io_ms,
                        llm_calls=llm_calls,
                        tool_calls=tool_calls,
                    )
                )
    finally:
        profiler.enabled = previous_enabled
    return rows


def aggregate_results(rows: Iterable[ScenarioRunResult]) -> list[ScenarioAggregate]:
    """Aggregate repeated benchmark runs by scenario name."""
    grouped: dict[str, list[ScenarioRunResult]] = {}
    for row in rows:
        grouped.setdefault(row.scenario, []).append(row)

    aggregates: list[ScenarioAggregate] = []
    for scenario, items in sorted(grouped.items()):
        aggregates.append(
            ScenarioAggregate(
                scenario=scenario,
                runs=len(items),
                total_ms_avg=mean(item.total_ms for item in items),
                total_ms_min=min(item.total_ms for item in items),
                total_ms_max=max(item.total_ms for item in items),
                llm_ms_avg=mean(item.llm_ms for item in items),
                tools_ms_avg=mean(item.tools_ms for item in items),
                context_ms_avg=mean(item.context_ms for item in items),
                session_io_ms_avg=mean(item.session_io_ms for item in items),
                llm_calls_avg=mean(item.llm_calls for item in items),
                tool_calls_avg=mean(item.tool_calls for item in items),
            )
        )
    return aggregates


def format_results_table(rows: Iterable[ScenarioAggregate]) -> str:
    """Render aggregates as a lightweight plain-text table."""
    row_list = list(rows)
    if not row_list:
        return "No benchmark results."
    header = (
        "scenario | runs | avg_total_ms | min_total_ms | max_total_ms | "
        "avg_llm_ms | avg_tools_ms"
    )
    lines = [header, "-" * len(header)]
    for row in row_list:
        lines.append(
            f"{row.scenario} | {row.runs} | {row.total_ms_avg:.1f} | "
            f"{row.total_ms_min:.1f} | {row.total_ms_max:.1f} | "
            f"{row.llm_ms_avg:.1f} | {row.tools_ms_avg:.1f}"
        )
    return "\n".join(lines)


def results_to_json(rows: Iterable[ScenarioAggregate]) -> str:
    """Serialize aggregates as JSON."""
    return json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False)
