"""Tests for agent profiling module."""

import pytest

from aeloon.core.agent.profiler import AgentProfiler, SpanCategory


def test_profiler_start_record_end_turn_flow() -> None:
    profiler = AgentProfiler(enabled=True)

    profiler.start_turn()
    profiler.record(
        SpanCategory.LLM,
        "gpt-test",
        120.5,
        meta={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )
    profiler.record(
        SpanCategory.TOOL,
        "read_file",
        22.0,
        meta={"args_summary": "path=/tmp/test.txt"},
    )
    profiler.record(SpanCategory.CONTEXT, "build", 11.0)
    profiler.record(SpanCategory.SESSION_LOAD, "load", 3.0)
    profiler.record(SpanCategory.SESSION_SAVE, "save", 4.0)
    report = profiler.end_turn()

    assert report.total_ms >= 0
    assert len(report.llm_calls) == 1
    assert len(report.tool_calls) == 1
    assert report.context_build_ms == 11.0
    assert report.session_io_ms == 7.0
    assert report.session_load_ms == 3.0
    assert report.session_save_ms == 4.0
    assert profiler.last_report is report


def test_profiler_report_format_contains_sections() -> None:
    profiler = AgentProfiler(enabled=True)
    profiler.start_turn()
    profiler.record(
        SpanCategory.LLM,
        "gpt-test",
        8.0,
        meta={"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
    )
    profiler.record(SpanCategory.TOOL, "list_dir", 4.0, meta={"args_summary": "path=/tmp"})
    profiler.record(SpanCategory.CONTEXT, "build", 2.0)
    profiler.record(SpanCategory.SESSION_LOAD, "load", 1.0)
    profiler.end_turn()

    text = profiler.report()
    assert "Profile Report" in text
    assert "LLM Calls:" in text
    assert "Tool Calls:" in text
    assert "tokens p/c/t=1/2/3" in text


def test_profiler_disabled_collects_nothing() -> None:
    profiler = AgentProfiler(enabled=False)

    profiler.start_turn()
    profiler.record(SpanCategory.LLM, "gpt-test", 99.0)
    report = profiler.end_turn()

    assert report.llm_calls == []
    assert report.tool_calls == []
    assert profiler.last_report is None
    assert profiler.report() == "No profiling report available."


def test_report_deep_profile_renders_science_timeline_summary() -> None:
    profiler = AgentProfiler(enabled=True)
    profiler.start_turn()
    profiler.record(SpanCategory.LLM, "gpt-test", 8.0)
    profiler.add_deep_profile_section(
        "Science · Interpret",
        ["Task ID: task-1", "Goal: Study catalysts"],
    )
    profiler.add_deep_profile_section(
        "Science · Plan",
        ["Nodes: 3"],
    )
    profiler.add_deep_profile_section(
        "Science · Execute Wave",
        [
            "Completed so far: 1/3",
            "Ready nodes: search_lit, collect_data",
            "Elapsed: 2.50s / 120s",
        ],
    )
    profiler.add_deep_profile_section(
        "Science · Node Completed",
        ["Node: search_lit", "Elapsed: 1.20s", "Output chars: 540"],
    )
    profiler.add_deep_profile_section(
        "Science · Validate",
        ["Status: passed", "Next action: deliver"],
    )
    profiler.end_turn()

    text = profiler.report_deep_profile()
    assert "Science Workflow Timeline" in text
    assert "- Interpret — Task ID: task-1" in text
    assert "- Plan — Nodes: 3" in text
    assert "DAG Waves:" in text
    assert "- search_lit, collect_data (done 1/3; 2.50s / 120s)" in text
    assert "Node Outcomes:" in text
    assert "- search_lit (1.20s, 540 chars)" in text
    assert "Workflow Stages:" not in text


@pytest.mark.asyncio
async def test_profiler_span_records_duration() -> None:
    profiler = AgentProfiler(enabled=True)
    profiler.start_turn()

    async with profiler.span(SpanCategory.LLM, "gpt-test") as ctx:
        ctx.meta["usage"] = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}

    report = profiler.end_turn()
    assert len(report.llm_calls) == 1
    assert report.llm_calls[0].duration_ms >= 0
    assert report.llm_calls[0].meta["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_profiler_span_records_on_exception() -> None:
    profiler = AgentProfiler(enabled=True)
    profiler.start_turn()

    with pytest.raises(RuntimeError, match="boom"):
        async with profiler.span(SpanCategory.TOOL, "read_file"):
            raise RuntimeError("boom")

    report = profiler.end_turn()
    assert len(report.tool_calls) == 1
    assert report.tool_calls[0].label == "read_file"


@pytest.mark.asyncio
async def test_profiler_span_noop_when_disabled() -> None:
    profiler = AgentProfiler(enabled=False)
    profiler.start_turn()

    async with profiler.span(SpanCategory.CONTEXT, "build") as ctx:
        ctx.meta["k"] = "v"

    report = profiler.end_turn()
    assert report.context_build_ms == 0
    assert report.llm_calls == []
    assert profiler.last_report is None


def test_nested_turn_preserves_sections_between_inner_turns() -> None:
    """Sections added between nested turns (e.g. orchestrator after process_direct)
    must be preserved and appear in the final report."""
    profiler = AgentProfiler(enabled=True)

    # Outer turn
    profiler.start_turn()
    profiler.add_deep_profile_section("Outer · Start", ["Phase: init"])

    # First nested turn (simulates process_direct for node 1)
    profiler.start_turn()
    profiler.record(SpanCategory.LLM, "node-1-llm", 10.0)
    profiler.end_turn()

    # Section added AFTER nested turn (like orchestrator adds SE sections)
    profiler.add_deep_profile_section("SE · Design", ["Node: design_1", "Elapsed: 1.2s"])

    # Second nested turn (simulates process_direct for node 2)
    profiler.start_turn()
    profiler.record(SpanCategory.LLM, "node-2-llm", 20.0)
    profiler.end_turn()

    # Another section between turns
    profiler.add_deep_profile_section("SE · Code", ["Node: code_1", "Elapsed: 2.5s"])

    # Outer turn ends
    report = profiler.end_turn()

    # All sections should be present
    titles = [line.split("\n")[0] for line in report.deep_profile_sections]
    assert "Outer · Start" in titles
    assert "SE · Design" in titles
    assert "SE · Code" in titles


def test_market_sections_render_in_deep_profile() -> None:
    """Market sections should render under Market Workflow Timeline."""
    profiler = AgentProfiler(enabled=True)
    profiler.start_turn()
    profiler.record(SpanCategory.LLM, "llm-call", 5.0)
    profiler.add_deep_profile_section(
        "Market · Collect Start",
        ["Scope: all", "New only: True", "Limit: 0"],
    )
    profiler.add_deep_profile_section(
        "Market · Radar Check",
        ["Hit: false", "Reason: no active radar snapshot"],
    )
    profiler.add_deep_profile_section(
        "Market · Build Events",
        ["Signal count: 10", "Event count: 3"],
    )
    profiler.end_turn()

    text = profiler.report_deep_profile()
    assert "Deep Profile" in text
    # Structured timeline renders stage names
    assert "Market Workflow Timeline" in text
    assert "Collection Pipeline:" in text
    assert "- Collect Start" in text
    assert "- Radar Check" in text
    assert "Event Processing:" in text
    assert "- Build Events" in text
    # Summary lines preserved
    assert "Scope: all" in text
    assert "Hit: false" in text


def test_nested_science_turns_render_in_report_deep_profile() -> None:
    """Simulates science agent: outer turn + multiple nested inner turns.

    Verifies report_deep_profile() includes all sections from every nesting
    level (outer, between inner turns, and inner turns themselves).
    """
    profiler = AgentProfiler(enabled=True)

    # Outer turn (plugin command wrapper)
    profiler.start_turn()
    profiler.add_deep_profile_section("Science · Interpret", ["Goal: test"])
    profiler.add_deep_profile_section("Science · Plan", ["Nodes: 2"])

    # Inner turn 1 (process_direct for node 1)
    profiler.start_turn()
    profiler.record(SpanCategory.LLM, "node-1-llm", 100.0)
    profiler.add_deep_profile_section("Science · Execute Step", ["Step: 1/2"])
    profiler.end_turn()

    # Between inner turns — orchestrator sections
    profiler.add_deep_profile_section("Science · Node Completed", ["Node: search"])

    # Inner turn 2 (process_direct for node 2)
    profiler.start_turn()
    profiler.record(SpanCategory.LLM, "node-2-llm", 200.0)
    profiler.add_deep_profile_section("Science · Execute Step", ["Step: 2/2"])
    profiler.end_turn()

    # Final orchestrator section
    profiler.add_deep_profile_section("Science · Finalize", ["Status: ok"])

    # Outer turn ends
    profiler.end_turn()

    # last_report must be populated
    assert profiler.last_report is not None

    # report_deep_profile must include all sections
    text = profiler.report_deep_profile()
    assert "Deep Profile" in text
    assert "Science Workflow Timeline" in text
    # All section stages from outer and inner turns
    assert "Interpret" in text
    assert "Plan" in text
    assert "Execute Step" in text
    assert "Finalize" in text
    # "Node Completed" sections render under "Node Outcomes"
    assert "Node Outcomes" in text
    assert "search" in text
