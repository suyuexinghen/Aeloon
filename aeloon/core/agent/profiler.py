"""Lightweight runtime profiler for agent turns."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from types import SimpleNamespace
from typing import Any


def _parse_deep_profile_section(section: str) -> tuple[str, list[str]]:
    """Parse a stored deep-profile section into title and content lines."""
    lines = section.splitlines()
    if not lines:
        return "", []
    title = lines[0].strip()
    content = lines[2:] if len(lines) >= 2 and set(lines[1]) == {"-"} else lines[1:]
    return title, [line.rstrip() for line in content if line.strip()]


def _parse_key_value_line(line: str) -> tuple[str, str] | None:
    """Parse `Key: Value` lines used in deep-profile sections."""
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        return None
    return key, value


def _compact_stage_label(title: str) -> str:
    """Collapse `Science · <Stage>` titles to their stage name."""
    return title.split("·", 1)[-1].strip() if "·" in title else title.strip()


def _extract_execute_metrics(lines: list[str]) -> tuple[str | None, str | None]:
    """Extract execution counts from an Execute section."""
    executions = None
    failed = None
    for line in lines:
        parsed = _parse_key_value_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key == "Executions":
            executions = value
        elif key == "Failed":
            failed = value
    return executions, failed


def _extract_wave_metrics(lines: list[str]) -> tuple[str | None, str | None, str | None]:
    """Extract wave summary values from an Execute Wave section."""
    completed = None
    ready_nodes = None
    elapsed = None
    for line in lines:
        parsed = _parse_key_value_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key == "Completed so far":
            completed = value
        elif key == "Ready nodes":
            ready_nodes = value
        elif key == "Elapsed":
            elapsed = value
    return completed, ready_nodes, elapsed


def _extract_node_metric(lines: list[str], field: str) -> str | None:
    """Extract a named field from node completion/failure sections."""
    for line in lines:
        parsed = _parse_key_value_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key == field:
            return value
    return None


def _render_science_workflow_summary(sections: list[str]) -> list[str]:
    """Render science workflow sections into a compact timeline/wave summary."""
    parsed_sections = [_parse_deep_profile_section(section) for section in sections]
    science_sections = [
        (title, lines) for title, lines in parsed_sections if title.startswith("Science")
    ]
    if not science_sections:
        return []

    timeline: list[str] = []
    wave_lines: list[str] = []
    completed_nodes: list[str] = []
    failed_nodes: list[str] = []

    for title, lines in science_sections:
        stage = _compact_stage_label(title)
        if stage == "Execute":
            executions, failed = _extract_execute_metrics(lines)
            summary_parts = []
            if executions is not None:
                summary_parts.append(f"exec={executions}")
            if failed is not None:
                summary_parts.append(f"failed={failed}")
            suffix = f" [{' · '.join(summary_parts)}]" if summary_parts else ""
            timeline.append(f"- Execute{suffix}")
            continue

        if stage == "Execute Wave":
            completed, ready_nodes, elapsed = _extract_wave_metrics(lines)
            ready = ready_nodes or "(none)"
            detail_parts = []
            if completed is not None:
                detail_parts.append(f"done {completed}")
            if elapsed is not None:
                detail_parts.append(elapsed)
            details = f" ({'; '.join(detail_parts)})" if detail_parts else ""
            wave_lines.append(f"- {ready}{details}")
            continue

        if stage == "Node Completed":
            node = _extract_node_metric(lines, "Node")
            elapsed = _extract_node_metric(lines, "Elapsed")
            output_chars = _extract_node_metric(lines, "Output chars")
            if node:
                detail_parts = []
                if elapsed is not None:
                    detail_parts.append(elapsed)
                if output_chars is not None:
                    detail_parts.append(f"{output_chars} chars")
                details = f" ({', '.join(detail_parts)})" if detail_parts else ""
                completed_nodes.append(f"- {node}{details}")
            continue

        if stage == "Node Failed":
            node = _extract_node_metric(lines, "Node")
            error = _extract_node_metric(lines, "Error")
            if node:
                failed_nodes.append(f"- {node}: {error or 'unknown error'}")
            continue

        if stage == "Retry":
            node = _extract_node_metric(lines, "Node")
            attempt = _extract_node_metric(lines, "Attempt")
            backoff = _extract_node_metric(lines, "Backoff")
            retry_parts = [part for part in (attempt, backoff) if part is not None]
            retry_suffix = f" [{' · '.join(retry_parts)}]" if retry_parts else ""
            timeline.append(f"- Retry {node or '?'}{retry_suffix}")
            continue

        summary_line = next(
            (line for line in lines if _parse_key_value_line(line) is not None), None
        )
        if summary_line is None and lines:
            summary_line = lines[0]
        suffix = f" — {summary_line}" if summary_line else ""
        timeline.append(f"- {stage}{suffix}")

    output = ["Science Workflow Timeline", "-------------------------"]
    if timeline:
        output.extend(timeline)
    if wave_lines:
        output.append("")
        output.append("DAG Waves:")
        output.extend(wave_lines)
    if completed_nodes:
        output.append("")
        output.append("Node Outcomes:")
        output.extend(completed_nodes)
    if failed_nodes:
        output.append("")
        output.append("Failures:")
        output.extend(failed_nodes)
    return output


# ---------------------------------------------------------------------------
# Market workflow summary renderer
# ---------------------------------------------------------------------------

_COLLECTION_STAGES = frozenset(
    {
        "Collect Start",
        "Radar Check",
        "Live Collect",
        "Radar Snapshot",
        "Scope Filter",
        "Recent Filter",
        "Selection",
        "Collect Result",
    }
)

_EVENT_STAGES = frozenset(
    {
        "Build Events",
        "Analyze Events",
        "Organize Digest",
        "Read Article",
        "Analyze News Item",
    }
)

_FAILURE_STAGES = frozenset(
    {
        "Failure",
        "Command Failed",
    }
)


def _render_market_workflow_summary(sections: list[str]) -> list[str]:
    """Render Market workflow sections into a structured timeline.

    Groups sections into:
      - Collection Pipeline (radar, live collect, filters, selection)
      - Event Processing (build events, analyze, digest)
      - Failures
    """
    parsed_sections = [_parse_deep_profile_section(section) for section in sections]
    market_sections = [
        (title, lines) for title, lines in parsed_sections if title.startswith("Market")
    ]
    if not market_sections:
        return []

    collection_lines: list[str] = []
    event_lines: list[str] = []
    failure_lines: list[str] = []

    for title, lines in market_sections:
        stage = _compact_stage_label(title)
        # Find a short summary line for the timeline entry
        summary_line = next(
            (line for line in lines if _parse_key_value_line(line) is not None), None
        )
        if summary_line is None and lines:
            summary_line = lines[0]
        suffix = f" — {summary_line}" if summary_line else ""

        if stage in _COLLECTION_STAGES:
            collection_lines.append(f"- {stage}{suffix}")
        elif stage in _EVENT_STAGES:
            event_lines.append(f"- {stage}{suffix}")
        elif stage in _FAILURE_STAGES:
            failure_lines.append(f"- {stage}{suffix}")

    output = ["Market Workflow Timeline", "------------------------"]
    if collection_lines:
        output.append("")
        output.append("Collection Pipeline:")
        output.extend(collection_lines)
    if event_lines:
        output.append("")
        output.append("Event Processing:")
        output.extend(event_lines)
    if failure_lines:
        output.append("")
        output.append("Failures:")
        output.extend(failure_lines)
    return output


class SpanCategory(StrEnum):
    """Profiling span categories."""

    LLM = "llm"
    TOOL = "tool"
    CONTEXT = "context"
    SESSION_LOAD = "session_load"
    SESSION_SAVE = "session_save"


@dataclass
class ProfileSample:
    """A single timing sample in a profile report."""

    label: str
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileReport:
    """Structured profile report for a single user turn."""

    total_ms: float = 0.0
    llm_calls: list[ProfileSample] = field(default_factory=list)
    tool_calls: list[ProfileSample] = field(default_factory=list)
    context_build_ms: float = 0.0
    session_io_ms: float = 0.0
    session_load_ms: float = 0.0
    session_save_ms: float = 0.0
    deep_profile_sections: list[str] = field(default_factory=list)

    @property
    def llm_total_ms(self) -> float:
        """Total LLM-call latency in milliseconds."""
        return sum(item.duration_ms for item in self.llm_calls)

    @property
    def tools_total_ms(self) -> float:
        """Total tool-call latency in milliseconds."""
        return sum(item.duration_ms for item in self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """Return report as a plain dict."""
        return {
            "total_ms": self.total_ms,
            "llm_total_ms": self.llm_total_ms,
            "tools_total_ms": self.tools_total_ms,
            "context_build_ms": self.context_build_ms,
            "session_io_ms": self.session_io_ms,
            "session_load_ms": self.session_load_ms,
            "session_save_ms": self.session_save_ms,
            "deep_profile_sections": list(self.deep_profile_sections),
            "llm_calls": [
                {"label": s.label, "duration_ms": s.duration_ms, "meta": s.meta}
                for s in self.llm_calls
            ],
            "tool_calls": [
                {"label": s.label, "duration_ms": s.duration_ms, "meta": s.meta}
                for s in self.tool_calls
            ],
        }


class AgentProfiler:
    """Collect timing samples for each agent turn."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._turn_started_at: float | None = None
        self._current_report: ProfileReport | None = None
        self._last_report: ProfileReport | None = None
        self._saved_sections: list[str] | None = None
        self._nesting_depth: int = 0

    @property
    def last_report(self) -> ProfileReport | None:
        """Most recent completed report."""
        return self._last_report

    def start_turn(self) -> None:
        """Start collecting samples for a new turn.

        Supports nesting: if a turn is already active, saves its
        deep-profile sections before starting a fresh sub-report.
        """
        if not self.enabled:
            return
        # Preserve sections from the outer turn so they aren't lost
        # when a nested process_direct() call triggers process_turn().
        if self._current_report is not None and self._current_report.deep_profile_sections:
            if self._saved_sections is None:
                self._saved_sections = []
            self._saved_sections.extend(self._current_report.deep_profile_sections)
        self._turn_started_at = perf_counter()
        self._current_report = ProfileReport()

    def record(
        self,
        category: SpanCategory,
        label: str,
        duration_ms: float,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Record one sample."""
        if not self.enabled or self._current_report is None:
            return

        sample = ProfileSample(label=label, duration_ms=duration_ms, meta=meta or {})
        match category:
            case SpanCategory.LLM:
                self._current_report.llm_calls.append(sample)
            case SpanCategory.TOOL:
                self._current_report.tool_calls.append(sample)
            case SpanCategory.CONTEXT:
                self._current_report.context_build_ms += duration_ms
            case SpanCategory.SESSION_LOAD:
                self._current_report.session_io_ms += duration_ms
                self._current_report.session_load_ms += duration_ms
            case SpanCategory.SESSION_SAVE:
                self._current_report.session_io_ms += duration_ms
                self._current_report.session_save_ms += duration_ms

    def end_turn(self) -> ProfileReport:
        """Finalize and return current report.

        When nested, merges the sub-report into the saved sections
        so that the outer turn's final report includes everything.
        Sections from previous nested turns (``_last_report``) are also
        carried forward so nothing is lost across multiple nested calls.
        """
        if not self.enabled or self._current_report is None:
            # Outer turn: only prepend new saved sections (added between inner
            # turns).  _last_report already contains sections from all inner
            # turns, so re-adding them would cause duplication.
            if self._saved_sections and self._last_report is not None:
                self._last_report.deep_profile_sections = (
                    self._saved_sections + self._last_report.deep_profile_sections
                )
                self._saved_sections = None
            return self._last_report or ProfileReport()

        # Inner turn: inherit both saved sections AND previous last report
        inherited: list[str] = []
        if self._saved_sections:
            inherited.extend(self._saved_sections)
            self._saved_sections = None
        if self._last_report and self._last_report.deep_profile_sections:
            inherited.extend(self._last_report.deep_profile_sections)

        if self._turn_started_at is not None:
            self._current_report.total_ms = (perf_counter() - self._turn_started_at) * 1000

        # Prepend inherited sections (from outer turns + previous nested turns)
        if inherited:
            self._current_report.deep_profile_sections = (
                inherited + self._current_report.deep_profile_sections
            )

        self._last_report = self._current_report
        self._current_report = None
        self._turn_started_at = None
        return self._last_report

    def add_deep_profile_section(self, title: str, lines: list[str]) -> None:
        """Append a structured deep-profile section to the current report.

        When called between nested turns (``_current_report`` is ``None``),
        stashes the section into ``_saved_sections`` so it is merged by the
        next ``end_turn()``.  This prevents sections added by orchestrators
        after a ``process_direct()`` call from being silently dropped.
        """
        if not self.enabled:
            return
        section_lines = [title, "-" * len(title), *lines]
        section = "\n".join(section_lines)
        if self._current_report is not None:
            self._current_report.deep_profile_sections.append(section)
        else:
            # Between nested turns — save for later restoration
            if self._saved_sections is None:
                self._saved_sections = []
            self._saved_sections.append(section)

    @asynccontextmanager
    async def span(self, category: SpanCategory, label: str):
        """Async context manager for recording one timed span."""
        if not self.enabled or self._current_report is None:
            yield SimpleNamespace(meta={})
            return

        t0 = perf_counter()
        ctx = SimpleNamespace(meta={})
        try:
            yield ctx
        finally:
            self.record(category, label, (perf_counter() - t0) * 1000, ctx.meta)

    def report(self) -> str:
        """Render the most recent report to readable text."""
        report = self._last_report
        if report is None:
            return "No profiling report available."

        lines = [
            "Profile Report",
            "==============",
            f"Total: {report.total_ms:.1f} ms",
            f"LLM: {report.llm_total_ms:.1f} ms across {len(report.llm_calls)} call(s)",
            f"Tools: {report.tools_total_ms:.1f} ms across {len(report.tool_calls)} call(s)",
            f"Context Build: {report.context_build_ms:.1f} ms",
            (
                "Session I/O: "
                f"{report.session_io_ms:.1f} ms "
                f"(load {report.session_load_ms:.1f} ms, save {report.session_save_ms:.1f} ms)"
            ),
        ]

        if report.llm_calls:
            lines.append("")
            lines.append("LLM Calls:")
            for idx, call in enumerate(report.llm_calls, start=1):
                usage = call.meta.get("usage") if isinstance(call.meta, dict) else None
                if isinstance(usage, dict) and usage:
                    p = int(usage.get("prompt_tokens", 0))
                    c = int(usage.get("completion_tokens", 0))
                    t = int(usage.get("total_tokens", 0))
                    token_info = f" (tokens p/c/t={p}/{c}/{t})"
                else:
                    token_info = ""
                lines.append(f"{idx}. {call.label}: {call.duration_ms:.1f} ms{token_info}")

        if report.tool_calls:
            lines.append("")
            lines.append("Tool Calls:")
            for idx, call in enumerate(report.tool_calls, start=1):
                args_summary = (
                    call.meta.get("args_summary") if isinstance(call.meta, dict) else None
                )
                suffix = f" ({args_summary})" if args_summary else ""
                lines.append(f"{idx}. {call.label}: {call.duration_ms:.1f} ms{suffix}")

        return "\n".join(lines)

    def report_top_heavy(self, threshold: float = 0.8) -> str:
        """Render the heaviest items whose cumulative duration reaches the threshold."""
        report = self._last_report
        if report is None:
            return "No profiling report available."

        items: list[tuple[str, float]] = []
        items.extend((f"LLM: {call.label}", call.duration_ms) for call in report.llm_calls)
        items.extend((f"Tool: {call.label}", call.duration_ms) for call in report.tool_calls)
        if report.context_build_ms > 0:
            items.append(("Context Build", report.context_build_ms))
        if report.session_load_ms > 0:
            items.append(("Session Load", report.session_load_ms))
        if report.session_save_ms > 0:
            items.append(("Session Save", report.session_save_ms))

        if not items:
            return "No profiling report available."

        total = sum(duration for _, duration in items)
        if total <= 0:
            return "No profiling report available."

        sorted_items = sorted(items, key=lambda item: item[1], reverse=True)
        cutoff = total * threshold
        running = 0.0
        selected: list[tuple[str, float]] = []
        for label, duration in sorted_items:
            selected.append((label, duration))
            running += duration
            if running >= cutoff:
                break

        lines = [
            "Profile Hotspots",
            "================",
            f"Coverage: {running:.1f} / {total:.1f} ms ({(running / total) * 100:.0f}%)",
        ]
        for idx, (label, duration) in enumerate(selected, start=1):
            lines.append(f"{idx}. {label}: {duration:.1f} ms")
        return "\n".join(lines)

    def report_deep_profile(self, threshold: float = 0.8) -> str:
        """Render all LLM/tools plus heavy remaining items covering threshold of the remainder."""
        report = self._last_report
        if report is None:
            return "No profiling report available."

        lines = [
            "Deep Profile",
            "============",
            f"Total: {report.total_ms:.1f} ms",
            f"LLM Total: {report.llm_total_ms:.1f} ms",
            f"Tools Total: {report.tools_total_ms:.1f} ms",
        ]

        if report.llm_calls:
            lines.append("")
            lines.append("LLM Calls:")
            for idx, call in enumerate(report.llm_calls, start=1):
                lines.append(f"{idx}. {call.label}: {call.duration_ms:.1f} ms")

        if report.tool_calls:
            lines.append("")
            lines.append("Tool Calls:")
            for idx, call in enumerate(report.tool_calls, start=1):
                args_summary = (
                    call.meta.get("args_summary") if isinstance(call.meta, dict) else None
                )
                suffix = f" ({args_summary})" if args_summary else ""
                lines.append(f"{idx}. {call.label}: {call.duration_ms:.1f} ms{suffix}")

        remaining_items: list[tuple[str, float]] = []
        if report.context_build_ms > 0:
            remaining_items.append(("Context Build", report.context_build_ms))
        if report.session_load_ms > 0:
            remaining_items.append(("Session Load", report.session_load_ms))
        if report.session_save_ms > 0:
            remaining_items.append(("Session Save", report.session_save_ms))

        if remaining_items:
            remaining_total = sum(duration for _, duration in remaining_items)
            sorted_items = sorted(remaining_items, key=lambda item: item[1], reverse=True)
            cutoff = remaining_total * threshold
            running = 0.0
            selected: list[tuple[str, float]] = []
            for label, duration in sorted_items:
                selected.append((label, duration))
                running += duration
                if running >= cutoff:
                    break

            lines.append("")
            lines.append("Other Hotspots:")
            lines.append(
                f"Coverage: {running:.1f} / {remaining_total:.1f} ms ({(running / remaining_total) * 100:.0f}%)"
            )
            for idx, (label, duration) in enumerate(selected, start=1):
                lines.append(f"{idx}. {label}: {duration:.1f} ms")

        if report.deep_profile_sections:
            lines.append("")

            # Tier 1: Science workflow summary
            science_summary = _render_science_workflow_summary(report.deep_profile_sections)
            if science_summary:
                lines.extend(science_summary)

            # Tier 2: Market workflow summary
            market_summary = _render_market_workflow_summary(report.deep_profile_sections)
            if market_summary:
                lines.extend(market_summary)

            # Tier 3: Remaining sections as raw dump
            other_sections = [
                section
                for section in report.deep_profile_sections
                if not _parse_deep_profile_section(section)[0].startswith(("Science", "Market"))
            ]
            if other_sections:
                lines.append("")
                lines.append("Workflow Stages:")
                for section in other_sections:
                    lines.append("")
                    lines.append(section)

        return "\n".join(lines)
