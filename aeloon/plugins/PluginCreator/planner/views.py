"""Plan view renderers — markdown output for full and compact views.

Sprint 1 stubs: basic rendering.  Full formatting in a later sprint.
"""

from __future__ import annotations

from ..models import PlanPackage


def render_full_plan(pkg: PlanPackage) -> str:
    """Render full plan as human-readable markdown."""
    lines: list[str] = [
        f"# Plan: {pkg.project_id}",
        "",
        f"Status: {pkg.planning_status.value}",
        f"Phases: {len(pkg.phase_contracts)}",
        f"Items: {len(pkg.plan_items)}",
        "",
    ]
    for phase in pkg.phase_contracts:
        lines.append(f"## {phase.phase_name} (`{phase.phase_id}`)")
        lines.append(f"Goal: {phase.goal}")
        lines.append(f"Tasks: {', '.join(phase.task_ids) or 'none'}")
        lines.append("")
    return "\n".join(lines)


def render_compact_plan(pkg: PlanPackage) -> str:
    """Render compact resume-safe markdown from ResumeBlock + critical path."""
    rb = pkg.resume_block
    lines: list[str] = [
        f"[Resume] Phase: {rb.current_phase}",
        f"Next: {rb.next_safe_action}",
        f"Suggest: {rb.next_prompt_suggestion}",
        "",
        f"Critical path: {', '.join(pkg.critical_path_item_ids) or 'TBD'}",
    ]
    if rb.open_blockers:
        lines.append(f"Blockers: {', '.join(rb.open_blockers)}")
    return "\n".join(lines)
