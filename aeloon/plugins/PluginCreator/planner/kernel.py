"""PlanningKernel — assembles a PlanPackage from raw requirements.

Sprint 1 stub: builds a minimal valid PlanPackage skeleton.  Full LLM-driven
planning will be added in a later sprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..models import (
    BackgroundSnapshot,
    DesignReview,
    PhaseContract,
    PlanItem,
    PlanItemKind,
    PlanningStatus,
    PlanPackage,
    ProgrammeStructure,
    ResumeBlock,
)
from ..validator.plan_package import ValidationError, validate_plan_package
from .views import render_compact_plan, render_full_plan

if TYPE_CHECKING:
    from aeloon.plugins._sdk.runtime import PluginRuntime


@dataclass
class PlanningKernelInput:
    """Input to the planning kernel."""

    project_id: str
    raw_requirement: str
    diagram_inputs: list[str] = field(default_factory=list)
    user_constraints: dict[str, Any] = field(default_factory=dict)
    maturity: Literal["prototype", "mvp", "production_ready"] = "mvp"
    compatibility_mode: str | None = None


@dataclass
class PlanningKernelOutput:
    """Output from the planning kernel."""

    plan_package: PlanPackage
    validation_errors: list[ValidationError]
    full_view: str
    compact_view: str


class PlanningKernel:
    """Assembles a PlanPackage from a PlanningKernelInput.

    Responsibilities (full implementation in later sprints):
    1. Scope framing
    2. Design review synthesis
    3. Phase decomposition
    4. Execution item graph construction
    5. Artifact specification
    6. Verification planning
    7. Defer / resume planning
    8. Critical-path and parallel-opportunity extraction
    """

    def __init__(self, runtime: PluginRuntime) -> None:
        self._runtime = runtime

    async def plan(self, inp: PlanningKernelInput) -> PlanningKernelOutput:
        """Main entry point.  Returns PlanPackage + validation results + rendered views."""
        pkg = self._build_skeleton(inp)
        errors = validate_plan_package(pkg)
        full_view = render_full_plan(pkg)
        compact_view = render_compact_plan(pkg)
        return PlanningKernelOutput(
            plan_package=pkg,
            validation_errors=errors,
            full_view=full_view,
            compact_view=compact_view,
        )

    def _build_skeleton(self, inp: PlanningKernelInput) -> PlanPackage:
        """Stub: build minimal valid package from input."""
        phase_id = "phase_1_analysis"
        item_id = "item_1_scope"

        return PlanPackage(
            project_id=inp.project_id,
            planning_status=PlanningStatus.PLANNED,
            background_snapshot=BackgroundSnapshot(summary=inp.raw_requirement),
            programme_structure=ProgrammeStructure(phases=[phase_id]),
            design_review=DesignReview(scope_framing=inp.raw_requirement),
            phase_contracts=[
                PhaseContract(
                    phase_id=phase_id,
                    phase_name="Analysis",
                    goal="Understand requirements and scope",
                    task_ids=[item_id],
                ),
            ],
            plan_items=[
                PlanItem(
                    id=item_id,
                    title="Scope analysis",
                    kind=PlanItemKind.ANALYSIS,
                    phase_id=phase_id,
                ),
            ],
            resume_block=ResumeBlock(
                current_phase=phase_id,
                next_safe_action="Run scope analysis",
                next_prompt_suggestion="Describe the plugin you want to create",
            ),
        )
