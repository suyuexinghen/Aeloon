"""PlanPackage — canonical root aggregate for the PluginCreator workflow.

Contains all sub-models that define the full planning state: background
snapshot, programme structure, design review, phase contracts, plan items,
artifact specs, decision points, defer ledger, verification gates, resume
block, next steps, risk register, and acceptance criteria.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from .artifacts import ArtifactSpec
from .governance import (
    AcceptanceCriterion,
    DecisionPoint,
    DeferLedger,
    PlanningStatus,
    RiskItem,
    VerificationGate,
)
from .phases import PhaseContract, PlanItem
from .resume import ResumeBlock

# ---------------------------------------------------------------------------
# Supporting sub-models
# ---------------------------------------------------------------------------


class BackgroundSnapshot(BaseModel):
    """Captures the context in which the planning takes place."""

    summary: str
    sdk_constraints: list[str] = Field(default_factory=list)
    baseline_capabilities: list[str] = Field(default_factory=list)
    input_sources: list[str] = Field(default_factory=list)
    existing_examples: list[str] = Field(default_factory=list)
    workspace_context: str | None = None
    output_constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)


class StatusSummary(BaseModel):
    """Status entry for a single phase in the programme structure."""

    phase_id: str
    status: PlanningStatus
    note: str = ""


class ProgrammeStructure(BaseModel):
    """Overall programme shape: ordered phases, critical path, parallel opportunities."""

    phases: list[str]  # ordered phase_id list
    critical_path_summary: list[str] = Field(default_factory=list)
    parallel_opportunity_summary: list[str] = Field(default_factory=list)
    phase_ordering_constraints: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    status_summary: list[StatusSummary] = Field(default_factory=list)


class DesignReview(BaseModel):
    """Scoped design review capturing key decisions and open questions."""

    scope_framing: str
    goals: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    key_constraints: list[str] = Field(default_factory=list)
    design_decisions: list[str] = Field(default_factory=list)
    plugin_kind_decision: str | None = None
    capability_rationale: list[str] = Field(default_factory=list)
    naming_strategy: str | None = None
    workspace_strategy: str | None = None
    validation_strategy_preview: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    linked_decision_point_ids: list[str] = Field(default_factory=list)
    linked_risk_ids: list[str] = Field(default_factory=list)


class NextSteps(BaseModel):
    """Recommended next actions for the operator."""

    immediate: list[str] = Field(default_factory=list)
    if_blocked: list[str] = Field(default_factory=list)
    if_gate_fails: list[str] = Field(default_factory=list)
    if_review_requested: list[str] = Field(default_factory=list)
    operator_prompts: list[str] = Field(default_factory=list)
    priority_order: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root aggregate
# ---------------------------------------------------------------------------


class PlanPackage(BaseModel):
    """Canonical root aggregate for a PluginCreator planning session."""

    schema_version: str = "1.0"
    plan_package_id: str = Field(default_factory=lambda: f"pp_{uuid4().hex[:16]}")
    project_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    planning_status: PlanningStatus = PlanningStatus.PLANNED
    background_snapshot: BackgroundSnapshot
    programme_structure: ProgrammeStructure
    design_review: DesignReview
    phase_contracts: list[PhaseContract] = Field(default_factory=list)
    plan_items: list[PlanItem] = Field(default_factory=list)
    artifact_specs: list[ArtifactSpec] = Field(default_factory=list)
    decision_points: list[DecisionPoint] = Field(default_factory=list)
    defer_ledger: DeferLedger = Field(default_factory=DeferLedger)
    verification_gates: list[VerificationGate] = Field(default_factory=list)
    resume_block: ResumeBlock
    next_steps: NextSteps = Field(default_factory=NextSteps)
    risk_register: list[RiskItem] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    current_phase_id: str | None = None
    critical_path_item_ids: list[str] = Field(default_factory=list)
    parallel_opportunity_groups: list[list[str]] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)
    compatibility_mode: str | None = None
