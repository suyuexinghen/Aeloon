"""Governance models — decision points, defer ledger, verification gates, risks.

Three separate status domains:
  1. PlanningStatus — work-item / phase status (planned, partial, blocked, validated, deferred)
  2. GateStatus — gate outcome (planned, passed, failed, waived, blocked)
  3. ArtifactStatus — artifact lifecycle (in artifacts.py)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .artifacts import EvidenceRef

# ---------------------------------------------------------------------------
# Domain 1: planning / work-item status
# ---------------------------------------------------------------------------


class PlanningStatus(str, Enum):
    """Status of a planning item, phase, or work unit."""

    PLANNED = "planned"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    VALIDATED = "validated"
    DEFERRED = "deferred"


# ---------------------------------------------------------------------------
# Domain 2: gate outcome status
# ---------------------------------------------------------------------------


class GateStatus(str, Enum):
    """Outcome status of a verification gate."""

    PLANNED = "planned"
    PASSED = "passed"
    FAILED = "failed"
    WAIVED = "waived"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Gate type
# ---------------------------------------------------------------------------


class GateType(str, Enum):
    """Classification of verification gates."""

    DESIGN = "design"
    PLAN = "plan"
    BUILD = "build"
    STRUCTURAL_VALIDATION = "structural_validation"
    EXTENDED_VALIDATION = "extended_validation"
    DELIVERY = "delivery"


# ---------------------------------------------------------------------------
# Decision points
# ---------------------------------------------------------------------------


class DecisionPoint(BaseModel):
    """An explicit decision that may affect the workflow."""

    id: str = Field(default_factory=lambda: f"dp_{uuid4().hex[:12]}")
    question: str
    options: list[str] = Field(default_factory=list)
    recommended_default: str | None = None
    status: Literal["open", "resolved", "deferred"] = "open"
    impact_if_wrong: str = ""
    must_resolve_before_phase: str | None = None
    resolved_option: str | None = None
    resolved_at: str | None = None
    linked_phase_ids: list[str] = Field(default_factory=list)
    linked_artifact_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None


# ---------------------------------------------------------------------------
# Defer ledger
# ---------------------------------------------------------------------------


class DeferItem(BaseModel):
    """A work item deferred to a later phase."""

    id: str = Field(default_factory=lambda: f"def_{uuid4().hex[:12]}")
    title: str
    reason_deferred: str
    target_phase: str
    reentry_condition: str
    status: Literal["deferred", "reentered", "dropped"] = "deferred"
    priority: str | None = None
    risk_if_ignored: str | None = None
    origin_phase_id: str | None = None
    origin_plan_item_id: str | None = None
    linked_artifact_ids: list[str] = Field(default_factory=list)
    linked_decision_point_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class DeferLedger(BaseModel):
    """Collection of deferred work items."""

    items: list[DeferItem] = Field(default_factory=list)
    summary: str | None = None

    @property
    def active_item_ids(self) -> list[str]:
        """Return IDs of items still in deferred status."""
        return [i.id for i in self.items if i.status == "deferred"]


# ---------------------------------------------------------------------------
# Verification gates
# ---------------------------------------------------------------------------


class VerificationGate(BaseModel):
    """A phase-boundary validation checkpoint."""

    id: str = Field(default_factory=lambda: f"gate_{uuid4().hex[:12]}")
    name: str
    phase_id: str
    gate_type: GateType
    required: bool = True
    criteria: list[str] = Field(default_factory=list)
    status: GateStatus = GateStatus.PLANNED
    validator_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    failure_actions: list[str] = Field(default_factory=list)
    blocking_reason: str | None = None
    waiver_reason: str | None = None
    depends_on_gate_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk and acceptance
# ---------------------------------------------------------------------------


class RiskItem(BaseModel):
    """A risk entry in the project risk register."""

    id: str = Field(default_factory=lambda: f"risk_{uuid4().hex[:12]}")
    description: str
    impact: Literal["low", "medium", "high"] = "medium"
    likelihood: Literal["low", "medium", "high"] = "medium"
    mitigation: str = ""
    affected_phases: list[str] = Field(default_factory=list)


class AcceptanceCriterion(BaseModel):
    """An acceptance criterion for the project."""

    id: str = Field(default_factory=lambda: f"ac_{uuid4().hex[:12]}")
    description: str
    phase_id: str | None = None
    done_when: list[str] = Field(default_factory=list)
