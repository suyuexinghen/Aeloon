"""Cross-model invariant validator for :class:`PlanPackage`.

Enforces 10 invariants covering ID uniqueness, reference integrity, DAG
acyclicity, phase/task consistency, gate rules, decision-point requirements,
defer-item completeness, and resume-block validity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from ..models import (
    PlanItem,
    PlanPackage,
)


@dataclass
class ValidationError:
    """A single validation finding."""

    code: str
    message: str
    severity: Literal["error", "warning"]
    context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_plan_package(pkg: PlanPackage) -> list[ValidationError]:
    """Run all invariant checks.  Returns list of errors (empty = valid)."""
    errors: list[ValidationError] = []
    errors.extend(_check_global_id_uniqueness(pkg))
    errors.extend(_check_all_references_resolvable(pkg))
    errors.extend(_check_plan_item_dag(pkg))
    errors.extend(_check_phase_task_consistency(pkg))
    errors.extend(_check_gate_phase_consistency(pkg))
    errors.extend(_check_gate_evidence_rule(pkg))
    errors.extend(_check_decision_point_invariants(pkg))
    errors.extend(_check_defer_item_invariants(pkg))
    errors.extend(_check_resume_block_references(pkg))
    errors.extend(_check_conservative_status(pkg))
    return errors


def validate_plan_item_dag(items: list[PlanItem]) -> list[ValidationError]:
    """Standalone DAG check (cycle detection via Kahn's algorithm)."""
    return _check_plan_item_dag_items(items)


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------


def _collect_all_ids(pkg: PlanPackage) -> list[tuple[str, str]]:
    """Return [(id, source_label)] for every ID in the package (may have duplicates)."""
    pairs: list[tuple[str, str]] = []
    for phase in pkg.phase_contracts:
        pairs.append((phase.phase_id, "PhaseContract.phase_id"))
    for item in pkg.plan_items:
        pairs.append((item.id, "PlanItem.id"))
    for art in pkg.artifact_specs:
        pairs.append((art.id, "ArtifactSpec.id"))
    for dp in pkg.decision_points:
        pairs.append((dp.id, "DecisionPoint.id"))
    for gate in pkg.verification_gates:
        pairs.append((gate.id, "VerificationGate.id"))
    for risk in pkg.risk_register:
        pairs.append((risk.id, "RiskItem.id"))
    for ac in pkg.acceptance_criteria:
        pairs.append((ac.id, "AcceptanceCriterion.id"))
    for di in pkg.defer_ledger.items:
        pairs.append((di.id, "DeferItem.id"))
    return pairs


def _check_global_id_uniqueness(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 1: all IDs globally unique within package."""
    errors: list[ValidationError] = []
    seen: dict[str, str] = {}
    for oid, source in _collect_all_ids(pkg):
        if oid in seen:
            errors.append(
                ValidationError(
                    code="duplicate_id",
                    message=f"Duplicate ID '{oid}' found in {source} and {seen[oid]}",
                    severity="error",
                    context={"id": oid, "sources": [source, seen[oid]]},
                )
            )
        else:
            seen[oid] = source
    return errors


def _build_id_set(pkg: PlanPackage) -> set[str]:
    """Return the set of all valid IDs in the package."""
    return {oid for oid, _ in _collect_all_ids(pkg)}


def _check_all_references_resolvable(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 2: all reference fields point to existing IDs."""
    errors: list[ValidationError] = []
    valid = _build_id_set(pkg)

    def _check_ref(ref_id: str, source: str) -> None:
        if ref_id not in valid:
            errors.append(
                ValidationError(
                    code="unresolvable_reference",
                    message=f"{source} references unknown ID '{ref_id}'",
                    severity="error",
                    context={"ref_id": ref_id, "source": source},
                )
            )

    for phase in pkg.phase_contracts:
        for tid in phase.task_ids:
            _check_ref(tid, f"PhaseContract({phase.phase_id}).task_ids")
        for gid in phase.verification_gate_ids:
            _check_ref(gid, f"PhaseContract({phase.phase_id}).verification_gate_ids")
        for did in phase.decision_point_ids:
            _check_ref(did, f"PhaseContract({phase.phase_id}).decision_point_ids")
        for aid in phase.artifact_ids:
            _check_ref(aid, f"PhaseContract({phase.phase_id}).artifact_ids")

    for item in pkg.plan_items:
        for dep in item.depends_on:
            _check_ref(dep, f"PlanItem({item.id}).depends_on")

    for gate in pkg.verification_gates:
        for gid in gate.depends_on_gate_ids:
            _check_ref(gid, f"VerificationGate({gate.id}).depends_on_gate_ids")

    return errors


def _check_plan_item_dag(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 3: PlanItem.depends_on forms a DAG (no cycles)."""
    return _check_plan_item_dag_items(pkg.plan_items)


def _check_plan_item_dag_items(items: list[PlanItem]) -> list[ValidationError]:
    """Kahn's algorithm for cycle detection on PlanItem graph."""
    errors: list[ValidationError] = []
    if not items:
        return errors

    in_degree: dict[str, int] = {item.id: 0 for item in items}

    adj: dict[str, list[str]] = {item.id: [] for item in items}
    for item in items:
        for dep_id in item.depends_on:
            if dep_id in adj:
                adj[dep_id].append(item.id)
                in_degree[item.id] += 1

    queue = deque(iid for iid, deg in in_degree.items() if deg == 0)
    visited = 0

    while queue:
        iid = queue.popleft()
        visited += 1
        for neighbour in adj[iid]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if visited != len(items):
        remaining = [iid for iid, deg in in_degree.items() if deg > 0]
        errors.append(
            ValidationError(
                code="dag_cycle",
                message=f"Plan item dependency cycle detected among: {remaining}",
                severity="error",
                context={"cycle_items": remaining},
            )
        )

    return errors


def _check_phase_task_consistency(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 4: PhaseContract.task_ids <-> PlanItem.phase_id bidirectional consistency."""
    errors: list[ValidationError] = []

    # Build phase -> claimed task IDs
    phase_tasks: dict[str, set[str]] = {
        phase.phase_id: set(phase.task_ids) for phase in pkg.phase_contracts
    }
    # Build task -> actual phase_id
    item_phases: dict[str, str | None] = {item.id: item.phase_id for item in pkg.plan_items}

    for phase in pkg.phase_contracts:
        for tid in phase.task_ids:
            actual = item_phases.get(tid)
            if actual is not None and actual != phase.phase_id:
                errors.append(
                    ValidationError(
                        code="phase_task_mismatch",
                        message=(
                            f"PlanItem '{tid}' has phase_id='{actual}' but "
                            f"PhaseContract '{phase.phase_id}' claims it in task_ids"
                        ),
                        severity="error",
                        context={
                            "task_id": tid,
                            "expected_phase": phase.phase_id,
                            "actual_phase": actual,
                        },
                    )
                )

    for item in pkg.plan_items:
        if item.phase_id is not None and item.phase_id in phase_tasks:
            if item.id not in phase_tasks[item.phase_id]:
                errors.append(
                    ValidationError(
                        code="phase_task_mismatch",
                        message=(
                            f"PlanItem '{item.id}' claims phase_id='{item.phase_id}' "
                            f"but is not in that phase's task_ids"
                        ),
                        severity="error",
                        context={"task_id": item.id, "phase_id": item.phase_id},
                    )
                )

    return errors


def _check_gate_phase_consistency(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 5: required gate failed/blocked -> phase must not be validated."""
    errors: list[ValidationError] = []
    phase_status = {phase.phase_id: phase.status for phase in pkg.phase_contracts}

    for gate in pkg.verification_gates:
        if gate.required and gate.status in ("failed", "blocked"):
            pstatus = phase_status.get(gate.phase_id)
            if pstatus is not None and pstatus.value == "validated":
                errors.append(
                    ValidationError(
                        code="gate_phase_conflict",
                        message=(
                            f"Required gate '{gate.id}' is {gate.status.value} but "
                            f"phase '{gate.phase_id}' is validated"
                        ),
                        severity="error",
                        context={
                            "gate_id": gate.id,
                            "phase_id": gate.phase_id,
                            "gate_status": gate.status.value,
                        },
                    )
                )

    return errors


def _check_gate_evidence_rule(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 6: passed gate requires evidence (unless waived)."""
    errors: list[ValidationError] = []

    for gate in pkg.verification_gates:
        if gate.status.value == "passed" and not gate.evidence_refs and not gate.waiver_reason:
            errors.append(
                ValidationError(
                    code="gate_no_evidence",
                    message=f"Gate '{gate.id}' is passed but has no evidence_refs and no waiver_reason",
                    severity="error",
                    context={"gate_id": gate.id},
                )
            )

    return errors


def _check_decision_point_invariants(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 7: resolved decision point must have resolved_option."""
    errors: list[ValidationError] = []

    for dp in pkg.decision_points:
        if dp.status == "resolved" and not dp.resolved_option:
            errors.append(
                ValidationError(
                    code="decision_point_no_resolution",
                    message=f"DecisionPoint '{dp.id}' is resolved but has no resolved_option",
                    severity="error",
                    context={"decision_point_id": dp.id},
                )
            )

    return errors


def _check_defer_item_invariants(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 8: each DeferItem must have reason_deferred, target_phase, reentry_condition."""
    errors: list[ValidationError] = []

    for di in pkg.defer_ledger.items:
        if not di.reason_deferred:
            errors.append(
                ValidationError(
                    code="defer_item_incomplete",
                    message=f"DeferItem '{di.id}' missing reason_deferred",
                    severity="error",
                    context={"defer_item_id": di.id, "field": "reason_deferred"},
                )
            )
        if not di.target_phase:
            errors.append(
                ValidationError(
                    code="defer_item_incomplete",
                    message=f"DeferItem '{di.id}' missing target_phase",
                    severity="error",
                    context={"defer_item_id": di.id, "field": "target_phase"},
                )
            )
        if not di.reentry_condition:
            errors.append(
                ValidationError(
                    code="defer_item_incomplete",
                    message=f"DeferItem '{di.id}' missing reentry_condition",
                    severity="error",
                    context={"defer_item_id": di.id, "field": "reentry_condition"},
                )
            )

    return errors


def _check_resume_block_references(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 9: ResumeBlock.current_phase must reference an existing phase."""
    errors: list[ValidationError] = []
    phase_ids = {phase.phase_id for phase in pkg.phase_contracts}

    if pkg.resume_block.current_phase and pkg.resume_block.current_phase not in phase_ids:
        errors.append(
            ValidationError(
                code="resume_phase_not_found",
                message=(
                    f"ResumeBlock.current_phase '{pkg.resume_block.current_phase}' "
                    f"does not match any PhaseContract"
                ),
                severity="error",
                context={"current_phase": pkg.resume_block.current_phase},
            )
        )

    return errors


def _check_conservative_status(pkg: PlanPackage) -> list[ValidationError]:
    """Invariant 10: conservative status — unknown/unconfirmed should not be validated."""
    errors: list[ValidationError] = []

    # If the package has no phase contracts but claims validated, that's suspicious
    if pkg.planning_status.value == "validated" and not pkg.phase_contracts:
        errors.append(
            ValidationError(
                code="premature_validated",
                message="PlanPackage is validated but has no phase contracts",
                severity="warning",
                context={},
            )
        )

    return errors
