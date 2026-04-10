"""Tests for PluginCreator cross-model invariant validator."""

from __future__ import annotations

from aeloon.plugins.PluginCreator.models import (
    BackgroundSnapshot,
    DecisionPoint,
    DeferItem,
    DeferLedger,
    DesignReview,
    GateStatus,
    GateType,
    PhaseContract,
    PlanItem,
    PlanItemKind,
    PlanningStatus,
    PlanPackage,
    ProgrammeStructure,
    ResumeBlock,
    VerificationGate,
)
from aeloon.plugins.PluginCreator.validator.plan_package import (
    ValidationError,
    validate_plan_package,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_package(**overrides) -> PlanPackage:
    """Build a minimal valid PlanPackage for testing."""
    phase_id = overrides.pop("phase_id", "phase_1")
    item_id = overrides.pop("item_id", "item_1")

    defaults = dict(
        project_id="test_proj",
        background_snapshot=BackgroundSnapshot(summary="test"),
        programme_structure=ProgrammeStructure(phases=[phase_id]),
        design_review=DesignReview(scope_framing="test"),
        phase_contracts=[
            PhaseContract(
                phase_id=phase_id,
                phase_name="Analysis",
                goal="Understand requirements",
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
            next_safe_action="start",
            next_prompt_suggestion="begin",
        ),
    )
    defaults.update(overrides)
    return PlanPackage(**defaults)


def _error_codes(errors: list[ValidationError]) -> set[str]:
    return {e.code for e in errors}


# ---------------------------------------------------------------------------
# Test: valid package
# ---------------------------------------------------------------------------


class TestValidPackage:
    def test_minimal_valid_package_passes(self) -> None:
        pkg = _make_valid_package()
        errors = validate_plan_package(pkg)
        assert errors == [], f"Unexpected errors: {errors}"


# ---------------------------------------------------------------------------
# Test: duplicate IDs
# ---------------------------------------------------------------------------


class TestDuplicateIds:
    def test_duplicate_plan_item_and_phase_id(self) -> None:
        pkg = _make_valid_package(
            phase_contracts=[
                PhaseContract(phase_id="dup_id", phase_name="P", goal="G"),
            ],
            plan_items=[
                PlanItem(id="dup_id", title="T", kind=PlanItemKind.ANALYSIS),
            ],
            resume_block=ResumeBlock(
                current_phase="dup_id",
                next_safe_action="x",
                next_prompt_suggestion="y",
            ),
        )
        errors = validate_plan_package(pkg)
        assert "duplicate_id" in _error_codes(errors)


# ---------------------------------------------------------------------------
# Test: unresolvable references
# ---------------------------------------------------------------------------


class TestUnresolvableReferences:
    def test_task_id_references_nonexistent_item(self) -> None:
        pkg = _make_valid_package(
            phase_contracts=[
                PhaseContract(
                    phase_id="p1",
                    phase_name="P",
                    goal="G",
                    task_ids=["nonexistent_item"],
                ),
            ],
            resume_block=ResumeBlock(
                current_phase="p1",
                next_safe_action="x",
                next_prompt_suggestion="y",
            ),
        )
        errors = validate_plan_package(pkg)
        assert "unresolvable_reference" in _error_codes(errors)


# ---------------------------------------------------------------------------
# Test: DAG cycle
# ---------------------------------------------------------------------------


class TestDagCycle:
    def test_plan_item_cycle(self) -> None:
        pkg = _make_valid_package(
            plan_items=[
                PlanItem(id="a", title="A", kind=PlanItemKind.GENERATE, depends_on=["b"]),
                PlanItem(id="b", title="B", kind=PlanItemKind.GENERATE, depends_on=["a"]),
            ],
        )
        errors = validate_plan_package(pkg)
        assert "dag_cycle" in _error_codes(errors)


# ---------------------------------------------------------------------------
# Test: phase/task mismatch
# ---------------------------------------------------------------------------


class TestPhaseTaskMismatch:
    def test_item_claims_wrong_phase(self) -> None:
        pkg = _make_valid_package(
            phase_contracts=[
                PhaseContract(phase_id="p1", phase_name="P1", goal="G", task_ids=["i1"]),
            ],
            plan_items=[
                PlanItem(id="i1", title="T", kind=PlanItemKind.ANALYSIS, phase_id="p2"),
            ],
            resume_block=ResumeBlock(
                current_phase="p1",
                next_safe_action="x",
                next_prompt_suggestion="y",
            ),
        )
        errors = validate_plan_package(pkg)
        assert "phase_task_mismatch" in _error_codes(errors)

    def test_item_not_in_phase_task_ids(self) -> None:
        pkg = _make_valid_package(
            phase_contracts=[
                PhaseContract(phase_id="p1", phase_name="P1", goal="G", task_ids=[]),
            ],
            plan_items=[
                PlanItem(id="i1", title="T", kind=PlanItemKind.ANALYSIS, phase_id="p1"),
            ],
            resume_block=ResumeBlock(
                current_phase="p1",
                next_safe_action="x",
                next_prompt_suggestion="y",
            ),
        )
        errors = validate_plan_package(pkg)
        assert "phase_task_mismatch" in _error_codes(errors)


# ---------------------------------------------------------------------------
# Test: gate/phase conflict
# ---------------------------------------------------------------------------


class TestGatePhaseConflict:
    def test_failed_gate_with_validated_phase(self) -> None:
        pkg = _make_valid_package(
            phase_contracts=[
                PhaseContract(
                    phase_id="p1",
                    phase_name="P1",
                    goal="G",
                    status=PlanningStatus.VALIDATED,
                ),
            ],
            resume_block=ResumeBlock(
                current_phase="p1",
                next_safe_action="x",
                next_prompt_suggestion="y",
            ),
            verification_gates=[
                VerificationGate(
                    id="g1",
                    name="Gate 1",
                    phase_id="p1",
                    gate_type=GateType.BUILD,
                    required=True,
                    status=GateStatus.FAILED,
                ),
            ],
        )
        errors = validate_plan_package(pkg)
        assert "gate_phase_conflict" in _error_codes(errors)


# ---------------------------------------------------------------------------
# Test: gate no evidence
# ---------------------------------------------------------------------------


class TestGateNoEvidence:
    def test_passed_gate_without_evidence(self) -> None:
        pkg = _make_valid_package(
            verification_gates=[
                VerificationGate(
                    id="g1",
                    name="Gate 1",
                    phase_id="phase_1",
                    gate_type=GateType.BUILD,
                    status=GateStatus.PASSED,
                ),
            ],
        )
        errors = validate_plan_package(pkg)
        assert "gate_no_evidence" in _error_codes(errors)

    def test_passed_gate_with_waiver_is_ok(self) -> None:
        pkg = _make_valid_package(
            verification_gates=[
                VerificationGate(
                    id="g1",
                    name="Gate 1",
                    phase_id="phase_1",
                    gate_type=GateType.BUILD,
                    status=GateStatus.PASSED,
                    waiver_reason="Accepted by lead",
                ),
            ],
        )
        errors = validate_plan_package(pkg)
        gate_errors = [e for e in errors if e.code == "gate_no_evidence"]
        assert gate_errors == []


# ---------------------------------------------------------------------------
# Test: decision point no resolution
# ---------------------------------------------------------------------------


class TestDecisionPointNoResolution:
    def test_resolved_without_option(self) -> None:
        pkg = _make_valid_package(
            decision_points=[
                DecisionPoint(question="Q?", status="resolved"),
            ],
        )
        errors = validate_plan_package(pkg)
        assert "decision_point_no_resolution" in _error_codes(errors)


# ---------------------------------------------------------------------------
# Test: defer item incomplete
# ---------------------------------------------------------------------------


class TestDeferItemIncomplete:
    def test_missing_reason_deferred(self) -> None:
        pkg = _make_valid_package(
            defer_ledger=DeferLedger(
                items=[
                    DeferItem(
                        title="X",
                        reason_deferred="",
                        target_phase="p1",
                        reentry_condition="c",
                    ),
                ],
            ),
        )
        errors = validate_plan_package(pkg)
        assert "defer_item_incomplete" in _error_codes(errors)
        field_names = {e.context.get("field") for e in errors if e.code == "defer_item_incomplete"}
        assert "reason_deferred" in field_names

    def test_missing_target_phase(self) -> None:
        pkg = _make_valid_package(
            defer_ledger=DeferLedger(
                items=[
                    DeferItem(
                        title="X",
                        reason_deferred="r",
                        target_phase="",
                        reentry_condition="c",
                    ),
                ],
            ),
        )
        errors = validate_plan_package(pkg)
        assert "defer_item_incomplete" in _error_codes(errors)

    def test_missing_reentry_condition(self) -> None:
        pkg = _make_valid_package(
            defer_ledger=DeferLedger(
                items=[
                    DeferItem(
                        title="X",
                        reason_deferred="r",
                        target_phase="p1",
                        reentry_condition="",
                    ),
                ],
            ),
        )
        errors = validate_plan_package(pkg)
        assert "defer_item_incomplete" in _error_codes(errors)


# ---------------------------------------------------------------------------
# Test: resume block bad reference
# ---------------------------------------------------------------------------


class TestResumeBlockBadReference:
    def test_current_phase_not_found(self) -> None:
        pkg = _make_valid_package(
            resume_block=ResumeBlock(
                current_phase="nonexistent_phase",
                next_safe_action="x",
                next_prompt_suggestion="y",
            ),
        )
        errors = validate_plan_package(pkg)
        assert "resume_phase_not_found" in _error_codes(errors)
