"""Tests for PluginCreator data models — enums, round-trip, defaults, properties."""

from __future__ import annotations

from aeloon.plugins.PluginCreator.models import (
    ArtifactSpec,
    ArtifactStatus,
    ArtifactType,
    DeferItem,
    DeferLedger,
    GateStatus,
    PlanItemKind,
    PlanningStatus,
    PlanPackage,
    ResumeBlock,
)

# ---------------------------------------------------------------------------
# Enum value assertions
# ---------------------------------------------------------------------------


class TestPlanningStatus:
    def test_values(self) -> None:
        assert PlanningStatus.PLANNED == "planned"
        assert PlanningStatus.PARTIAL == "partial"
        assert PlanningStatus.BLOCKED == "blocked"
        assert PlanningStatus.VALIDATED == "validated"
        assert PlanningStatus.DEFERRED == "deferred"


class TestGateStatus:
    def test_values(self) -> None:
        assert GateStatus.PLANNED == "planned"
        assert GateStatus.PASSED == "passed"
        assert GateStatus.FAILED == "failed"
        assert GateStatus.WAIVED == "waived"
        assert GateStatus.BLOCKED == "blocked"


class TestArtifactStatus:
    def test_values(self) -> None:
        assert ArtifactStatus.PLANNED == "planned"
        assert ArtifactStatus.PARTIAL == "partial"
        assert ArtifactStatus.VALIDATED == "validated"
        assert ArtifactStatus.MISSING == "missing"
        assert ArtifactStatus.SUPERSEDED == "superseded"


class TestPlanItemKind:
    def test_values(self) -> None:
        assert PlanItemKind.ANALYSIS == "analysis"
        assert PlanItemKind.DESIGN == "design"
        assert PlanItemKind.GENERATE == "generate"
        assert PlanItemKind.VALIDATE == "validate"
        assert PlanItemKind.REPAIR == "repair"
        assert PlanItemKind.REPORT == "report"
        assert PlanItemKind.REVIEW == "review"
        assert PlanItemKind.GATE == "gate"
        assert PlanItemKind.RESUME == "resume"


# ---------------------------------------------------------------------------
# PlanPackage round-trip
# ---------------------------------------------------------------------------


class TestPlanPackageRoundTrip:
    def _make_minimal_package(self) -> PlanPackage:
        from aeloon.plugins.PluginCreator.models import (
            BackgroundSnapshot,
            DesignReview,
            ProgrammeStructure,
        )

        return PlanPackage(
            project_id="test_proj",
            background_snapshot=BackgroundSnapshot(summary="test"),
            programme_structure=ProgrammeStructure(phases=["p1"]),
            design_review=DesignReview(scope_framing="test"),
            resume_block=ResumeBlock(
                current_phase="p1",
                next_safe_action="start",
                next_prompt_suggestion="begin",
            ),
        )

    def test_model_dump_and_validate(self) -> None:
        pkg = self._make_minimal_package()
        data = pkg.model_dump(mode="json")
        restored = PlanPackage.model_validate(data)
        assert restored.project_id == pkg.project_id
        assert restored.schema_version == "1.0"
        assert restored.plan_package_id == pkg.plan_package_id

    def test_default_field_generation(self) -> None:
        pkg = self._make_minimal_package()
        assert pkg.plan_package_id.startswith("pp_")
        assert pkg.schema_version == "1.0"
        assert pkg.planning_status == PlanningStatus.PLANNED
        assert pkg.created_at  # non-empty ISO timestamp
        assert pkg.updated_at


# ---------------------------------------------------------------------------
# DeferLedger property
# ---------------------------------------------------------------------------


class TestDeferLedger:
    def test_active_item_ids_empty(self) -> None:
        ledger = DeferLedger()
        assert ledger.active_item_ids == []

    def test_active_item_ids_filters(self) -> None:
        items = [
            DeferItem(
                title="a",
                reason_deferred="r",
                target_phase="p1",
                reentry_condition="c",
                status="deferred",
            ),
            DeferItem(
                id="def_2",
                title="b",
                reason_deferred="r",
                target_phase="p2",
                reentry_condition="c",
                status="reentered",
            ),
        ]
        ledger = DeferLedger(items=items)
        assert len(ledger.active_item_ids) == 1
        assert items[0].id in ledger.active_item_ids


# ---------------------------------------------------------------------------
# ResumeBlock field presence
# ---------------------------------------------------------------------------


class TestResumeBlock:
    def test_required_fields(self) -> None:
        rb = ResumeBlock(
            current_phase="p1",
            next_safe_action="do X",
            next_prompt_suggestion="say Y",
        )
        assert rb.current_phase == "p1"
        assert rb.resume_version == "1"
        assert rb.generated_at  # non-empty

    def test_default_lists(self) -> None:
        rb = ResumeBlock(
            current_phase="p1",
            next_safe_action="do X",
            next_prompt_suggestion="say Y",
        )
        assert rb.completed_artifacts == []
        assert rb.open_blockers == []
        assert rb.deferred_items == []


# ---------------------------------------------------------------------------
# ArtifactSpec defaults
# ---------------------------------------------------------------------------


class TestArtifactSpec:
    def test_auto_id(self) -> None:
        spec = ArtifactSpec(type=ArtifactType.PLAN, producer="kernel")
        assert spec.id.startswith("art_")
        assert spec.status == ArtifactStatus.PLANNED
