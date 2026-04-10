"""Tests for Pydantic models in aeloon/plugins/SoftwareEngineering/task.py.

Covers all 23 data models with round-trip serialization and
topological ordering tests.
"""

from __future__ import annotations

from aeloon.plugins.SoftwareEngineering.task import (
    ArchitectureGraph,
    Execution,
    ExecutionMetrics,
    ExecutionState,
    FileChange,
    LogEntry,
    ModuleDef,
    ModuleType,
    NextAction,
    NodeType,
    Project,
    ProjectStatus,
    Provenance,
    Requirement,
    RetryPolicy,
    SEBudget,
    SETaskGraph,
    SETaskNode,
    SEValidation,
    TechConstraint,
    TestResult,
    ValidationStatus,
    Violation,
)

# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------


def test_project_status_enum_values() -> None:
    assert ProjectStatus.CREATED == "created"
    assert ProjectStatus.ANALYZED == "analyzed"
    assert ProjectStatus.PLANNED == "planned"
    assert ProjectStatus.RUNNING == "running"
    assert ProjectStatus.COMPLETED == "completed"
    assert ProjectStatus.FAILED == "failed"
    assert ProjectStatus.CANCELLED == "cancelled"


def test_module_type_enum_values() -> None:
    assert ModuleType.CORE == "core"
    assert ModuleType.API == "api"
    assert ModuleType.SERVICE == "service"
    assert ModuleType.TEST == "test"
    assert ModuleType.CONFIG == "config"


def test_node_type_enum_values() -> None:
    assert NodeType.SCAFFOLD == "scaffold"
    assert NodeType.IMPLEMENT == "implement"
    assert NodeType.TEST == "test"
    assert NodeType.VALIDATE == "validate"
    assert NodeType.FIX == "fix"


def test_execution_state_enum_values() -> None:
    assert ExecutionState.PENDING == "pending"
    assert ExecutionState.RUNNING == "running"
    assert ExecutionState.VALIDATED == "validated"
    assert ExecutionState.FAILED == "failed"
    assert ExecutionState.CANCELLED == "cancelled"


def test_validation_status_enum_values() -> None:
    assert ValidationStatus.PASSED == "passed"
    assert ValidationStatus.FAILED == "failed"
    assert ValidationStatus.PARTIAL == "partial"


def test_next_action_enum_values() -> None:
    assert NextAction.DELIVER == "deliver"
    assert NextAction.RETRY == "retry"
    assert NextAction.REPLAN == "replan"
    assert NextAction.ESCALATE == "escalate"


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------


def test_project_auto_generates_ids() -> None:
    p = Project(description="test")
    assert p.project_id.startswith("proj_")
    assert p.trace_id.startswith("trace_")


def test_project_ids_are_unique() -> None:
    p1 = Project(description="A")
    p2 = Project(description="B")
    assert p1.project_id != p2.project_id
    assert p1.trace_id != p2.trace_id


def test_budget_defaults() -> None:
    b = SEBudget()
    assert b.max_tokens == 50_000
    assert b.max_seconds == 600
    assert b.max_tool_calls == 100
    assert b.max_repair_cycles == 3


def test_tech_constraint_defaults() -> None:
    tc = TechConstraint()
    assert tc.language == "python"
    assert tc.test_framework == "pytest"
    assert tc.linter == "ruff"
    assert tc.line_length == 100


def test_project_default_status() -> None:
    p = Project(description="test")
    assert p.status == ProjectStatus.CREATED


# ---------------------------------------------------------------------------
# Round-trip serialization (all 23 models)
# ---------------------------------------------------------------------------


def test_tech_constraint_round_trip() -> None:
    m = TechConstraint(language="go", framework="gin")
    assert TechConstraint.model_validate(m.model_dump(mode="json")).language == "go"


def test_requirement_round_trip() -> None:
    m = Requirement(functional=["do X"], non_functional=["fast"])
    r = Requirement.model_validate(m.model_dump(mode="json"))
    assert r.functional == ["do X"]
    assert r.non_functional == ["fast"]


def test_se_budget_round_trip() -> None:
    m = SEBudget(max_tokens=100_000)
    assert SEBudget.model_validate(m.model_dump(mode="json")).max_tokens == 100_000


def test_module_def_round_trip() -> None:
    m = ModuleDef(id="m1", name="core", dependencies=["m2"])
    r = ModuleDef.model_validate(m.model_dump(mode="json"))
    assert r.id == "m1"
    assert r.dependencies == ["m2"]


def test_architecture_graph_round_trip() -> None:
    m = ArchitectureGraph(
        project_id="p1",
        modules=[ModuleDef(id="m1", name="api"), ModuleDef(id="m2", name="data")],
    )
    r = ArchitectureGraph.model_validate(m.model_dump(mode="json"))
    assert len(r.modules) == 2
    assert r.project_id == "p1"


def test_retry_policy_round_trip() -> None:
    m = RetryPolicy(max_retries=5)
    assert RetryPolicy.model_validate(m.model_dump(mode="json")).max_retries == 5


def test_se_task_node_round_trip() -> None:
    m = SETaskNode(id="n1", node_type=NodeType.IMPLEMENT, objective="do stuff")
    r = SETaskNode.model_validate(m.model_dump(mode="json"))
    assert r.id == "n1"
    assert r.node_type == NodeType.IMPLEMENT


def test_se_task_graph_round_trip() -> None:
    m = SETaskGraph(
        project_id="p1",
        nodes=[
            SETaskNode(id="n1", objective="step1"),
            SETaskNode(id="n2", objective="step2", dependencies=["n1"]),
        ],
    )
    r = SETaskGraph.model_validate(m.model_dump(mode="json"))
    assert len(r.nodes) == 2


def test_log_entry_round_trip() -> None:
    m = LogEntry(level="ERROR", msg="fail")
    assert LogEntry.model_validate(m.model_dump(mode="json")).level == "ERROR"


def test_execution_metrics_round_trip() -> None:
    m = ExecutionMetrics(tokens_used=100, elapsed_seconds=5.0)
    r = ExecutionMetrics.model_validate(m.model_dump(mode="json"))
    assert r.tokens_used == 100


def test_file_change_round_trip() -> None:
    m = FileChange(path="a.py", action="modify")
    r = FileChange.model_validate(m.model_dump(mode="json"))
    assert r.action == "modify"


def test_test_result_round_trip() -> None:
    m = TestResult(total=10, passed=8, failed=2)
    r = TestResult.model_validate(m.model_dump(mode="json"))
    assert r.failed == 2


def test_execution_round_trip() -> None:
    m = Execution(project_id="p1", node_id="n1", output="result")
    r = Execution.model_validate(m.model_dump(mode="json"))
    assert r.output == "result"


def test_violation_round_trip() -> None:
    m = Violation(rule="lint", msg="bad code")
    r = Violation.model_validate(m.model_dump(mode="json"))
    assert r.rule == "lint"


def test_se_validation_round_trip() -> None:
    m = SEValidation(status=ValidationStatus.FAILED, next_action=NextAction.RETRY)
    r = SEValidation.model_validate(m.model_dump(mode="json"))
    assert r.status == ValidationStatus.FAILED
    assert r.next_action == NextAction.RETRY


def test_provenance_round_trip() -> None:
    m = Provenance(entity_id="e1", generated_by="llm")
    r = Provenance.model_validate(m.model_dump(mode="json"))
    assert r.entity_id == "e1"


def test_project_round_trip() -> None:
    m = Project(description="build an API")
    r = Project.model_validate(m.model_dump(mode="json"))
    assert r.description == "build an API"
    assert r.status == ProjectStatus.CREATED


# ---------------------------------------------------------------------------
# Topological ordering — ArchitectureGraph
# ---------------------------------------------------------------------------


def test_arch_graph_topological_order_linear() -> None:
    """A -> B -> C should return [A, B, C]."""
    g = ArchitectureGraph(
        project_id="p1",
        modules=[
            ModuleDef(id="A", name="a", dependencies=[]),
            ModuleDef(id="B", name="b", dependencies=["A"]),
            ModuleDef(id="C", name="c", dependencies=["B"]),
        ],
    )
    order = g.topological_order()
    assert order.index("A") < order.index("B") < order.index("C")


def test_arch_graph_topological_order_cycle_raises() -> None:
    """Cyclic deps should raise ValueError."""
    g = ArchitectureGraph(
        project_id="p1",
        modules=[
            ModuleDef(id="A", name="a", dependencies=["B"]),
            ModuleDef(id="B", name="b", dependencies=["A"]),
        ],
    )
    try:
        g.topological_order()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cyclic" in str(e) or "cycle" in str(e).lower()


def test_arch_graph_parallel_groups_diamond() -> None:
    """Diamond: A -> B, A -> C, B -> D, C -> D → [[A], [B, C], [D]]."""
    g = ArchitectureGraph(
        project_id="p1",
        modules=[
            ModuleDef(id="A", name="a", dependencies=[]),
            ModuleDef(id="B", name="b", dependencies=["A"]),
            ModuleDef(id="C", name="c", dependencies=["A"]),
            ModuleDef(id="D", name="d", dependencies=["B", "C"]),
        ],
    )
    groups = g.get_parallel_groups()
    assert len(groups) == 3
    assert groups[0] == ["A"]
    assert set(groups[1]) == {"B", "C"}
    assert groups[2] == ["D"]


# ---------------------------------------------------------------------------
# Topological ordering — SETaskGraph
# ---------------------------------------------------------------------------


def test_task_graph_topological_order_linear() -> None:
    a = SETaskNode(id="A", objective="step A", dependencies=[])
    b = SETaskNode(id="B", objective="step B", dependencies=["A"])
    c = SETaskNode(id="C", objective="step C", dependencies=["B"])
    g = SETaskGraph(project_id="p1", nodes=[a, b, c])
    order = g.topological_order()
    ids = [n.id for n in order]
    assert ids.index("A") < ids.index("B") < ids.index("C")


def test_task_graph_get_node() -> None:
    n = SETaskNode(id="n1", objective="test")
    g = SETaskGraph(project_id="p1", nodes=[n])
    assert g.get_node("n1") is n
    assert g.get_node("missing") is None


def test_task_graph_waves_parallel() -> None:
    a = SETaskNode(id="A", objective="a", dependencies=[])
    b = SETaskNode(id="B", objective="b", dependencies=[])
    c = SETaskNode(id="C", objective="c", dependencies=["A", "B"])
    g = SETaskGraph(project_id="p1", nodes=[a, b, c])
    waves = g.get_waves()
    assert len(waves) == 2
    assert set(n.id for n in waves[0]) == {"A", "B"}
    assert waves[1][0].id == "C"


def test_task_graph_waves_cycle_raises() -> None:
    a = SETaskNode(id="A", objective="a", dependencies=["B"])
    b = SETaskNode(id="B", objective="b", dependencies=["A"])
    g = SETaskGraph(project_id="p1", nodes=[a, b])
    try:
        g.get_waves()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Validation model — specific scenarios
# ---------------------------------------------------------------------------


def test_validation_passed_defaults() -> None:
    v = SEValidation()
    assert v.status == ValidationStatus.PASSED
    assert v.next_action == NextAction.DELIVER
    assert v.violations == []
    assert v.confidence == 1.0


def test_project_with_architecture() -> None:
    p = Project(
        description="test",
        architecture=ArchitectureGraph(
            project_id="p1",
            modules=[ModuleDef(id="m1", name="core")],
        ),
    )
    assert p.architecture is not None
    assert len(p.architecture.modules) == 1
