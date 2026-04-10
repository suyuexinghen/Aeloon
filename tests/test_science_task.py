"""Tests for Pydantic models in aeloon/plugins/science/task.py."""

from __future__ import annotations

from aeloon.plugins.ScienceResearch.task import (
    Budget,
    Execution,
    ExecutionState,
    NextAction,
    ScienceTaskGraph,
    ScienceTaskNode,
    Task,
    TaskStatus,
    Validation,
    ValidationStatus,
)

# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------


def test_task_status_enum_values():
    assert TaskStatus.CREATED == "created"
    assert TaskStatus.PLANNED == "planned"
    assert TaskStatus.RUNNING == "running"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"


def test_execution_state_enum_values():
    assert ExecutionState.PENDING == "pending"
    assert ExecutionState.RUNNING == "running"
    assert ExecutionState.VALIDATED == "validated"
    assert ExecutionState.FAILED == "failed"


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------


def test_task_auto_generates_task_id_and_trace_id():
    task = Task(goal="Test goal")
    assert task.task_id.startswith("task_")
    assert task.trace_id.startswith("trace_")


def test_task_ids_are_unique():
    t1 = Task(goal="A")
    t2 = Task(goal="B")
    assert t1.task_id != t2.task_id
    assert t1.trace_id != t2.trace_id


def test_budget_defaults():
    b = Budget()
    assert b.max_tokens == 50_000
    assert b.max_seconds == 600


def test_task_default_status_and_priority():
    task = Task(goal="some goal")
    assert task.status == TaskStatus.CREATED
    from aeloon.plugins.ScienceResearch.task import Priority

    assert task.priority == Priority.NORMAL


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------


def test_task_round_trip():
    task = Task(goal="Summarise protein folding research")
    data = task.model_dump(mode="json")
    restored = Task.model_validate(data)
    assert restored.task_id == task.task_id
    assert restored.goal == task.goal
    assert restored.status == task.status


def test_science_task_node_round_trip():
    node = ScienceTaskNode(
        id="n1",
        objective="Search for papers",
        candidate_capabilities=["web_search"],
    )
    data = node.model_dump(mode="json")
    restored = ScienceTaskNode.model_validate(data)
    assert restored.id == node.id
    assert restored.objective == node.objective
    assert restored.candidate_capabilities == ["web_search"]


def test_science_task_graph_round_trip():
    graph = ScienceTaskGraph(
        task_id="task_abc",
        nodes=[
            ScienceTaskNode(id="n1", objective="step1"),
            ScienceTaskNode(id="n2", objective="step2", dependencies=["n1"]),
        ],
    )
    data = graph.model_dump(mode="json")
    restored = ScienceTaskGraph.model_validate(data)
    assert restored.task_id == "task_abc"
    assert len(restored.nodes) == 2


def test_execution_round_trip():
    ex = Execution(task_id="task_x", node_id="n1", output="some output")
    data = ex.model_dump(mode="json")
    restored = Execution.model_validate(data)
    assert restored.task_id == "task_x"
    assert restored.node_id == "n1"
    assert restored.output == "some output"


def test_validation_round_trip():
    v = Validation(status=ValidationStatus.PASSED)
    data = v.model_dump(mode="json")
    restored = Validation.model_validate(data)
    assert restored.status == ValidationStatus.PASSED
    assert restored.next_action == NextAction.DELIVER


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------


def test_topological_order_linear_three_nodes():
    """A -> B -> C should return [A, B, C]."""
    a = ScienceTaskNode(id="A", objective="step A", dependencies=[])
    b = ScienceTaskNode(id="B", objective="step B", dependencies=["A"])
    c = ScienceTaskNode(id="C", objective="step C", dependencies=["B"])
    graph = ScienceTaskGraph(task_id="t1", nodes=[a, b, c])
    order = graph.topological_order()
    ids = [n.id for n in order]
    assert ids.index("A") < ids.index("B") < ids.index("C")


def test_topological_order_single_node():
    node = ScienceTaskNode(id="only", objective="do everything")
    graph = ScienceTaskGraph(task_id="t1", nodes=[node])
    order = graph.topological_order()
    assert len(order) == 1
    assert order[0].id == "only"


def test_topological_order_returns_all_nodes():
    nodes = [
        ScienceTaskNode(id="n1", objective="first"),
        ScienceTaskNode(id="n2", objective="second", dependencies=["n1"]),
        ScienceTaskNode(id="n3", objective="third", dependencies=["n2"]),
    ]
    graph = ScienceTaskGraph(task_id="t_order", nodes=nodes)
    order = graph.topological_order()
    assert len(order) == 3


# ---------------------------------------------------------------------------
# Validation model — specific scenarios
# ---------------------------------------------------------------------------


def test_validation_passed_has_deliver_action_and_empty_violations():
    v = Validation(status=ValidationStatus.PASSED, violations=[], next_action=NextAction.DELIVER)
    assert v.violations == []
    assert v.next_action == NextAction.DELIVER


def test_validation_default_is_passed():
    v = Validation()
    assert v.status == ValidationStatus.PASSED
    assert v.next_action == NextAction.DELIVER
    assert v.violations == []
