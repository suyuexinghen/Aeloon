"""Tests for LinearPlanner in aeloon/plugins/science/planner.py."""

from __future__ import annotations

from aeloon.plugins.ScienceResearch.planner import LinearPlanner
from aeloon.plugins.ScienceResearch.task import ScienceTaskGraph, Task


def _make_task(goal: str = "Summarise quantum computing research") -> Task:
    return Task(goal=goal)


# ---------------------------------------------------------------------------
# Basic plan generation
# ---------------------------------------------------------------------------


def test_plan_returns_science_task_graph():
    planner = LinearPlanner()
    task = _make_task()
    graph = planner.plan(task)
    assert isinstance(graph, ScienceTaskGraph)


def test_plan_task_id_matches_input_task():
    planner = LinearPlanner()
    task = _make_task()
    graph = planner.plan(task)
    assert graph.task_id == task.task_id


def test_plan_generates_exactly_three_nodes():
    planner = LinearPlanner()
    task = _make_task()
    graph = planner.plan(task)
    assert len(graph.nodes) == 3


# ---------------------------------------------------------------------------
# Node ordering: first has no deps; each subsequent depends on prior
# ---------------------------------------------------------------------------


def test_first_node_has_no_dependencies():
    planner = LinearPlanner()
    graph = planner.plan(_make_task())
    assert graph.nodes[0].dependencies == []


def test_second_node_depends_on_first():
    planner = LinearPlanner()
    graph = planner.plan(_make_task())
    assert graph.nodes[0].id in graph.nodes[1].dependencies


def test_third_node_depends_on_second():
    planner = LinearPlanner()
    graph = planner.plan(_make_task())
    assert graph.nodes[1].id in graph.nodes[2].dependencies


def test_topological_order_matches_node_list():
    planner = LinearPlanner()
    graph = planner.plan(_make_task())
    ordered = graph.topological_order()
    node_ids = [n.id for n in graph.nodes]
    ordered_ids = [n.id for n in ordered]
    assert node_ids == ordered_ids


# ---------------------------------------------------------------------------
# Node content correctness
# ---------------------------------------------------------------------------


def test_each_node_has_non_empty_objective():
    planner = LinearPlanner()
    graph = planner.plan(_make_task())
    for node in graph.nodes:
        assert node.objective.strip() != ""


def test_each_node_has_non_empty_id():
    planner = LinearPlanner()
    graph = planner.plan(_make_task())
    for node in graph.nodes:
        assert node.id.strip() != ""


def test_each_node_has_candidate_capabilities():
    planner = LinearPlanner()
    graph = planner.plan(_make_task())
    for node in graph.nodes:
        assert len(node.candidate_capabilities) > 0


def test_no_duplicate_node_ids():
    planner = LinearPlanner()
    graph = planner.plan(_make_task())
    ids = [n.id for n in graph.nodes]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Goal is included in node objectives
# ---------------------------------------------------------------------------


def test_goal_appears_in_node_objectives():
    goal = "unique_research_goal_xyz"
    planner = LinearPlanner()
    graph = planner.plan(Task(goal=goal))
    combined = " ".join(n.objective for n in graph.nodes)
    assert goal in combined
