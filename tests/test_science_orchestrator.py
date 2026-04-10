"""Tests for science orchestrators in aeloon/plugins/science/orchestrator.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.plugins.ScienceResearch.orchestrator import (
    DAGOrchestrator,
    SequentialOrchestrator,
    _build_node_prompt,
)
from aeloon.plugins.ScienceResearch.task import (
    ExecutionState,
    ScienceTaskGraph,
    ScienceTaskNode,
    Task,
)


def _make_two_node_graph(task_id: str) -> ScienceTaskGraph:
    n1 = ScienceTaskNode(id=f"{task_id}_n1", objective="Search for papers", dependencies=[])
    n2 = ScienceTaskNode(id=f"{task_id}_n2", objective="Summarise results", dependencies=[n1.id])
    return ScienceTaskGraph(task_id=task_id, nodes=[n1, n2])


def _make_mock_runtime(return_value: str = "fixed output") -> MagicMock:
    runtime = MagicMock()
    runtime.process_direct = AsyncMock(return_value=return_value)
    runtime.tool_execute = AsyncMock(return_value="web result")
    runtime.add_deep_profile_section = MagicMock()
    runtime.supports_async_tool_execute = True
    return runtime


def _make_fetch_node(task_id: str) -> ScienceTaskNode:
    return ScienceTaskNode(
        id=f"{task_id}_fetch",
        objective="Fetch and extract key information from search results",
        dependencies=[f"{task_id}_search"],
        candidate_capabilities=["web_fetch"],
    )


# ---------------------------------------------------------------------------
# Happy-path: 2-node graph, both succeed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_two_executions_for_two_node_graph():
    task = Task(goal="Test two-node run")
    graph = _make_two_node_graph(task.task_id)
    runtime = _make_mock_runtime("some output")

    orchestrator = SequentialOrchestrator(runtime=runtime)
    executions = await orchestrator.run(task, graph)

    assert len(executions) == 2


@pytest.mark.asyncio
async def test_run_all_executions_are_waiting_validation_on_success():
    task = Task(goal="All success")
    graph = _make_two_node_graph(task.task_id)
    runtime = _make_mock_runtime("nice output")

    orchestrator = SequentialOrchestrator(runtime=runtime)
    executions = await orchestrator.run(task, graph)

    for ex in executions:
        assert ex.state == ExecutionState.WAITING_VALIDATION


@pytest.mark.asyncio
async def test_run_output_field_populated():
    task = Task(goal="Output populated")
    graph = _make_two_node_graph(task.task_id)
    runtime = _make_mock_runtime("returned output text")

    orchestrator = SequentialOrchestrator(runtime=runtime)
    executions = await orchestrator.run(task, graph)

    for ex in executions:
        assert ex.output == "returned output text"


# ---------------------------------------------------------------------------
# Failure on second node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_second_node_failure_returns_two_executions():
    task = Task(goal="Second fails")
    graph = _make_two_node_graph(task.task_id)

    runtime = MagicMock()
    runtime.process_direct = AsyncMock(side_effect=["first output", Exception("node 2 boom")])
    runtime.add_deep_profile_section = MagicMock()

    orchestrator = SequentialOrchestrator(runtime=runtime)
    executions = await orchestrator.run(task, graph)

    assert len(executions) == 2


@pytest.mark.asyncio
async def test_run_first_execution_is_waiting_validation_when_second_fails():
    task = Task(goal="Second fails v2")
    graph = _make_two_node_graph(task.task_id)

    runtime = MagicMock()
    runtime.process_direct = AsyncMock(side_effect=["first output", Exception("boom")])
    runtime.add_deep_profile_section = MagicMock()

    orchestrator = SequentialOrchestrator(runtime=runtime)
    executions = await orchestrator.run(task, graph)

    assert executions[0].state == ExecutionState.WAITING_VALIDATION


@pytest.mark.asyncio
async def test_run_second_execution_is_failed_with_error():
    task = Task(goal="Second fails v3")
    graph = _make_two_node_graph(task.task_id)

    runtime = MagicMock()
    runtime.process_direct = AsyncMock(
        side_effect=["first output", Exception("node 2 error message")]
    )
    runtime.add_deep_profile_section = MagicMock()

    orchestrator = SequentialOrchestrator(runtime=runtime)
    executions = await orchestrator.run(task, graph)

    assert executions[1].state == ExecutionState.FAILED
    assert executions[1].error == "node 2 error message"


# ---------------------------------------------------------------------------
# _build_node_prompt content
# ---------------------------------------------------------------------------


def test_build_node_prompt_contains_task_goal():
    task = Task(goal="My research goal")
    node = ScienceTaskNode(id="n1", objective="Search for X")
    prompt = _build_node_prompt(task, node, prior_context=[])
    assert "My research goal" in prompt


def test_build_node_prompt_contains_node_objective():
    task = Task(goal="A goal")
    node = ScienceTaskNode(id="n1", objective="Very specific objective text")
    prompt = _build_node_prompt(task, node, prior_context=[])
    assert "Very specific objective text" in prompt


def test_build_node_prompt_includes_prior_context():
    task = Task(goal="Goal with context")
    node = ScienceTaskNode(id="n2", objective="Use prior results")
    prior = ["[task_n1] Prior step:\nSome prior output here"]
    prompt = _build_node_prompt(task, node, prior_context=prior)
    assert "Some prior output here" in prompt


def test_build_node_prompt_no_prior_context_section_when_empty():
    task = Task(goal="No prior")
    node = ScienceTaskNode(id="n1", objective="First step")
    prompt = _build_node_prompt(task, node, prior_context=[])
    assert "Previous step outputs" not in prompt


# ---------------------------------------------------------------------------
# Context accumulation: prior output from node 1 in node 2 prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prior_context_from_node1_in_node2_prompt():
    task = Task(goal="Context accumulation test")
    graph = _make_two_node_graph(task.task_id)

    captured_prompts: list[str] = []

    async def capture_prompt(content, **kwargs):
        captured_prompts.append(content)
        return "output for node"

    runtime = MagicMock()
    runtime.process_direct = capture_prompt
    runtime.add_deep_profile_section = MagicMock()

    orchestrator = SequentialOrchestrator(runtime=runtime)
    await orchestrator.run(task, graph)

    assert len(captured_prompts) == 2
    # The second prompt should contain context from node 1
    assert "output for node" in captured_prompts[1]


@pytest.mark.asyncio
async def test_dag_fetch_node_batches_urls_and_stops_when_enough_information():
    task = Task(goal="Fetch enough information")
    node = _make_fetch_node(task.task_id)

    runtime = MagicMock()
    runtime.add_deep_profile_section = MagicMock()
    runtime.supports_async_tool_execute = True

    orchestrator = DAGOrchestrator(runtime=runtime)

    urls = [f"https://example.com/paper-{i}" for i in range(12)]
    outputs = {f"{task.task_id}_search": "\n".join(urls)}

    runtime.tool_execute = AsyncMock(side_effect=[f"content-{i}" for i in range(12)])

    runtime.process_direct = AsyncMock(
        side_effect=[
            "Round 1 summary\nDecision: continue_research",
            "Round 2 summary\nDecision: enough_information",
        ]
    )

    exec_obj, output = await orchestrator._execute_with_retry(task, node, outputs, None)

    assert exec_obj.state == ExecutionState.WAITING_VALIDATION
    assert runtime.tool_execute.await_count == 12
    assert runtime.process_direct.await_count == 2
    assert "Stop reason: enough_information" in output
    assert "Round 2 summary" in output


@pytest.mark.asyncio
async def test_dag_fetch_node_caps_at_five_rounds_of_ten():
    task = Task(goal="Cap fetch depth")
    node = _make_fetch_node(task.task_id)

    runtime = MagicMock()
    runtime.add_deep_profile_section = MagicMock()
    runtime.supports_async_tool_execute = True

    orchestrator = DAGOrchestrator(runtime=runtime)

    urls = [f"https://example.com/paper-{i}" for i in range(80)]
    outputs = {f"{task.task_id}_search": "\n".join(urls)}

    runtime.tool_execute = AsyncMock(side_effect=[f"content-{i}" for i in range(50)])
    runtime.process_direct = AsyncMock(
        side_effect=["Round summary\nDecision: continue_research"] * 5
    )

    exec_obj, output = await orchestrator._execute_with_retry(task, node, outputs, None)

    assert exec_obj.state == ExecutionState.WAITING_VALIDATION
    assert runtime.tool_execute.await_count == 50
    assert runtime.process_direct.await_count == 5
    assert "Stop reason: max_rounds" in output
