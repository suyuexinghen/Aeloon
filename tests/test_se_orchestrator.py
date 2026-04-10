"""Tests for SE task orchestrators."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aeloon.plugins.SoftwareEngineering.orchestrator import (
    BudgetExceededError,
    DAGSEOrchestrator,
    SequentialSEOrchestrator,
)
from aeloon.plugins.SoftwareEngineering.planner import DAGSEPlanner, LinearSEPlanner
from aeloon.plugins.SoftwareEngineering.task import (
    ArchitectureGraph,
    ExecutionState,
    ModuleDef,
    Project,
    SEBudget,
)

from aeloon.plugins._sdk.runtime import PluginRuntime


@pytest.fixture
def mock_agent_loop() -> MagicMock:
    loop = MagicMock()
    loop.provider = MagicMock()
    loop.provider.chat = AsyncMock(return_value=MagicMock(content="test response"))
    loop.model = "test-model"
    loop.process_direct = AsyncMock(return_value="orchestrated output")
    loop.profiler = MagicMock(enabled=False)
    loop.tools = MagicMock()
    loop.tools.execute = AsyncMock(return_value="tool result")
    return loop


@pytest.fixture
def runtime(mock_agent_loop: MagicMock) -> PluginRuntime:
    return PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.se",
        config={},
        storage_base=Path("/tmp"),
    )


class TestSequentialSEOrchestrator:
    @pytest.mark.asyncio
    async def test_run_linear_order(self, runtime: PluginRuntime) -> None:
        planner = LinearSEPlanner()
        project = Project(description="test project")
        graph = planner.plan(project)
        orchestrator = SequentialSEOrchestrator(runtime=runtime)

        executions = await orchestrator.run(project, graph)

        assert len(executions) == 5
        for ex in executions:
            assert ex.state in (
                ExecutionState.WAITING_VALIDATION,
                ExecutionState.VALIDATED,
            )

    @pytest.mark.asyncio
    async def test_stops_on_failure(self, runtime: PluginRuntime) -> None:
        """Failed node should break the pipeline."""
        planner = LinearSEPlanner()
        project = Project(description="test")
        graph = planner.plan(project)
        orchestrator = SequentialSEOrchestrator(runtime=runtime)

        # Make the first call fail
        async def mock_fail(content: str, **kwargs):
            raise RuntimeError("Node failed")

        runtime._agent_loop.process_direct = mock_fail
        executions = await orchestrator.run(project, graph)
        assert len(executions) == 1
        assert executions[0].state == ExecutionState.FAILED

    @pytest.mark.asyncio
    async def test_progress_callback(self, runtime: PluginRuntime) -> None:
        planner = LinearSEPlanner()
        project = Project(description="test")
        graph = planner.plan(project)
        orchestrator = SequentialSEOrchestrator(runtime=runtime)

        progress = AsyncMock()
        _executions = await orchestrator.run(project, graph, on_progress=progress)
        assert progress.call_count >= 4  # One per node


class TestDAGSEOrchestrator:
    @pytest.mark.asyncio
    async def test_parallel_execution(self, runtime: PluginRuntime) -> None:
        """Two independent modules should run in parallel."""
        project = Project(
            description="multi-module project",
            architecture=ArchitectureGraph(
                project_id="p1",
                modules=[
                    ModuleDef(id="mod_a", name="module_a", dependencies=[]),
                    ModuleDef(id="mod_b", name="module_b", dependencies=[]),
                ],
            ),
        )
        planner = DAGSEPlanner()
        graph = planner.plan(project)
        orchestrator = DAGSEOrchestrator(runtime=runtime, max_concurrency=5)

        executions = await orchestrator.run(project, graph)
        # 2 impl + 2 test + 1 integrate + 1 validate + 1 deliver = 7
        assert len(executions) == 7
        impl_execs = [e for e in executions if "implement" in e.node_id]
        assert len(impl_execs) == 2
        assert all(e.state == ExecutionState.WAITING_VALIDATION for e in impl_execs)

    @pytest.mark.asyncio
    async def test_budget_exceeded(self, runtime: PluginRuntime) -> None:
        """Should raise BudgetExceededError when time budget is exceeded."""
        project = Project(
            description="test",
            budget=SEBudget(max_seconds=0),  # Immediate timeout
        )
        planner = LinearSEPlanner()
        graph = planner.plan(project)
        orchestrator = DAGSEOrchestrator(runtime=runtime, max_concurrency=5)

        with pytest.raises(BudgetExceededError):
            await orchestrator.run(project, graph)
