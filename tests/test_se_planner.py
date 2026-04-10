"""Tests for SE task planners."""

from __future__ import annotations

from aeloon.plugins.SoftwareEngineering.planner import DAGSEPlanner, LinearSEPlanner
from aeloon.plugins.SoftwareEngineering.task import (
    ArchitectureGraph,
    ModuleDef,
    NodeType,
    Project,
    SETaskGraph,
)


class TestLinearSEPlanner:
    def test_plan_returns_five_nodes(self) -> None:
        planner = LinearSEPlanner()
        project = Project(description="create a CSV validator")
        graph = planner.plan(project)

        assert isinstance(graph, SETaskGraph)
        assert graph.project_id == project.project_id
        assert len(graph.nodes) == 5

    def test_plan_node_types(self) -> None:
        planner = LinearSEPlanner()
        project = Project(description="test project")
        graph = planner.plan(project)

        types = [n.node_type for n in graph.nodes]
        assert types[0] == NodeType.SCAFFOLD
        assert types[1] == NodeType.IMPLEMENT
        assert types[2] == NodeType.TEST
        assert types[3] == NodeType.VALIDATE
        assert types[4] == NodeType.DELIVER

    def test_plan_linear_dependencies(self) -> None:
        planner = LinearSEPlanner()
        project = Project(description="test")
        graph = planner.plan(project)

        ids = [n.id for n in graph.nodes]
        assert graph.nodes[0].dependencies == []
        assert ids[0] in graph.nodes[1].dependencies
        assert ids[1] in graph.nodes[2].dependencies
        assert ids[2] in graph.nodes[3].dependencies
        assert ids[3] in graph.nodes[4].dependencies

    def test_plan_topological_order_valid(self) -> None:
        planner = LinearSEPlanner()
        project = Project(description="test")
        graph = planner.plan(project)
        order = graph.topological_order()
        assert len(order) == 5


class TestDAGSEPlanner:
    def test_falls_back_to_linear_without_architecture(self) -> None:
        planner = DAGSEPlanner()
        project = Project(description="single module")
        graph = planner.plan(project)
        assert isinstance(graph, SETaskGraph)
        assert len(graph.nodes) == 5
        planner = DAGSEPlanner()
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
        graph = planner.plan(project)
        # 2 modules * 2 nodes (impl+test) + 1 integrate + 1 validate + 1 deliver = 7
        assert len(graph.nodes) == 7
        impl_nodes = [n for n in graph.nodes if n.node_type == NodeType.IMPLEMENT]
        assert len(impl_nodes) == 2
        assert impl_nodes[0].dependencies == []
        assert impl_nodes[1].dependencies == []

    def test_plan_with_dependencies(self) -> None:
        planner = DAGSEPlanner()
        project = Project(
            description="dependent modules",
            architecture=ArchitectureGraph(
                project_id="p1",
                modules=[
                    ModuleDef(id="core", name="core", dependencies=[]),
                    ModuleDef(id="api", name="api", dependencies=["core"]),
                ],
            ),
        )
        graph = planner.plan(project)
        assert len(graph.nodes) == 7
        api_impl = [n for n in graph.nodes if "api" in n.id and n.node_type == NodeType.IMPLEMENT]
        assert any("core" in dep for dep in api_impl[0].dependencies)

    def test_integrate_depends_on_all_tests(self) -> None:
        planner = DAGSEPlanner()
        project = Project(
            description="multi",
            architecture=ArchitectureGraph(
                project_id="p1",
                modules=[
                    ModuleDef(id="a", name="a", dependencies=[]),
                    ModuleDef(id="b", name="b", dependencies=[]),
                ],
            ),
        )
        graph = planner.plan(project)
        integrate_node = [n for n in graph.nodes if n.node_type == NodeType.INTEGRATE]
        assert len(integrate_node) == 1
        test_ids = [n.id for n in graph.nodes if n.node_type == NodeType.TEST]
        for tid in test_ids:
            assert tid in integrate_node[0].dependencies
