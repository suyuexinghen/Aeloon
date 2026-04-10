"""Tests for PluginCreator DAG utilities — topological sort and dependency graph."""

from __future__ import annotations

import pytest

from aeloon.plugins.PluginCreator.models import (
    PlanItem,
    PlanItemKind,
    build_dependency_graph,
    topological_sort,
)


def _item(item_id: str, depends_on: list[str] | None = None) -> PlanItem:
    return PlanItem(
        id=item_id, title=f"Item {item_id}", kind=PlanItemKind.GENERATE, depends_on=depends_on or []
    )


class TestBuildDependencyGraph:
    def test_empty(self) -> None:
        assert build_dependency_graph([]) == {}

    def test_linear(self) -> None:
        items = [_item("a"), _item("b", ["a"]), _item("c", ["b"])]
        graph = build_dependency_graph(items)
        assert graph == {"a": [], "b": ["a"], "c": ["b"]}


class TestTopologicalSort:
    def test_empty_list(self) -> None:
        assert topological_sort([]) == []

    def test_single_item(self) -> None:
        items = [_item("a")]
        result = topological_sort(items)
        assert [i.id for i in result] == ["a"]

    def test_linear_chain(self) -> None:
        items = [_item("a"), _item("b", ["a"]), _item("c", ["b"])]
        result = topological_sort(items)
        ids = [i.id for i in result]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_parallel_branches(self) -> None:
        items = [
            _item("a"),
            _item("b", ["a"]),
            _item("c", ["a"]),
            _item("d", ["b", "c"]),
        ]
        result = topological_sort(items)
        ids = [i.id for i in result]
        assert len(result) == 4
        assert len(set(ids)) == 4  # no duplicates
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_no_dependencies(self) -> None:
        items = [_item("a"), _item("b"), _item("c")]
        result = topological_sort(items)
        assert len(result) == 3
        assert {i.id for i in result} == {"a", "b", "c"}

    def test_cycle_raises(self) -> None:
        items = [_item("a", ["b"]), _item("b", ["a"])]
        with pytest.raises(ValueError, match="[Cc]ycle"):
            topological_sort(items)

    def test_self_reference_raises(self) -> None:
        items = [_item("a", ["a"])]
        with pytest.raises(ValueError, match="[Cc]ycle"):
            topological_sort(items)

    def test_longer_cycle(self) -> None:
        items = [_item("a", ["c"]), _item("b", ["a"]), _item("c", ["b"])]
        with pytest.raises(ValueError, match="[Cc]ycle"):
            topological_sort(items)
