"""Tests for CapabilityRegistry in aeloon/plugins/science/capability.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from aeloon.plugins.ScienceResearch.capability import (
    CapabilityLevel,
    CapabilityMetadata,
    CapabilityRegistry,
    CapabilityType,
    get_default_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cap(
    cap_id: str,
    name: str = "Test Cap",
    cap_type: CapabilityType = CapabilityType.TOOL,
    level: CapabilityLevel = CapabilityLevel.L1,
    enabled: bool = True,
) -> CapabilityMetadata:
    return CapabilityMetadata(id=cap_id, name=name, type=cap_type, level=level, enabled=enabled)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_registry_has_builtin_capabilities() -> None:
    registry = get_default_registry()
    assert len(registry.list_all()) >= 5


def test_register_and_get_capability() -> None:
    registry = CapabilityRegistry()
    cap = _make_cap("custom.my_cap", name="My Custom Cap")
    registry.register(cap)
    retrieved = registry.get("custom.my_cap")
    assert retrieved is not None
    assert retrieved.id == "custom.my_cap"
    assert retrieved.name == "My Custom Cap"


def test_list_by_type_returns_only_tools() -> None:
    registry = CapabilityRegistry()
    registry.register(_make_cap("tool.a", cap_type=CapabilityType.TOOL))
    registry.register(_make_cap("workflow.a", cap_type=CapabilityType.WORKFLOW))
    registry.register(_make_cap("model.a", cap_type=CapabilityType.MODEL))

    tools = registry.list_by_type(CapabilityType.TOOL)
    assert len(tools) >= 1
    assert all(cap.type == CapabilityType.TOOL for cap in tools)


def test_list_by_level_returns_only_l1() -> None:
    registry = CapabilityRegistry()
    registry.register(_make_cap("l1.a", level=CapabilityLevel.L1))
    registry.register(_make_cap("l2.a", level=CapabilityLevel.L2))
    registry.register(_make_cap("l3.a", level=CapabilityLevel.L3))

    l1_caps = registry.list_by_level(CapabilityLevel.L1)
    assert len(l1_caps) >= 1
    assert all(cap.level == CapabilityLevel.L1 for cap in l1_caps)


def test_search_finds_by_name_substring() -> None:
    registry = CapabilityRegistry()
    registry.register(_make_cap("search.web", name="web_search"))
    registry.register(_make_cap("other.cap", name="write_file"))

    results = registry.search("search")
    names = [cap.name for cap in results]
    assert any("search" in name for name in names)


def test_disabled_capability_not_in_list_all() -> None:
    registry = CapabilityRegistry()
    registry.register(_make_cap("disabled.cap", name="Disabled Cap", enabled=False))
    ids = [cap.id for cap in registry.list_all()]
    assert "disabled.cap" not in ids


def test_get_returns_none_for_unknown_id() -> None:
    registry = CapabilityRegistry()
    result = registry.get("nonexistent")
    assert result is None


def test_populate_from_tool_registry_adds_tools() -> None:
    registry = CapabilityRegistry()
    initial_count = len(registry.list_all())

    tool_a = MagicMock()
    tool_a.name = "tool_alpha"
    tool_a.description = "Alpha tool description"

    tool_b = MagicMock()
    tool_b.name = "tool_beta"
    tool_b.description = "Beta tool description"

    mock_tool_registry = MagicMock()
    mock_tool_registry.list.return_value = [tool_a, tool_b]

    registry.populate_from_tool_registry(mock_tool_registry)

    assert len(registry.list_all()) == initial_count + 2
    ids = [cap.id for cap in registry.list_all()]
    assert "aeloon.tool_alpha" in ids
    assert "aeloon.tool_beta" in ids
