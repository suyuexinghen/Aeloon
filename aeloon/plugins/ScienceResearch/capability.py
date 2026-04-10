"""Capability model and registry for the AI4S science platform.

Provides a typed, queryable catalog of all capabilities available to the
science planner and orchestrator.  L1 tool capabilities are auto-populated
from the Aeloon ToolRegistry.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class CapabilityType(str, Enum):
    """Category of a capability."""

    TOOL = "tool"
    WORKFLOW = "workflow"
    MODEL = "model"
    AGENT = "agent"


class CapabilityLevel(str, Enum):
    """Abstraction level of a capability."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class CapabilityMetadata(BaseModel):
    """Metadata describing a single capability in the registry."""

    id: str
    name: str
    type: CapabilityType = CapabilityType.TOOL
    level: CapabilityLevel = CapabilityLevel.L1
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    side_effects: list[str] = Field(default_factory=list)
    cost_estimate: str = "low"
    latency_estimate: str = "low"
    reliability: float = 0.95
    validator_hooks: list[str] = Field(default_factory=list)
    enabled: bool = True


class CapabilityRegistry:
    """Queryable catalog of all capabilities available to the science platform."""

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityMetadata] = {}

    def register(self, cap: CapabilityMetadata) -> None:
        """Register a capability, overwriting any existing entry with the same id."""
        self._capabilities[cap.id] = cap
        logger.debug("Registered capability: {}", cap.id)

    def get(self, cap_id: str) -> CapabilityMetadata | None:
        """Return the capability with the given id, or None if not found."""
        return self._capabilities.get(cap_id)

    def list_all(self) -> list[CapabilityMetadata]:
        """Return all enabled capabilities."""
        return [cap for cap in self._capabilities.values() if cap.enabled]

    def list_by_type(self, cap_type: CapabilityType) -> list[CapabilityMetadata]:
        """Return all enabled capabilities of the given type."""
        return [cap for cap in self.list_all() if cap.type == cap_type]

    def list_by_level(self, level: CapabilityLevel) -> list[CapabilityMetadata]:
        """Return all enabled capabilities at the given level."""
        return [cap for cap in self.list_all() if cap.level == level]

    def search(self, query: str) -> list[CapabilityMetadata]:
        """Case-insensitive search across name and description of enabled capabilities."""
        lower = query.lower()
        return [
            cap
            for cap in self.list_all()
            if lower in cap.name.lower() or lower in cap.description.lower()
        ]

    def populate_from_tool_registry(self, tool_registry: Any) -> None:
        """Auto-register L1 TOOL capabilities from an Aeloon ToolRegistry.

        Tries ``tool_registry.list()`` first, then falls back to
        ``tool_registry._tools`` (dict or list).  Missing attributes are handled
        gracefully.
        """
        tools: list[Any] = []

        # Prefer the public list() API
        try:
            tools = list(tool_registry.list())
        except AttributeError:
            pass

        # Fall back to private _tools attribute
        if not tools:
            try:
                raw = tool_registry._tools
                tools = list(raw.values()) if isinstance(raw, dict) else list(raw)
            except AttributeError:
                logger.warning("ToolRegistry has no list() or _tools; skipping auto-populate")
                return

        for tool in tools:
            try:
                tool_name = getattr(tool, "name", str(tool))
                tool_description = getattr(tool, "description", "")
                cap_id = f"aeloon.{tool_name}"
                self.register(
                    CapabilityMetadata(
                        id=cap_id,
                        name=tool_name,
                        type=CapabilityType.TOOL,
                        level=CapabilityLevel.L1,
                        description=tool_description,
                    )
                )
            except AttributeError as exc:
                logger.warning("Skipping tool during auto-populate: {}", exc)


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY = CapabilityRegistry()


def get_default_registry() -> CapabilityRegistry:
    """Return the module-level default CapabilityRegistry."""
    return _DEFAULT_REGISTRY


def _register_defaults() -> None:
    """Pre-populate the default registry with known L1 capabilities."""
    defaults: list[CapabilityMetadata] = [
        CapabilityMetadata(
            id="aeloon.web_search",
            name="Web Search",
            type=CapabilityType.TOOL,
            level=CapabilityLevel.L1,
            side_effects=["network"],
            cost_estimate="low",
        ),
        CapabilityMetadata(
            id="aeloon.web_fetch",
            name="Web Fetch",
            type=CapabilityType.TOOL,
            level=CapabilityLevel.L1,
            side_effects=["network"],
            cost_estimate="low",
        ),
        CapabilityMetadata(
            id="aeloon.exec",
            name="Shell Execute",
            type=CapabilityType.TOOL,
            level=CapabilityLevel.L1,
            side_effects=["process", "file_write"],
            cost_estimate="medium",
            reliability=0.85,
        ),
        CapabilityMetadata(
            id="aeloon.read_file",
            name="Read File",
            type=CapabilityType.TOOL,
            level=CapabilityLevel.L1,
            side_effects=[],
            cost_estimate="low",
        ),
        CapabilityMetadata(
            id="aeloon.write_file",
            name="Write File",
            type=CapabilityType.TOOL,
            level=CapabilityLevel.L1,
            side_effects=["file_write"],
            cost_estimate="low",
        ),
        CapabilityMetadata(
            id="aeloon.llm_analysis",
            name="LLM Analysis",
            type=CapabilityType.TOOL,
            level=CapabilityLevel.L1,
            description="Direct LLM reasoning without tool calls",
            side_effects=[],
            cost_estimate="medium",
        ),
    ]
    for cap in defaults:
        _DEFAULT_REGISTRY.register(cap)


_register_defaults()
