"""Phase and plan-item models — the structural backbone of a PlanPackage.

:class:`PhaseContract` defines phase boundaries; :class:`PlanItem` is the
executable work unit.  Items form a DAG via ``depends_on``.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from .governance import PlanningStatus


class PhaseContract(BaseModel):
    """Contract defining a single workflow phase boundary."""

    phase_id: str
    phase_name: str
    goal: str
    status: PlanningStatus = PlanningStatus.PLANNED
    inputs: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    entry_conditions: list[str] = Field(default_factory=list)
    exit_conditions: list[str] = Field(default_factory=list)
    verification_gate_ids: list[str] = Field(default_factory=list)
    next_phase_trigger: str = ""
    depends_on_phase_ids: list[str] = Field(default_factory=list)
    deferred_item_ids: list[str] = Field(default_factory=list)
    decision_point_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    phase_summary: str | None = None
    failure_handoff: str | None = None


class PlanItemKind(str, Enum):
    """Classification of plan items."""

    ANALYSIS = "analysis"
    DESIGN = "design"
    GENERATE = "generate"
    VALIDATE = "validate"
    REPAIR = "repair"
    REPORT = "report"
    REVIEW = "review"
    GATE = "gate"
    RESUME = "resume"


class PlanItem(BaseModel):
    """A single executable work item within a plan."""

    id: str = Field(default_factory=lambda: f"item_{uuid4().hex[:12]}")
    title: str
    kind: PlanItemKind
    status: PlanningStatus = PlanningStatus.PLANNED
    scope: str = ""
    depends_on: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    done_when: list[str] = Field(default_factory=list)
    phase_id: str | None = None
    parallelizable: bool = False
    validation_refs: list[str] = Field(default_factory=list)
    risk_refs: list[str] = Field(default_factory=list)
    decision_point_ids: list[str] = Field(default_factory=list)
    deferable: bool = False
    owner_hint: str | None = None
    priority: str | None = None


# ---------------------------------------------------------------------------
# DAG utilities
# ---------------------------------------------------------------------------


def build_dependency_graph(items: list[PlanItem]) -> dict[str, list[str]]:
    """Return adjacency list: item_id -> list of direct dependency item_ids."""
    return {item.id: list(item.depends_on) for item in items}


def topological_sort(items: list[PlanItem]) -> list[PlanItem]:
    """Sort plan items using Kahn's algorithm.

    Returns items in topological order.  Raises ``ValueError`` if a cycle
    is detected.
    """
    index = {item.id: item for item in items}
    in_degree: dict[str, int] = {item.id: len(item.depends_on) for item in items}

    # Build forward adjacency (dependency -> dependent)
    adj: dict[str, list[str]] = {item.id: [] for item in items}
    for item in items:
        for dep_id in item.depends_on:
            if dep_id in adj:
                adj[dep_id].append(item.id)

    queue = deque(item_id for item_id, deg in in_degree.items() if deg == 0)
    result: list[PlanItem] = []

    while queue:
        item_id = queue.popleft()
        result.append(index[item_id])
        for neighbour in adj[item_id]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(result) != len(items):
        remaining = [iid for iid, deg in in_degree.items() if deg > 0]
        raise ValueError(f"Cycle detected among plan items: {remaining}")

    return result
