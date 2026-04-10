"""Core data models for the AI4S science agent platform."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Lifecycle states for a science task."""

    CREATED = "created"
    CLARIFYING = "clarifying"
    PLANNED = "planned"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionState(str, Enum):
    """Lifecycle states for a single node execution."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING_VALIDATION = "waiting_validation"
    VALIDATED = "validated"
    FAILED = "failed"
    BLOCKED = "blocked"
    REPLANNED = "replanned"
    CANCELLED = "cancelled"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"


class NextAction(str, Enum):
    DELIVER = "deliver"
    RETRY = "retry"
    SUBSTITUTE = "substitute"
    REPLAN = "replan"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Budget(BaseModel):
    """Execution budget constraints."""

    max_tokens: int = 50_000
    max_seconds: int = 600
    max_tool_calls: int = 100


class Constraints(BaseModel):
    """Task-level constraints."""

    resources: dict[str, Any] = Field(default_factory=dict)
    time_limit: int | None = None
    forbidden_tools: list[str] = Field(default_factory=list)


class DeliverableSpec(BaseModel):
    """Expected output specification."""

    expected_format: str = "markdown"
    required_sections: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)


class TaskContext(BaseModel):
    """Input context for a science task."""

    prior_knowledge: list[str] = Field(default_factory=list)
    input_materials: list[str] = Field(default_factory=list)
    session_id: str | None = None


class RetryPolicy(BaseModel):
    """Node-level retry configuration."""

    max_retries: int = 2
    backoff_seconds: float = 1.0
    fallback_node_id: str | None = None


class LogEntry(BaseModel):
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    level: str = "INFO"
    msg: str


class ExecutionMetrics(BaseModel):
    tokens_used: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0


class Evidence(BaseModel):
    """Evidence produced during a node execution."""

    sources: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    validation_log: list[str] = Field(default_factory=list)


class Violation(BaseModel):
    rule: str
    msg: str
    severity: str = "error"


class Provenance(BaseModel):
    """Lineage record for a delivered result."""

    entity_id: str
    generated_by: str
    agent_id: str = "science"
    used_entities: list[str] = Field(default_factory=list)
    toolchain: dict[str, Any] = Field(default_factory=dict)
    time_range: dict[str, str] = Field(default_factory=dict)
    hashes: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Primary domain objects
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """Structured representation of a science task."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    goal: str
    scope: list[str] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    context: TaskContext = Field(default_factory=TaskContext)
    deliverables: DeliverableSpec = Field(default_factory=DeliverableSpec)
    budget: Budget = Field(default_factory=Budget)
    priority: Priority = Priority.NORMAL
    status: TaskStatus = TaskStatus.CREATED
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:16]}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScienceTaskNode(BaseModel):
    """A single step in a science task graph."""

    id: str
    objective: str
    dependencies: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    assigned_role: str = "executor"
    candidate_capabilities: list[str] = Field(default_factory=list)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class ScienceTaskGraph(BaseModel):
    """A directed graph of science task nodes."""

    task_id: str
    nodes: list[ScienceTaskNode] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def topological_order(self) -> list[ScienceTaskNode]:
        """Return nodes in topological execution order."""
        index = {n.id: n for n in self.nodes}
        visited: set[str] = set()
        order: list[ScienceTaskNode] = []

        def _visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            node = index[node_id]
            for dep in node.dependencies:
                if dep in index:
                    _visit(dep)
            order.append(node)

        for node in self.nodes:
            _visit(node.id)
        return order


class Execution(BaseModel):
    """Runtime instance of a single node execution."""

    execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    task_id: str
    node_id: str
    capability_id: str = "llm_agent"
    state: ExecutionState = ExecutionState.PENDING
    logs: list[LogEntry] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    evidence: Evidence | None = None
    error: str | None = None
    output: str | None = None


class Validation(BaseModel):
    """Result of validating a node or task output."""

    criteria: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: ValidationStatus = ValidationStatus.PASSED
    confidence: float = 1.0
    violations: list[Violation] = Field(default_factory=list)
    next_action: NextAction = NextAction.DELIVER
