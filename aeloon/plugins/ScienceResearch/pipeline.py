"""SciencePipeline: top-level coordinator for AI4S task execution.

Handles the full control path:
  intent -> (clarify?) -> task -> plan -> orchestrate -> validate -> deliver
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from aeloon.plugins._sdk.runtime import PluginRuntime

from .orchestrator import BudgetExceededError, DAGOrchestrator
from .planner import DAGPlanner
from .storage.jsonl import JsonlStorage
from .task import (
    Budget,
    Constraints,
    DeliverableSpec,
    Execution,
    ExecutionState,
    NextAction,
    Task,
    TaskContext,
    TaskStatus,
    Validation,
)
from .validator import make_default_validator

_DEFAULT_REQUIRED_SECTIONS = ["Summary", "Key Findings", "Sources"]

# Queries shorter than this trigger a clarification hint (not a block)
_MIN_GOAL_WORDS = 4


class SciencePipeline:
    """Coordinates the full AI4S execution pipeline for one task."""

    def __init__(
        self,
        runtime: PluginRuntime,
        storage_dir: str | Path | None = None,
    ) -> None:
        self._runtime = runtime
        self._planner = DAGPlanner()
        self._orchestrator = DAGOrchestrator(runtime=runtime)
        root = Path(storage_dir).expanduser() if storage_dir else runtime.storage_path
        self._storage = JsonlStorage(root)
        self._validator = make_default_validator()

        # In-memory state for /science status queries
        self._current_task: Task | None = None
        self._current_executions: list[Execution] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session_id: str | None = None,
    ) -> tuple[str, Task]:
        """Execute a full science pipeline for the given natural-language query.

        Returns (formatted_output, task).
        """
        # 0. Clarification hint for very short / vague queries
        clarification = _check_clarification(query)
        if clarification and on_progress:
            await on_progress(clarification)
        if clarification:
            self._runtime.add_deep_profile_section(
                "Science · Clarification",
                [clarification],
            )

        # 1. Interpret intent -> Task
        if on_progress:
            await on_progress("Interpreting task...")
        task = self._interpret(query, session_id)
        self._runtime.add_deep_profile_section(
            "Science · Interpret",
            [
                f"Task ID: {task.task_id}",
                f"Trace ID: {task.trace_id}",
                f"Goal: {task.goal}",
                f"Scope items: {len(task.scope)}",
            ],
        )
        self._current_task = task
        self._current_executions = []
        task.status = TaskStatus.PLANNED
        self._storage.save_task(task)
        logger.info(
            "Science task {} created: goal={} scope_items={}",
            task.task_id,
            task.goal[:80],
            len(task.scope),
        )

        # 2. Plan -> ScienceTaskGraph
        if on_progress:
            await on_progress("Generating execution plan...")
        graph = self._planner.plan(task)
        node_count = len(graph.nodes)
        self._runtime.add_deep_profile_section(
            "Science · Plan",
            [
                f"Nodes: {node_count}",
                *[
                    f"- {node.id}: deps={len(node.dependencies)} role={node.assigned_role} objective={node.objective[:80]}"
                    for node in graph.nodes
                ],
            ],
        )
        logger.info(
            "Planned {} node(s) for task {}: nodes={}",
            node_count,
            task.task_id,
            [n.id for n in graph.nodes],
        )

        # 3. Execute nodes
        task.status = TaskStatus.RUNNING
        task.updated_at = datetime.now(UTC)
        self._storage.save_task(task)

        try:
            executions = await self._orchestrator.run(task, graph, on_progress)
        except BudgetExceededError as exc:
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now(UTC)
            self._storage.save_task(task)
            logger.warning("Science task {} exceeded budget: {}", task.task_id, exc)
            return f"Error: Task stopped — budget exceeded. {exc}", task
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now(UTC)
            self._storage.save_task(task)
            logger.error("Science task {} failed during orchestration: {}", task.task_id, exc)
            return f"Error: Science task failed — {exc}", task

        self._current_executions = executions
        for ex in executions:
            self._storage.save_execution(ex)

        self._runtime.add_deep_profile_section(
            "Science · Execute",
            [
                f"Executions: {len(executions)}",
                *[
                    f"- {ex.node_id}: state={ex.state.value} elapsed={ex.metrics.elapsed_seconds:.2f}s tokens={ex.metrics.tokens_used} tools={ex.metrics.tool_calls}"
                    for ex in executions
                ],
            ],
        )
        logger.info(
            "Science task {} execution complete: {} nodes, {} succeeded, {} failed, "
            "total_elapsed={:.2f}s",
            task.task_id,
            len(executions),
            sum(1 for e in executions if e.state == ExecutionState.WAITING_VALIDATION),
            sum(1 for e in executions if e.state == ExecutionState.FAILED),
            sum(e.metrics.elapsed_seconds for e in executions),
        )
        for ex in executions:
            logger.debug(
                "Science task {} node {}: state={} elapsed={:.2f}s tokens={} "
                "tools={} output_chars={}",
                task.task_id,
                ex.node_id,
                ex.state.value,
                ex.metrics.elapsed_seconds,
                ex.metrics.tokens_used,
                ex.metrics.tool_calls,
                len(ex.output or ""),
            )

        # Check if any node failed
        failed = [e for e in executions if e.state == ExecutionState.FAILED]
        if failed:
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now(UTC)
            self._storage.save_task(task)
            errors = "; ".join(e.error or "unknown" for e in failed)
            return f"Error: Science task failed — {errors}", task

        # 4. Validate final node output
        task.status = TaskStatus.VALIDATING
        self._storage.save_task(task)

        last_exec = _last_successful_execution(executions)
        if last_exec is None:
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now(UTC)
            self._storage.save_task(task)
            return "Error: No output was produced by the science pipeline.", task

        validation = self._validator.validate(
            last_exec,
            task.deliverables,
            task_goal=task.goal,
        )
        self._runtime.add_deep_profile_section(
            "Science · Validate",
            [
                f"Status: {validation.status.value}",
                f"Next action: {validation.next_action.value}",
                f"Violations: {len(validation.violations)}",
                *[f"- {violation.rule}: {violation.msg}" for violation in validation.violations],
            ],
        )

        if validation.next_action in (NextAction.DELIVER,) or validation.status.value != "failed":
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED

        task.updated_at = datetime.now(UTC)
        self._storage.save_task(task)

        # 5. Format and deliver
        output = _format_output(task, executions, validation)
        self._runtime.add_deep_profile_section(
            "Science · Deliver",
            [
                f"Task status: {task.status.value}",
                f"Output length: {len(output)}",
                f"Last execution node: {last_exec.node_id}",
            ],
        )
        logger.info(
            "Science task {} completed: status={} validation={} violations={} output_chars={}",
            task.task_id,
            task.status,
            validation.status,
            len(validation.violations),
            len(output),
        )
        return output, task

    def get_status(self) -> str:
        """Return a human-readable status for the current/last task."""
        task = self._current_task
        if task is None:
            return "No science task has been run in this session."

        lines = [
            f"**Task:** `{task.task_id}`",
            f"**Goal:** {task.goal}",
            f"**Status:** {task.status.value}",
            f"**Created:** {task.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ]

        if self._current_executions:
            lines.append(f"\n**Steps ({len(self._current_executions)}):**")
            total_elapsed = 0.0
            total_tokens = 0
            for ex in self._current_executions:
                icon = {
                    "validated": "✓",
                    "waiting_validation": "✓",
                    "failed": "✗",
                    "cancelled": "⊘",
                    "running": "⟳",
                }.get(ex.state.value, "•")
                step_name = ex.node_id.rsplit("_", 1)[-1]
                elapsed = ex.metrics.elapsed_seconds if ex.metrics else 0
                total_elapsed += elapsed
                total_tokens += ex.metrics.tokens_used if ex.metrics else 0
                lines.append(f"  {icon} `{step_name}` — {ex.state.value} ({elapsed:.1f}s)")

            lines.append(f"\n**Budget used:** {total_elapsed:.0f}s elapsed, ~{total_tokens} tokens")
            lines.append(
                f"**Budget limit:** {task.budget.max_seconds}s, {task.budget.max_tokens} tokens"
            )

        return "\n".join(lines)

    def get_history(self) -> str:
        """Return a formatted list of past science tasks."""
        tasks = self._storage.list_tasks()
        if not tasks:
            return "No science tasks found in storage."

        lines = ["**Recent science tasks:**\n"]
        for t in tasks[:20]:
            icon = {"completed": "✓", "failed": "✗", "running": "⟳"}.get(t["status"], "•")
            lines.append(
                f"  {icon} `{t['task_id']}` — {t['goal']} ({t['status']}, {t['created_at'][:10]})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpret(self, query: str, session_id: str | None = None) -> Task:
        """Create a structured Task from the user's natural-language query."""
        return Task(
            goal=query.strip(),
            scope=[],
            constraints=Constraints(),
            context=TaskContext(session_id=session_id),
            deliverables=DeliverableSpec(
                expected_format="markdown",
                required_sections=_DEFAULT_REQUIRED_SECTIONS,
            ),
            budget=Budget(),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _check_clarification(query: str) -> str | None:
    """Return a clarification hint if the query seems too vague; else None."""
    words = [w for w in query.strip().split() if w]
    if len(words) < _MIN_GOAL_WORDS:
        return (
            f"Note: Your query is very short ({len(words)} word(s)). "
            "For best results, include the topic, scope, and what you want to know. "
            "Proceeding with the current query..."
        )
    return None


def _last_successful_execution(executions: list[Execution]) -> Execution | None:
    """Return the last execution that produced output."""
    for ex in reversed(executions):
        if ex.output:
            return ex
    return None


def _format_output(
    task: Task,
    executions: list[Execution],
    validation: Validation,
) -> str:
    last_output = ""
    for ex in reversed(executions):
        if ex.output:
            last_output = ex.output
            break

    if not last_output:
        return "Error: No output was produced by the science pipeline."

    parts = [last_output, "\n\n---"]
    parts.append(
        f"*Science task `{task.task_id}` · {len(executions)} step(s) · "
        f"validation: {validation.status.value}*"
    )

    error_violations = [v for v in validation.violations if v.severity == "error"]
    warn_violations = [v for v in validation.violations if v.severity == "warning"]
    if error_violations:
        parts.append(f"*Validation issues: {'; '.join(v.msg for v in error_violations)}*")
    elif warn_violations:
        parts.append(f"*Note: {'; '.join(v.msg for v in warn_violations)}*")

    return "\n".join(parts)


_HELP_TEXT = """\
## AI4S Science Agent

Run a structured scientific research task using AI.

**Usage:**
- `/science <your research question or task>`
- `/science status` — Show progress and budget of the current task
- `/science history` — List past science tasks
- `/science help` — Show this help

**Examples:**
```
/science search for recent papers on perovskite solar cell efficiency
/science summarize the state of high-entropy alloy research in catalysis
/science what are the latest developments in protein structure prediction?
```

**Tips:**
- Be specific: include the topic, desired scope, and output format
- For multi-topic tasks, list scope items to enable parallel research
- Results include a summary, key findings, and cited sources
"""


def get_help_text() -> str:
    return _HELP_TEXT
