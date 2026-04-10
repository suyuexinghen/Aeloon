"""Science task orchestrators: execute a ScienceTaskGraph node by node."""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from loguru import logger

from aeloon.plugins._sdk.runtime import PluginRuntime

from .task import (
    Budget,
    Execution,
    ExecutionMetrics,
    ExecutionState,
    LogEntry,
    ScienceTaskGraph,
    ScienceTaskNode,
    Task,
)


def _filter_kernel_thinking(
    on_progress: Callable[..., Awaitable[None]],
) -> Callable[..., Awaitable[None]]:
    """Wrap *on_progress* to suppress kernel-emitted "Thinking…" messages.

    The science orchestrator already emits its own step-level progress
    ("Step 1/N: …"), so the generic "Thinking…" / "Thinking (step N)…"
    from :func:`run_agent_kernel` is redundant noise.
    """

    async def _filtered(text: str, *, tool_hint: bool = False) -> None:
        if not tool_hint and text.startswith("Thinking"):
            return
        await on_progress(text, tool_hint=tool_hint)

    return _filtered


class BudgetExceededError(Exception):
    """Raised when a task exceeds its time, token, or tool-call budget."""

    def __init__(self, reason: str, budget: Budget) -> None:
        super().__init__(reason)
        self.budget = budget


class Orchestrator(ABC):
    """Abstract base class for science task orchestrators."""

    @abstractmethod
    async def run(
        self,
        task: Task,
        graph: ScienceTaskGraph,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> list[Execution]:
        """Execute all nodes in the graph and return one Execution per node."""
        ...


# ---------------------------------------------------------------------------
# Sequential (walking skeleton) orchestrator
# ---------------------------------------------------------------------------


class SequentialOrchestrator(Orchestrator):
    """Executes task-graph nodes in topological (linear) order.

    Each node is delegated to the Aeloon kernel via AgentLoop.process_direct().
    Outputs from completed nodes are concatenated into the context of the next.
    """

    def __init__(self, runtime: PluginRuntime) -> None:
        self._runtime = runtime

    async def run(
        self,
        task: Task,
        graph: ScienceTaskGraph,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> list[Execution]:
        executions: list[Execution] = []
        accumulated_context: list[str] = []
        ordered_nodes = graph.topological_order()

        for node in ordered_nodes:
            exec_obj = Execution(
                task_id=task.task_id,
                node_id=node.id,
                state=ExecutionState.RUNNING,
            )
            exec_obj.logs.append(LogEntry(msg=f"Starting node: {node.objective[:80]}"))

            if on_progress:
                await on_progress(
                    f"Step {len(executions) + 1}/{len(ordered_nodes)}: {node.objective[:60]}..."
                )
            self._runtime.add_deep_profile_section(
                "Science · Execute Step",
                [
                    f"Node: {node.id}",
                    f"Objective: {node.objective}",
                    f"Dependencies: {', '.join(node.dependencies) if node.dependencies else '(none)'}",
                ],
            )

            start = time.monotonic()
            try:
                output = await self._execute_node(
                    task=task,
                    node=node,
                    prior_context=accumulated_context,
                    on_progress=on_progress,
                )
                elapsed = time.monotonic() - start
                exec_obj.output = output
                exec_obj.state = ExecutionState.WAITING_VALIDATION
                exec_obj.metrics = ExecutionMetrics(elapsed_seconds=round(elapsed, 2))
                exec_obj.logs.append(LogEntry(msg=f"Node completed in {elapsed:.1f}s"))
                accumulated_context.append(f"[{node.id}] {node.objective[:60]}:\n{output}")

            except asyncio.CancelledError:
                exec_obj.state = ExecutionState.CANCELLED
                exec_obj.error = "Cancelled"
                executions.append(exec_obj)
                raise
            except Exception as exc:
                elapsed = time.monotonic() - start
                exec_obj.state = ExecutionState.FAILED
                exec_obj.error = str(exc)
                exec_obj.metrics = ExecutionMetrics(elapsed_seconds=round(elapsed, 2))
                exec_obj.logs.append(LogEntry(level="ERROR", msg=f"Node failed: {exc}"))
                logger.error("Science node {} failed: {}", node.id, exc)
                executions.append(exec_obj)
                break

            executions.append(exec_obj)

        return executions

    async def _execute_node(
        self,
        task: Task,
        node: ScienceTaskNode,
        prior_context: list[str],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        prompt = _build_node_prompt(task, node, prior_context)
        session_key = self._runtime.internal_session_key(task.task_id, node.id)
        logger.info(
            "Science node {} executing: objective={} deps={} context_chars={}",
            node.id,
            node.objective[:80],
            ",".join(node.dependencies) if node.dependencies else "(none)",
            sum(len(c) for c in prior_context),
        )
        logger.debug(
            "Science node {} session_key={} prompt_chars={}",
            node.id,
            session_key,
            len(prompt),
        )
        filtered_progress = _filter_kernel_thinking(on_progress) if on_progress else None
        start = time.monotonic()
        result = await self._runtime.process_direct(
            content=prompt,
            session_key=session_key,
            channel="science",
            chat_id=task.task_id,
            on_progress=filtered_progress,
        )
        elapsed = time.monotonic() - start
        logger.info(
            "Science node {} completed: elapsed={:.2f}s output_chars={}",
            node.id,
            elapsed,
            len(result or ""),
        )
        return result or ""


# ---------------------------------------------------------------------------
# DAG (concurrent) orchestrator — Phase 1
# ---------------------------------------------------------------------------


class DAGOrchestrator(Orchestrator):
    """Concurrent DAG orchestrator with wave-based parallel execution.

    Nodes whose dependencies are fully satisfied are executed concurrently
    in each wave.  Retry logic per-node RetryPolicy is applied on failure
    before propagating errors.  A task-level budget is enforced between
    waves.
    """

    def __init__(self, runtime: PluginRuntime) -> None:
        self._runtime = runtime
        self._fetch_batch_size = 10
        self._fetch_max_rounds = 5

    async def run(
        self,
        task: Task,
        graph: ScienceTaskGraph,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> list[Execution]:
        node_map: dict[str, ScienceTaskNode] = {n.id: n for n in graph.nodes}
        # Mutable copy of remaining dependencies per node
        pending_deps: dict[str, set[str]] = {n.id: set(n.dependencies) for n in graph.nodes}
        outputs: dict[str, str] = {}  # node_id -> output text
        executions: dict[str, Execution] = {}
        task_start = time.monotonic()

        total_nodes = len(graph.nodes)
        completed = 0

        while pending_deps:
            # --- Budget check (between waves) ---
            elapsed = time.monotonic() - task_start
            if elapsed > task.budget.max_seconds:
                for nid in list(pending_deps):
                    ex = Execution(
                        task_id=task.task_id,
                        node_id=nid,
                        state=ExecutionState.CANCELLED,
                        error="Budget exceeded: time limit",
                    )
                    executions[nid] = ex
                raise BudgetExceededError(
                    f"Time budget exceeded ({elapsed:.0f}s > {task.budget.max_seconds}s)",
                    task.budget,
                )

            # --- Identify ready nodes ---
            ready_ids = [nid for nid, deps in pending_deps.items() if not deps]
            if not ready_ids:
                # All remaining nodes have unmet deps — shouldn't happen in valid graphs
                raise RuntimeError(
                    f"Deadlock: {len(pending_deps)} nodes blocked with no ready nodes. "
                    f"Remaining: {list(pending_deps)}"
                )

            for nid in ready_ids:
                del pending_deps[nid]

            if on_progress:
                names = ", ".join(nid.split("_", 2)[-1] for nid in ready_ids)
                await on_progress(
                    f"Running {'parallel' if len(ready_ids) > 1 else 'step'} "
                    f"({completed + 1}/{total_nodes}): {names}"
                )
            self._runtime.add_deep_profile_section(
                "Science · Execute Wave",
                [
                    f"Completed so far: {completed}/{total_nodes}",
                    f"Ready nodes: {', '.join(ready_ids)}",
                    f"Elapsed: {elapsed:.2f}s / {task.budget.max_seconds}s",
                ],
            )

            # --- Execute wave concurrently ---
            wave_tasks = [
                asyncio.create_task(
                    self._execute_with_retry(
                        task=task,
                        node=node_map[nid],
                        outputs=outputs,
                        on_progress=on_progress,
                    )
                )
                for nid in ready_ids
            ]
            wave_results = await asyncio.gather(*wave_tasks, return_exceptions=True)

            # --- Collect results ---
            failed_ids: list[str] = []
            for nid, result in zip(ready_ids, wave_results):
                if isinstance(result, BaseException):
                    ex = Execution(
                        task_id=task.task_id,
                        node_id=nid,
                        state=ExecutionState.FAILED,
                        error=str(result),
                    )
                    ex.logs.append(LogEntry(level="ERROR", msg=f"Node failed: {result}"))
                    executions[nid] = ex
                    failed_ids.append(nid)
                    logger.error("Science node {} failed: {}", nid, result)
                    self._runtime.add_deep_profile_section(
                        "Science · Node Failed",
                        [
                            f"Node: {nid}",
                            f"Error: {result}",
                        ],
                    )
                else:
                    exec_obj, output = result
                    executions[nid] = exec_obj
                    outputs[nid] = output
                    completed += 1
                    # Unblock downstream nodes
                    for other_id in list(pending_deps):
                        pending_deps[other_id].discard(nid)
                    self._runtime.add_deep_profile_section(
                        "Science · Node Completed",
                        [
                            f"Node: {nid}",
                            f"State: {exec_obj.state.value}",
                            f"Elapsed: {exec_obj.metrics.elapsed_seconds:.2f}s",
                            f"Output chars: {len(output)}",
                        ],
                    )

            if failed_ids:
                # Cancel remaining queued nodes
                for nid in list(pending_deps):
                    executions[nid] = Execution(
                        task_id=task.task_id,
                        node_id=nid,
                        state=ExecutionState.CANCELLED,
                        error=f"Cancelled due to failure in: {failed_ids}",
                    )
                pending_deps.clear()
                break

        # Return nodes in topological order
        return [executions[n.id] for n in graph.topological_order() if n.id in executions]

    async def _execute_with_retry(
        self,
        task: Task,
        node: ScienceTaskNode,
        outputs: dict[str, str],
        on_progress: Callable[..., Awaitable[None]] | None,
    ) -> tuple[Execution, str]:
        """Execute a node with retry logic; returns (Execution, output_str)."""
        policy = node.retry_policy
        max_attempts = 1 + max(0, policy.max_retries)
        last_exc: Exception | None = None
        filtered = _filter_kernel_thinking(on_progress) if on_progress else None

        # Build prior context from already-completed upstream outputs
        prior_context = [
            f"[{dep_id}] {outputs[dep_id]}" for dep_id in node.dependencies if dep_id in outputs
        ]

        for attempt in range(1, max_attempts + 1):
            exec_obj = Execution(
                task_id=task.task_id,
                node_id=node.id,
                state=ExecutionState.RUNNING,
            )
            start = time.monotonic()
            try:
                if attempt > 1:
                    await asyncio.sleep(policy.backoff_seconds * (attempt - 1))
                    if on_progress:
                        await on_progress(f"Retrying {node.id} (attempt {attempt}/{max_attempts})")
                    exec_obj.logs.append(LogEntry(msg=f"Retry attempt {attempt}"))
                    self._runtime.add_deep_profile_section(
                        "Science · Retry",
                        [
                            f"Node: {node.id}",
                            f"Attempt: {attempt}/{max_attempts}",
                            f"Backoff: {policy.backoff_seconds * (attempt - 1):.2f}s",
                        ],
                    )

                prompt = _build_node_prompt(task, node, prior_context)
                session_key = self._runtime.internal_session_key(
                    task.task_id, node.id, str(attempt)
                )
                logger.info(
                    "Science node {} attempt {}/{}: objective={} deps={} context_chars={}",
                    node.id,
                    attempt,
                    max_attempts,
                    node.objective[:80],
                    ",".join(node.dependencies) if node.dependencies else "(none)",
                    sum(len(c) for c in prior_context),
                )
                logger.debug(
                    "Science node {} attempt {} session_key={} is_fetch={}",
                    node.id,
                    attempt,
                    session_key,
                    _is_fetch_research_node(node),
                )
                if _is_fetch_research_node(node):
                    output, fetch_meta = await self._execute_fetch_research_rounds(
                        task=task,
                        node=node,
                        prior_context=prior_context,
                        session_key=session_key,
                        on_progress=filtered,
                    )
                else:
                    node_start = time.monotonic()
                    output = await self._runtime.process_direct(
                        content=prompt,
                        session_key=session_key,
                        channel="science",
                        chat_id=task.task_id,
                        on_progress=filtered,
                    )
                    node_elapsed = time.monotonic() - node_start
                    logger.info(
                        "Science node {} process_direct completed: elapsed={:.2f}s output_chars={}",
                        node.id,
                        node_elapsed,
                        len(output or ""),
                    )
                    fetch_meta = None
                output = output or ""
                elapsed = time.monotonic() - start
                exec_obj.output = output
                exec_obj.state = ExecutionState.WAITING_VALIDATION
                exec_obj.metrics = ExecutionMetrics(elapsed_seconds=round(elapsed, 2))
                if fetch_meta is not None:
                    exec_obj.artifacts["fetch_rounds"] = json.dumps(fetch_meta, ensure_ascii=False)
                exec_obj.logs.append(
                    LogEntry(msg=f"Completed (attempt {attempt}) in {elapsed:.1f}s")
                )
                logger.info(
                    "Science node {} attempt {}/{} succeeded: elapsed={:.2f}s output_chars={}",
                    node.id,
                    attempt,
                    max_attempts,
                    elapsed,
                    len(output),
                )
                return exec_obj, output

            except asyncio.CancelledError:
                exec_obj.state = ExecutionState.CANCELLED
                exec_obj.error = "Cancelled"
                raise
            except Exception as exc:
                elapsed = time.monotonic() - start
                exec_obj.logs.append(
                    LogEntry(level="ERROR", msg=f"Attempt {attempt} failed: {exc}")
                )
                last_exc = exc
                logger.warning(
                    "Node {} attempt {}/{} failed: {}", node.id, attempt, max_attempts, exc
                )

        # All attempts exhausted
        exec_obj = Execution(
            task_id=task.task_id,
            node_id=node.id,
            state=ExecutionState.FAILED,
            error=str(last_exc),
        )
        raise last_exc or RuntimeError(f"Node {node.id} failed after {max_attempts} attempts")

    async def _execute_fetch_research_rounds(
        self,
        task: Task,
        node: ScienceTaskNode,
        prior_context: list[str],
        session_key: str,
        on_progress: Callable[..., Awaitable[None]] | None,
    ) -> tuple[str, dict[str, object]]:
        candidate_urls = _extract_urls_from_context(prior_context)
        remaining_urls = candidate_urls[: self._fetch_batch_size * self._fetch_max_rounds]
        fetched_records: list[dict[str, str]] = []
        round_summaries: list[str] = []
        stop_reason = "no_more_urls"

        if not remaining_urls:
            logger.info(
                "Science node {}: no candidate URLs, falling back to process_direct",
                node.id,
            )
            fallback_prompt = _build_node_prompt(task, node, prior_context)
            output = await self._runtime.process_direct(
                content=fallback_prompt,
                session_key=session_key,
                channel="science",
                chat_id=task.task_id,
                on_progress=on_progress,
            )
            return output or "", {
                "rounds_completed": 0,
                "candidate_urls": 0,
                "fetched_urls": 0,
                "stop_reason": "no_candidate_urls",
            }

        if not self._runtime.supports_async_tool_execute:
            logger.info(
                "Science node {}: async tool_execute not supported, falling back to process_direct",
                node.id,
            )
            fallback_prompt = _build_node_prompt(task, node, prior_context)
            output = await self._runtime.process_direct(
                content=fallback_prompt,
                session_key=session_key,
                channel="science",
                chat_id=task.task_id,
                on_progress=on_progress,
            )
            return output or "", {
                "rounds_completed": 0,
                "candidate_urls": len(candidate_urls),
                "fetched_urls": 0,
                "stop_reason": "fallback_process_direct",
            }

        total_rounds = min(
            self._fetch_max_rounds,
            (len(remaining_urls) + self._fetch_batch_size - 1) // self._fetch_batch_size,
        )

        for round_index in range(total_rounds):
            batch = remaining_urls[: self._fetch_batch_size]
            remaining_urls = remaining_urls[self._fetch_batch_size :]
            if on_progress:
                await on_progress(
                    f"Research round {round_index + 1}/{self._fetch_max_rounds}: "
                    f"concurrently fetching {len(batch)} sources"
                )
            self._runtime.add_deep_profile_section(
                f"Science · Fetch Round {round_index + 1}",
                [
                    f"Batch size: {len(batch)}",
                    f"Round: {round_index + 1}/{self._fetch_max_rounds}",
                    f"URLs: {', '.join(batch[:5])}{' ...' if len(batch) > 5 else ''}",
                ],
            )

            fetch_tasks = [
                asyncio.create_task(
                    self._runtime.tool_execute(
                        "web_fetch",
                        {
                            "url": url,
                            "extractMode": "markdown",
                            "maxChars": 6000,
                            "on_progress": on_progress,
                        },
                    )
                )
                for url in batch
            ]
            fetch_outputs = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            round_records: list[dict[str, str]] = []
            round_ok = 0
            round_err = 0
            round_status_lines: list[str] = []
            for url, result in zip(batch, fetch_outputs):
                if isinstance(result, BaseException):
                    round_records.append({"url": url, "status": "error", "content": str(result)})
                    round_err += 1
                    round_status_lines.append(f"  FAIL {url[:80]} — {result}")
                    logger.warning("Science fetch failed: url={} error={}", url[:120], result)
                else:
                    classified = _classify_fetch_result(str(result), url)
                    round_records.append(
                        {"url": url, "status": classified["status"], "content": str(result)}
                    )
                    if classified["status"] == "ok":
                        round_ok += 1
                        round_status_lines.append(f"  OK   {url[:80]}")
                    else:
                        round_err += 1
                        round_status_lines.append(f"  FAIL {url[:80]} — {classified['reason']}")
                    logger.debug(
                        "Science fetch {}: url={} content_chars={}",
                        classified["status"],
                        url[:120],
                        len(str(result)),
                    )
            fetched_records.extend(round_records)
            logger.info(
                "Science fetch round {}/{}: batch={} ok={} err={} cumulative={}",
                round_index + 1,
                self._fetch_max_rounds,
                len(batch),
                round_ok,
                round_err,
                len(fetched_records),
            )

            # Emit per-URL status summary via progress callback
            if on_progress:
                summary_lines = [
                    f"Fetch round {round_index + 1}: {round_ok} ok, {round_err} failed"
                ]
                summary_lines.extend(round_status_lines)
                await on_progress("\n".join(summary_lines))

            summary = await self._summarize_fetch_round(
                task=task,
                node=node,
                round_index=round_index,
                round_records=round_records,
                cumulative_records=fetched_records,
                session_key=session_key,
                on_progress=on_progress,
            )
            round_summaries.append(summary)
            # Build detailed per-URL status for deep profile
            profile_lines = [
                f"Fetched sources: {len(round_records)}",
                f"Cumulative sources: {len(fetched_records)}",
                f"Decision: {'enough_information' if _round_summary_is_sufficient(summary) else 'continue_research'}",
            ]
            for rec in round_records:
                if rec["status"] == "ok":
                    profile_lines.append(f"  OK   {rec['url'][:100]}")
                else:
                    # Re-classify for human-readable reason
                    classified = _classify_fetch_result(rec["content"], rec["url"])
                    profile_lines.append(
                        f"  FAIL {rec['url'][:100]} — {classified.get('reason') or rec['status']}"
                    )
            profile_lines.append(f"Summary preview: {summary[:200]}")
            self._runtime.add_deep_profile_section(
                f"Science · Fetch Round {round_index + 1} Summary",
                profile_lines,
            )

            if _round_summary_is_sufficient(summary):
                stop_reason = "enough_information"
                break
            if round_index == self._fetch_max_rounds - 1:
                stop_reason = "max_rounds"
                break
            if not remaining_urls:
                stop_reason = "no_more_urls"
                break

        final_output = _build_fetch_round_report(
            node=node,
            fetched_records=fetched_records,
            round_summaries=round_summaries,
            stop_reason=stop_reason,
        )
        logger.info(
            "Science fetch research done: node={} rounds={} fetched={}/{} "
            "stop_reason={} output_chars={}",
            node.id,
            len(round_summaries),
            len(fetched_records),
            len(candidate_urls),
            stop_reason,
            len(final_output),
        )
        return final_output, {
            "rounds_completed": len(round_summaries),
            "candidate_urls": len(candidate_urls),
            "fetched_urls": len(fetched_records),
            "stop_reason": stop_reason,
        }

    async def _summarize_fetch_round(
        self,
        task: Task,
        node: ScienceTaskNode,
        round_index: int,
        round_records: list[dict[str, str]],
        cumulative_records: list[dict[str, str]],
        session_key: str,
        on_progress: Callable[..., Awaitable[None]] | None,
    ) -> str:
        prompt_parts = [
            f"[Science Task: {task.task_id}]",
            f"Goal: {task.goal}",
            f"Fetch objective: {node.objective}",
            f"Round: {round_index + 1}/{self._fetch_max_rounds}",
            "",
            "Review the fetched source batch below and write a concise round summary.",
            "You must also decide whether enough information has been collected.",
            "Include a final line exactly in one of these forms:",
            "Decision: enough_information",
            "Decision: continue_research",
            "",
            "Round batch records:",
        ]
        for record in round_records:
            snippet = record["content"][:1200]
            prompt_parts.append(
                f"- URL: {record['url']}\n  Status: {record['status']}\n  Content: {snippet}"
            )
        prompt_parts.extend(
            [
                "",
                f"Cumulative sources collected so far: {len(cumulative_records)}",
                "Focus on novelty, redundancy, and whether the evidence already covers the task well.",
            ]
        )
        output = await self._runtime.process_direct(
            content="\n".join(prompt_parts),
            session_key=f"{session_key}:round{round_index + 1}:summary",
            channel="science",
            chat_id=task.task_id,
            on_progress=on_progress,
        )
        if on_progress:
            await on_progress(
                f"Completed research round {round_index + 1}: summarized {len(round_records)} fetched sources"
            )
        return output or "Decision: continue_research"


# ---------------------------------------------------------------------------
# Shared prompt builder
# ---------------------------------------------------------------------------


def _build_node_prompt(
    task: Task,
    node: ScienceTaskNode,
    prior_context: list[str],
) -> str:
    parts: list[str] = [
        f"[Science Task: {task.task_id}]\nOverall Goal: {task.goal}\nCurrent Step: {node.objective}"
    ]

    if prior_context:
        parts.append("--- Previous step outputs ---")
        for ctx in prior_context[-3:]:
            # Truncate very large prior context to stay within token budget
            parts.append(ctx[:3000] + ("..." if len(ctx) > 3000 else ""))
        parts.append("--- End of previous outputs ---")

    parts.append(
        "Please complete the current step described above. "
        "Be thorough and specific. "
        "Use the available tools (web_search, web_fetch, etc.) as needed."
    )

    return "\n\n".join(parts)


def _is_fetch_research_node(node: ScienceTaskNode) -> bool:
    return "web_fetch" in node.candidate_capabilities


def _extract_urls_from_context(prior_context: list[str]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for ctx in prior_context:
        for url in re.findall(r"https?://[^\s)\]>\"']+", ctx):
            normalized = url.rstrip(".,;")
            if normalized not in seen:
                seen.add(normalized)
                urls.append(normalized)
    return urls


def _round_summary_is_sufficient(summary: str) -> bool:
    lowered = summary.lower()
    return "decision: enough_information" in lowered or "decision: enough info" in lowered


def _build_fetch_round_report(
    *,
    node: ScienceTaskNode,
    fetched_records: list[dict[str, str]],
    round_summaries: list[str],
    stop_reason: str,
) -> str:
    lines = [
        f"Fetch research report for node `{node.id}`",
        "",
        f"Stop reason: {stop_reason}",
        f"Total fetched sources: {len(fetched_records)}",
        "",
        "## Round Summaries",
    ]
    for idx, summary in enumerate(round_summaries, start=1):
        lines.extend([f"### Round {idx}", summary, ""])
    lines.append("## Source Status")
    for record in fetched_records:
        lines.append(f"- {record['status']}: {record['url']}")
    return "\n".join(lines)


def _classify_fetch_result(content: str, url: str) -> dict[str, str]:
    """Classify a fetch result string into status + human-readable reason.

    Returns {"status": "ok"|"timeout"|"bot_detection"|"server_error"|"error", "reason": ...}.
    """
    # Try to parse as JSON (web_fetch returns JSON)
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        # Non-JSON — treat as ok if non-empty
        return {"status": "ok" if content.strip() else "error", "reason": ""}

    # Has explicit error key
    error_msg = data.get("error", "")
    if not error_msg:
        # No error — check for HTTP success
        status_code = data.get("status")
        if status_code and isinstance(status_code, int) and status_code >= 400:
            reason = _classify_http_status(status_code)
            return {"status": reason["category"], "reason": reason["label"]}
        return {"status": "ok", "reason": ""}

    # Classify the error message
    error_lower = error_msg.lower()
    if "timeout" in error_lower or "timed out" in error_lower:
        if "connection" in error_lower or "connect" in error_lower:
            return {"status": "timeout", "reason": "connection failed (slow network)"}
        if "read" in error_lower:
            return {"status": "timeout", "reason": "server too slow to respond"}
        return {"status": "timeout", "reason": "request timed out"}
    if "rate limit" in error_lower or "429" in error_lower:
        return {"status": "bot_detection", "reason": "rate limited (bot detection)"}
    if "access denied" in error_lower or "403" in error_lower or "forbidden" in error_lower:
        return {"status": "bot_detection", "reason": "access denied (bot detection)"}
    if "503" in error_lower or "service unavailable" in error_lower:
        return {"status": "server_error", "reason": "service unavailable"}
    if any(code in error_lower for code in ("500", "502", "504")):
        return {"status": "server_error", "reason": "upstream server error"}
    if "proxy" in error_lower:
        return {"status": "error", "reason": "proxy error"}
    if "redirect" in error_lower:
        return {"status": "error", "reason": "redirect blocked"}
    return {"status": "error", "reason": error_msg[:100]}


def _classify_http_status(status_code: int) -> dict[str, str]:
    """Classify an HTTP status code into category + label."""
    if status_code == 429:
        return {"category": "bot_detection", "label": "rate limited (bot detection)"}
    if status_code == 403:
        return {"category": "bot_detection", "label": "access denied (bot detection)"}
    if status_code == 503:
        return {"category": "server_error", "label": "service unavailable"}
    if 500 <= status_code < 600:
        return {"category": "server_error", "label": f"server error (HTTP {status_code})"}
    if 400 <= status_code < 500:
        return {"category": "error", "label": f"client error (HTTP {status_code})"}
    return {"category": "error", "label": f"HTTP {status_code}"}
