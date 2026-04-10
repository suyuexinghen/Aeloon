"""Science task planners: convert a Task into an executable ScienceTaskGraph."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .task import (
    RetryPolicy,
    ScienceTaskGraph,
    ScienceTaskNode,
    Task,
)


class Planner(ABC):
    """Abstract base class for science task planners."""

    @abstractmethod
    def plan(self, task: Task) -> ScienceTaskGraph:
        """Convert a structured Task into an executable ScienceTaskGraph."""
        ...

    @staticmethod
    def _make_node(
        node_id: str,
        objective: str,
        *,
        dependencies: list[str] | None = None,
        inputs: list[str] | None = None,
        expected_outputs: list[str] | None = None,
        role: str = "executor",
        capabilities: list[str] | None = None,
        max_retries: int = 1,
    ) -> ScienceTaskNode:
        return ScienceTaskNode(
            id=node_id,
            objective=objective,
            dependencies=dependencies or [],
            inputs=inputs or [],
            expected_outputs=expected_outputs or [],
            assigned_role=role,
            candidate_capabilities=capabilities or [],
            retry_policy=RetryPolicy(max_retries=max_retries),
        )


class LinearPlanner(Planner):
    """Generates a linear (sequential) task graph for a science task.

    For the walking skeleton this emits a template plan suitable for
    literature-analysis tasks.  A future LLM-based planner will replace
    the template logic once the full control path is proven.
    """

    def plan(self, task: Task) -> ScienceTaskGraph:
        goal = task.goal
        nodes = self._build_literature_analysis_plan(goal, task.task_id)
        return ScienceTaskGraph(task_id=task.task_id, nodes=nodes)

    # ------------------------------------------------------------------
    # Internal plan templates
    # ------------------------------------------------------------------

    def _build_literature_analysis_plan(self, goal: str, task_id: str) -> list[ScienceTaskNode]:
        """Three-node linear plan for a literature analysis task."""
        search = ScienceTaskNode(
            id=f"{task_id}_search",
            objective=(
                f"Search the web for recent academic papers, research articles, "
                f"and authoritative sources relevant to: {goal}. "
                f"Return a list of up to 8 highly relevant results with titles, URLs, and brief descriptions."
            ),
            dependencies=[],
            inputs=["task_goal"],
            expected_outputs=["search_results"],
            assigned_role="researcher",
            candidate_capabilities=["web_search"],
            retry_policy=RetryPolicy(max_retries=2),
        )

        fetch = ScienceTaskNode(
            id=f"{task_id}_fetch",
            objective=(
                "Using the search results from the previous step, fetch and read the content "
                "of the top 3-5 most relevant sources. "
                "Extract key information: title, authors/organization, publication date, "
                "main claims, key findings, and methodology."
            ),
            dependencies=[search.id],
            inputs=["search_results"],
            expected_outputs=["paper_contents", "structured_findings"],
            assigned_role="researcher",
            candidate_capabilities=["web_fetch"],
            retry_policy=RetryPolicy(max_retries=1),
        )

        synthesize = ScienceTaskNode(
            id=f"{task_id}_synthesize",
            objective=(
                "Using all the information gathered, produce a comprehensive scientific summary. "
                f"The summary must address: {goal}\n\n"
                "Structure the output as follows:\n"
                "## Summary\n[2-3 paragraph overview of the field and current state]\n\n"
                "## Key Findings\n[Bullet points of the most important findings across all sources]\n\n"
                "## Notable Research\n[Brief description of each source with key contributions]\n\n"
                "## Sources\n[Numbered list of all sources with titles and URLs]\n\n"
                "Be specific, cite sources inline, and highlight any consensus or disagreements."
            ),
            dependencies=[fetch.id],
            inputs=["paper_contents", "structured_findings"],
            expected_outputs=["final_report"],
            assigned_role="analyst",
            candidate_capabilities=["llm_analysis"],
            retry_policy=RetryPolicy(max_retries=1),
        )

        return [search, fetch, synthesize]


class DAGPlanner(Planner):
    """Generates a DAG task graph that exposes parallel execution opportunities.

    For a single-scope task this produces the same 3-node linear structure
    as LinearPlanner.  When the task has multiple scope items, each scope
    gets its own search+fetch branch that converges at a single synthesis
    node — enabling concurrent web research across sub-topics.
    """

    def plan(self, task: Task) -> ScienceTaskGraph:
        scopes = [s for s in (task.scope or []) if s.strip()]
        if len(scopes) > 1:
            nodes = self._build_parallel_scope_plan(task.goal, task.task_id, scopes)
        else:
            nodes = self._build_single_scope_plan(task.goal, task.task_id)
        return ScienceTaskGraph(task_id=task.task_id, nodes=nodes)

    def _build_single_scope_plan(self, goal: str, task_id: str) -> list[ScienceTaskNode]:
        """Falls back to a 3-node linear plan when only one scope is present."""
        return LinearPlanner()._build_literature_analysis_plan(goal, task_id)  # noqa: SLF001

    def _build_parallel_scope_plan(
        self,
        goal: str,
        task_id: str,
        scopes: list[str],
    ) -> list[ScienceTaskNode]:
        """One search+fetch branch per scope, merged by a single synthesis node."""
        nodes: list[ScienceTaskNode] = []
        fetch_ids: list[str] = []

        for i, scope in enumerate(scopes[:4]):  # cap at 4 parallel branches
            search_id = f"{task_id}_search_{i}"
            fetch_id = f"{task_id}_fetch_{i}"

            nodes.append(
                self._make_node(
                    search_id,
                    f"Search the web for: {scope} (as part of the broader goal: {goal}). "
                    f"Return up to 5 relevant results with titles, URLs, and descriptions.",
                    capabilities=["web_search"],
                    max_retries=2,
                )
            )
            nodes.append(
                self._make_node(
                    fetch_id,
                    f"Fetch and extract key information from the top 2-3 results about: {scope}.",
                    dependencies=[search_id],
                    inputs=[f"search_results_{i}"],
                    expected_outputs=[f"findings_{i}"],
                    capabilities=["web_fetch"],
                    max_retries=1,
                )
            )
            fetch_ids.append(fetch_id)

        nodes.append(
            self._make_node(
                f"{task_id}_synthesize",
                (
                    "Synthesize all gathered findings into a comprehensive report.\n"
                    f"Overall goal: {goal}\n\n"
                    "Structure:\n"
                    "## Summary\n## Key Findings\n## Notable Research\n## Sources"
                ),
                dependencies=fetch_ids,
                inputs=["all_findings"],
                expected_outputs=["final_report"],
                role="analyst",
                capabilities=["llm_analysis"],
                max_retries=1,
            )
        )
        return nodes
