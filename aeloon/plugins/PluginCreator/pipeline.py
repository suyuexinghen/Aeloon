"""PluginCreatorPipeline — stub pipeline for the /pc command.

Sprint 1: minimal routing.  Full pipeline orchestration in a later sprint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import PlanPackage
from .planner.kernel import PlanningKernel, PlanningKernelInput
from .storage.jsonl import PlanStore

if TYPE_CHECKING:
    from aeloon.plugins._sdk.runtime import PluginRuntime


class PluginCreatorPipeline:
    """Orchestrates the PluginCreator workflow via PlanningKernel."""

    def __init__(self, runtime: PluginRuntime, storage_dir: str) -> None:
        self._runtime = runtime
        self._store = PlanStore(storage_dir)
        self._kernel = PlanningKernel(runtime)

    async def plan(self, requirement: str, **kwargs: Any) -> tuple[str, PlanPackage | None]:
        """Run PlanningKernel and return (rendered_output, package)."""
        inp = PlanningKernelInput(
            project_id=kwargs.get("project_id", "default"),
            raw_requirement=requirement,
        )
        output = await self._kernel.plan(inp)
        if output.plan_package:
            self._store.save(output.plan_package)
        return output.full_view, output.plan_package

    def get_status(self) -> str:
        """Return a summary of stored plans."""
        ids = self._store.list_project_ids()
        if not ids:
            return "No PluginCreator plans stored."
        return f"Stored plans: {len(ids)} projects ({', '.join(ids[:5])})"

    def get_history(self) -> str:
        """Return a history of stored plans."""
        ids = self._store.list_project_ids()
        if not ids:
            return "No PluginCreator history."
        lines = [f"  {pid}" for pid in ids]
        return "PluginCreator history:\n" + "\n".join(lines)


def get_help_text() -> str:
    """Return help text for the /pc command."""
    return (
        "**PluginCreator** — intelligent plugin development workflow\n\n"
        "Commands:\n"
        "  /pc help          — Show this help\n"
        "  /pc status        — Show stored plan status\n"
        "  /pc history       — Show plan history\n"
        "  /pc plan <desc>   — Create a new plugin plan\n"
        "  /pc <desc>        — Shortcut for plan\n"
    )
