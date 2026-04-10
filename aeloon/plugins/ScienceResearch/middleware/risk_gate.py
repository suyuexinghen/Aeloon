"""Risk gate middleware stub (v0.8+ placeholder).

Risk classification for science tasks:
  Green  — auto-approved, no intervention
  Yellow — budget check required before proceeding
  Red    — human approval required (deferred to v0.8+)

Currently only Green is active; Yellow and Red are stubs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Awaitable, Callable

from loguru import logger


class RiskLevel(str, Enum):
    GREEN = "green"  # auto-approved
    YELLOW = "yellow"  # budget check required
    RED = "red"  # human approval required (future)


class RiskClassification:
    """Classifies a task based on its goal and scope (stub logic)."""

    def classify(self, goal: str, scope: list[str]) -> RiskLevel:
        """Return a risk level for the given task parameters.

        Current logic:
          - Always returns GREEN (stub — real classification in v0.8+)
          - Future: inspect tool side-effects, data sensitivity, external writes
        """
        return RiskLevel.GREEN


class RiskGateMiddleware:
    """Middleware stub that enforces risk-level policies.

    Active behaviour:
      - GREEN: pass-through (no action)
      - YELLOW: logs a warning (budget enforcement deferred to BudgetMiddleware)
      - RED: raises NotImplementedError (human approval not yet implemented)
    """

    def __init__(self, risk_level: RiskLevel = RiskLevel.GREEN) -> None:
        self._risk_level = risk_level

    @property
    def risk_level(self) -> RiskLevel:
        return self._risk_level

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable[Any]],
    ) -> Any:
        self._check_risk()
        return await call_llm(messages, tool_defs)

    async def around_tool(
        self,
        name: str,
        args: dict | list | None,
        execute: Callable[[], Awaitable[str]],
    ) -> str:
        self._check_risk()
        return await execute()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_risk(self) -> None:
        if self._risk_level == RiskLevel.GREEN:
            return
        if self._risk_level == RiskLevel.YELLOW:
            logger.warning(
                "RiskGateMiddleware: YELLOW risk level — budget enforcement active. "
                "Ensure BudgetMiddleware is also in the pipeline."
            )
            return
        # RED — not implemented yet
        raise NotImplementedError(
            "RiskGateMiddleware: RED risk level requires human approval, "
            "which is not yet implemented (planned for v0.8+)."
        )
