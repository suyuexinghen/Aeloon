"""Budget middleware: tracks token and time consumption per science task."""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from loguru import logger
from pydantic import BaseModel, Field


class BudgetState(BaseModel):
    """Mutable snapshot of resource consumption for a single science task."""

    tokens_used: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0
    task_start: float = Field(default_factory=time.monotonic)
    max_tokens: int = 50_000
    max_tool_calls: int = 100
    max_seconds: int = 600
    halted: bool = False

    model_config = {"arbitrary_types_allowed": True}

    @property
    def tokens_remaining(self) -> int:
        """Tokens still available before the token budget is exceeded."""
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def is_over_budget(self) -> bool:
        """True if any resource limit has been exceeded."""
        return (
            self.tokens_used > self.max_tokens
            or self.tool_calls > self.max_tool_calls
            or self.elapsed_seconds > self.max_seconds
        )


class BudgetExceededError(Exception):
    """Raised when any budget limit is exceeded during task execution."""

    def __init__(self, budget_state: BudgetState) -> None:
        self.budget_state = budget_state
        super().__init__(
            f"Budget exceeded — tokens={budget_state.tokens_used}/{budget_state.max_tokens}, "
            f"tool_calls={budget_state.tool_calls}/{budget_state.max_tool_calls}, "
            f"elapsed={budget_state.elapsed_seconds:.1f}s/{budget_state.max_seconds}s"
        )


class BudgetMiddleware:
    """Middleware that enforces token, tool-call, and wall-clock time budgets."""

    def __init__(self, budget_state: BudgetState) -> None:
        self._budget = budget_state

    def snapshot(self) -> BudgetState:
        """Return a copy of the current budget state."""
        return self._budget.model_copy()

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable[Any]],
    ) -> Any:
        """Call the LLM then account for token usage; halt if over budget."""
        response = await call_llm(messages, tool_defs)

        # Rough token estimate from response content length
        token_estimate = len(response.content or "") // 4
        self._budget.tokens_used += token_estimate
        self._budget.elapsed_seconds = time.monotonic() - self._budget.task_start

        logger.debug(
            "BudgetMiddleware: LLM call — tokens_used={}, elapsed={:.1f}s",
            self._budget.tokens_used,
            self._budget.elapsed_seconds,
        )

        if self._budget.is_over_budget:
            self._budget.halted = True
            raise BudgetExceededError(self._budget)

        return response

    async def around_tool(
        self,
        name: str,
        args: dict | list | None,
        execute: Callable[[], Awaitable[str]],
    ) -> str:
        """Check tool-call limit before execution; update elapsed time after."""
        # Guard: check tool_calls limit BEFORE executing
        self._budget.tool_calls += 1
        if self._budget.tool_calls > self._budget.max_tool_calls:
            self._budget.halted = True
            raise BudgetExceededError(self._budget)

        result = await execute()

        self._budget.elapsed_seconds = time.monotonic() - self._budget.task_start

        logger.debug(
            "BudgetMiddleware: tool '{}' — tool_calls={}, elapsed={:.1f}s",
            name,
            self._budget.tool_calls,
            self._budget.elapsed_seconds,
        )

        if self._budget.is_over_budget:
            self._budget.halted = True
            raise BudgetExceededError(self._budget)

        return result
