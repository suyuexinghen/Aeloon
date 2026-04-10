"""Middleware contracts and built-in middleware for agent kernel."""

from __future__ import annotations

from abc import ABC
from time import perf_counter
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from aeloon.core.agent.profiler import SpanCategory

if TYPE_CHECKING:
    from aeloon.core.agent.profiler import AgentProfiler
    from aeloon.providers.base import LLMResponse


def default_tool_args_summary(arguments: dict | list | None) -> str:
    """Summarize tool arguments in one short string."""
    if arguments is None:
        return ""

    args = arguments[0] if isinstance(arguments, list) and arguments else arguments
    if not isinstance(args, dict):
        text = str(args)
        return text[:100] + "…" if len(text) > 100 else text

    parts: list[str] = []
    for key, value in list(args.items())[:3]:
        if isinstance(value, str):
            preview = value if len(value) <= 40 else value[:40] + "…"
        elif isinstance(value, (int, float, bool)) or value is None:
            preview = str(value)
        else:
            preview = f"<{type(value).__name__}>"
        parts.append(f"{key}={preview}")
    if len(args) > 3:
        parts.append("…")
    return ", ".join(parts)


class AgentMiddleware(Protocol):
    """Protocol for around-style middleware hooks in the agent kernel."""

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable["LLMResponse"]],
    ) -> "LLMResponse": ...

    async def around_tool(
        self,
        name: str,
        args: dict | list | None,
        execute: Callable[[], Awaitable[str]],
    ) -> str: ...


class BaseAgentMiddleware(ABC):
    """No-op base class for middleware implementations."""

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable["LLMResponse"]],
    ) -> "LLMResponse":
        return await call_llm(messages, tool_defs)

    async def around_tool(
        self,
        name: str,
        args: dict | list | None,
        execute: Callable[[], Awaitable[str]],
    ) -> str:
        return await execute()


class ProfilerMiddleware(BaseAgentMiddleware):
    """Middleware that records LLM/tool timing into AgentProfiler."""

    def __init__(
        self,
        profiler: "AgentProfiler",
        model: str,
        tool_args_summary: Callable[[dict | list | None], str] | None = None,
    ):
        self._profiler = profiler
        self._model = model
        self._tool_args_summary = tool_args_summary or default_tool_args_summary

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable["LLMResponse"]],
    ) -> "LLMResponse":
        t0 = perf_counter()
        try:
            response = await call_llm(messages, tool_defs)
        except BaseException as exc:
            self._profiler.record(
                SpanCategory.LLM,
                self._model,
                (perf_counter() - t0) * 1000,
                meta={"error": True, "error_type": type(exc).__name__},
            )
            raise

        self._profiler.record(
            SpanCategory.LLM,
            self._model,
            (perf_counter() - t0) * 1000,
            meta={"usage": response.usage},
        )
        return response

    async def around_tool(
        self,
        name: str,
        args: dict | list | None,
        execute: Callable[[], Awaitable[str]],
    ) -> str:
        t0 = perf_counter()
        args_summary = self._tool_args_summary(args)
        try:
            result = await execute()
        except BaseException as exc:
            self._profiler.record(
                SpanCategory.TOOL,
                name,
                (perf_counter() - t0) * 1000,
                meta={
                    "args_summary": args_summary,
                    "error": True,
                    "error_type": type(exc).__name__,
                },
            )
            raise

        self._profiler.record(
            SpanCategory.TOOL,
            name,
            (perf_counter() - t0) * 1000,
            meta={"args_summary": args_summary},
        )
        return result
