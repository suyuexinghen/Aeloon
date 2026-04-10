from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.core.agent.kernel import run_agent_kernel
from aeloon.core.agent.middleware import BaseAgentMiddleware, ProfilerMiddleware
from aeloon.core.agent.profiler import AgentProfiler
from aeloon.core.agent.tools.base import Tool
from aeloon.core.agent.tools.registry import ToolRegistry
from aeloon.providers.base import LLMResponse, ToolCallRequest


class _EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo content"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, text: str, **kwargs) -> str:
        return f"echo:{text}"


class _TraceMiddleware(BaseAgentMiddleware):
    def __init__(self, label: str, events: list[str]):
        self._label = label
        self._events = events

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm,
    ) -> LLMResponse:
        self._events.append(f"{self._label}:llm:before")
        response = await call_llm(messages, tool_defs)
        self._events.append(f"{self._label}:llm:after")
        return response

    async def around_tool(self, name: str, args: dict | list | None, execute) -> str:
        self._events.append(f"{self._label}:tool:{name}:before")
        result = await execute()
        self._events.append(f"{self._label}:tool:{name}:after")
        return result


@pytest.mark.asyncio
async def test_kernel_middleware_order_is_stable() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="echo", arguments={"text": "hi"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_EchoTool())

    events: list[str] = []
    final_content, tools_used, _ = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "hello"}],
        max_iterations=5,
        middlewares=[_TraceMiddleware("m1", events), _TraceMiddleware("m2", events)],
    )

    assert final_content == "done"
    assert tools_used == ["echo"]
    assert events == [
        "m1:llm:before",
        "m2:llm:before",
        "m2:llm:after",
        "m1:llm:after",
        "m1:tool:echo:before",
        "m2:tool:echo:before",
        "m2:tool:echo:after",
        "m1:tool:echo:after",
        "m1:llm:before",
        "m2:llm:before",
        "m2:llm:after",
        "m1:llm:after",
    ]


@pytest.mark.asyncio
async def test_profiler_middleware_handles_concurrent_llm_calls() -> None:
    profiler = AgentProfiler(enabled=True)
    profiler.start_turn()
    middleware = ProfilerMiddleware(profiler=profiler, model="test-model")

    async def _call(delay: float, total_tokens: int) -> None:
        async def _do_llm(_messages: list[dict], _tool_defs: list[dict]) -> LLMResponse:
            await asyncio.sleep(delay)
            return LLMResponse(content="ok", tool_calls=[], usage={"total_tokens": total_tokens})

        await middleware.around_llm([], [], _do_llm)

    await asyncio.gather(_call(0.03, 31), _call(0.01, 11))
    report = profiler.end_turn()

    assert len(report.llm_calls) == 2
    token_counts = sorted(
        int(sample.meta.get("usage", {}).get("total_tokens", 0)) for sample in report.llm_calls
    )
    assert token_counts == [11, 31]
    assert all(sample.duration_ms >= 0 for sample in report.llm_calls)


@pytest.mark.asyncio
async def test_profiler_middleware_records_llm_span_on_exception() -> None:
    profiler = AgentProfiler(enabled=True)
    profiler.start_turn()
    middleware = ProfilerMiddleware(profiler=profiler, model="test-model")

    async def _raise_llm(_messages: list[dict], _tool_defs: list[dict]) -> LLMResponse:
        raise RuntimeError("llm boom")

    with pytest.raises(RuntimeError, match="llm boom"):
        await middleware.around_llm([], [], _raise_llm)

    report = profiler.end_turn()
    assert len(report.llm_calls) == 1
    assert report.llm_calls[0].meta.get("error") is True
    assert report.llm_calls[0].meta.get("error_type") == "RuntimeError"


@pytest.mark.asyncio
async def test_profiler_middleware_records_tool_span_on_exception() -> None:
    profiler = AgentProfiler(enabled=True)
    profiler.start_turn()
    middleware = ProfilerMiddleware(profiler=profiler, model="test-model")

    async def _raise_tool() -> str:
        raise ValueError("tool boom")

    with pytest.raises(ValueError, match="tool boom"):
        await middleware.around_tool("echo", {"text": "x"}, _raise_tool)

    report = profiler.end_turn()
    assert len(report.tool_calls) == 1
    assert report.tool_calls[0].meta.get("args_summary") == "text=x"
    assert report.tool_calls[0].meta.get("error") is True
    assert report.tool_calls[0].meta.get("error_type") == "ValueError"
