from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.core.agent.kernel import run_agent_kernel
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


class _SleepReadTool(Tool):
    def __init__(self, name: str, delay: float, events: list[str] | None = None):
        self._name = name
        self._delay = delay
        self._events = events if events is not None else []

    @property
    def name(self) -> str:
        return self._name

    @property
    def concurrency_mode(self) -> str:
        return "read_only"

    @property
    def description(self) -> str:
        return "Sleepy read tool"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs) -> str:
        self._events.append(f"start:{self._name}:{path}")
        await asyncio.sleep(self._delay)
        self._events.append(f"end:{self._name}:{path}")
        return f"{self._name}:{path}"


class _SleepBarrierTool(Tool):
    def __init__(self, events: list[str] | None = None):
        self._events = events if events is not None else []

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Barrier tool"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(self, command: str, **kwargs) -> str:
        self._events.append(f"start:exec:{command}")
        await asyncio.sleep(0.02)
        self._events.append(f"end:exec:{command}")
        return f"exec:{command}"


@pytest.mark.asyncio
async def test_kernel_tool_loop_preserves_reasoning_fields() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="echo", arguments={"text": "hi"})],
                reasoning_content="hidden",
                thinking_blocks=[{"type": "thinking", "thinking": "step-1"}],
            ),
            LLMResponse(
                content="done",
                tool_calls=[],
                reasoning_content="final reasoning",
                thinking_blocks=[{"type": "thinking", "thinking": "step-2"}],
            ),
        ]
    )

    tools = ToolRegistry()
    tools.register(_EchoTool())

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "hello"}],
        max_iterations=5,
    )

    assert final_content == "done"
    assert tools_used == ["echo"]

    assistant_with_tool = next(
        message
        for message in messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_with_tool["reasoning_content"] == "hidden"
    assert assistant_with_tool["thinking_blocks"] == [{"type": "thinking", "thinking": "step-1"}]

    final_assistant = messages[-1]
    assert final_assistant["role"] == "assistant"
    assert final_assistant["reasoning_content"] == "final reasoning"


@pytest.mark.asyncio
async def test_kernel_handles_error_finish_reason() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="provider error", tool_calls=[], finish_reason="error")
    )

    final_content, tools_used, _ = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=ToolRegistry(),
        messages=[{"role": "user", "content": "hello"}],
        max_iterations=3,
    )

    assert final_content == "provider error"
    assert tools_used == []


@pytest.mark.asyncio
async def test_kernel_returns_max_iteration_message() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content="keep going",
            tool_calls=[ToolCallRequest(id="call_1", name="echo", arguments={"text": "x"})],
        )
    )

    tools = ToolRegistry()
    tools.register(_EchoTool())

    final_content, tools_used, _ = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "hello"}],
        max_iterations=2,
    )

    assert "maximum number of tool call iterations (2)" in (final_content or "")
    assert tools_used == ["echo", "echo"]


@pytest.mark.asyncio
async def test_kernel_executes_non_conflicting_read_tools_in_parallel() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="parallel reads",
                tool_calls=[
                    ToolCallRequest(id="call_1", name="read_file", arguments={"path": "a.py"}),
                    ToolCallRequest(id="call_2", name="list_dir", arguments={"path": "src"}),
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    tools = ToolRegistry()
    tools.register(_SleepReadTool("read_file", 0.05))
    tools.register(_SleepReadTool("list_dir", 0.05))

    started = time.perf_counter()
    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "hello"}],
    )
    elapsed = time.perf_counter() - started

    assert final_content == "done"
    assert tools_used == ["read_file", "list_dir"]
    assert elapsed < 0.09
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assert [message["name"] for message in tool_messages] == ["read_file", "list_dir"]


@pytest.mark.asyncio
async def test_kernel_serializes_conflicting_file_operations() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="conflicting file ops",
                tool_calls=[
                    ToolCallRequest(id="call_1", name="read_file", arguments={"path": "same.py"}),
                    ToolCallRequest(id="call_2", name="edit_file", arguments={"path": "same.py"}),
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    events: list[str] = []
    tools = ToolRegistry()
    tools.register(_SleepReadTool("read_file", 0.02, events))
    tools.register(_SleepReadTool("edit_file", 0.02, events))

    final_content, tools_used, _ = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert final_content == "done"
    assert tools_used == ["read_file", "edit_file"]
    assert events == [
        "start:read_file:same.py",
        "end:read_file:same.py",
        "start:edit_file:same.py",
        "end:edit_file:same.py",
    ]


@pytest.mark.asyncio
async def test_kernel_preserves_tool_result_order_when_parallel_tasks_finish_out_of_order() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="parallel reads",
                tool_calls=[
                    ToolCallRequest(id="call_1", name="read_file", arguments={"path": "slow.py"}),
                    ToolCallRequest(id="call_2", name="list_dir", arguments={"path": "fast"}),
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    tools = ToolRegistry()
    tools.register(_SleepReadTool("read_file", 0.05))
    tools.register(_SleepReadTool("list_dir", 0.01))

    _, _, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "hello"}],
    )

    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assert [message["name"] for message in tool_messages] == ["read_file", "list_dir"]
    assert [message["content"] for message in tool_messages] == [
        "read_file:slow.py",
        "list_dir:fast",
    ]


@pytest.mark.asyncio
async def test_kernel_treats_exec_as_exclusive_barrier() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="barrier",
                tool_calls=[
                    ToolCallRequest(id="call_1", name="read_file", arguments={"path": "a.py"}),
                    ToolCallRequest(id="call_2", name="exec", arguments={"command": "pytest"}),
                    ToolCallRequest(id="call_3", name="list_dir", arguments={"path": "src"}),
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    events: list[str] = []
    tools = ToolRegistry()
    tools.register(_SleepReadTool("read_file", 0.01, events))
    tools.register(_SleepBarrierTool(events))
    tools.register(_SleepReadTool("list_dir", 0.01, events))

    final_content, tools_used, messages = await run_agent_kernel(
        provider=provider,
        model="test-model",
        tools=tools,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert final_content == "done"
    assert tools_used == ["read_file", "exec", "list_dir"]
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assert [message["name"] for message in tool_messages] == ["read_file", "exec", "list_dir"]
    assert events == [
        "start:read_file:a.py",
        "end:read_file:a.py",
        "start:exec:pytest",
        "end:exec:pytest",
        "start:list_dir:src",
        "end:list_dir:src",
    ]
