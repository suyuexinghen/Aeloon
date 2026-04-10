"""Reusable LLM<->tool iteration kernel."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from aeloon.core.agent.middleware import AgentMiddleware
from aeloon.core.agent.task_graph import TaskNode, TaskState, build_task_graph
from aeloon.providers.base import ToolCallRequest
from aeloon.utils.helpers import build_assistant_message

if TYPE_CHECKING:
    from aeloon.core.agent.tools.registry import ToolRegistry
    from aeloon.plugins._sdk.hooks import HookDispatcher
    from aeloon.providers.base import LLMProvider, LLMResponse


def _default_strip_think(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None


def _default_tool_hint(tool_calls: list[ToolCallRequest]) -> str:
    def _fmt(tc: ToolCallRequest) -> str:
        args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
        val = next(iter(args.values()), None) if isinstance(args, dict) else None
        if not isinstance(val, str):
            return tc.name
        return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

    return ", ".join(_fmt(tc) for tc in tool_calls)


def _default_add_assistant_message(
    messages: list[dict],
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> list[dict]:
    messages.append(
        build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    return messages


def _default_add_tool_result(
    messages: list[dict],
    tool_call_id: str,
    tool_name: str,
    result: str,
) -> list[dict]:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }
    )
    return messages


async def _call_llm_with_middlewares(
    *,
    middlewares: list[AgentMiddleware],
    messages: list[dict],
    tool_defs: list[dict],
    call_llm: Callable[[list[dict], list[dict]], Awaitable["LLMResponse"]],
) -> "LLMResponse":
    async def _invoke(index: int, msgs: list[dict], defs: list[dict]) -> "LLMResponse":
        if index >= len(middlewares):
            return await call_llm(msgs, defs)

        middleware = middlewares[index]

        async def _next(next_messages: list[dict], next_defs: list[dict]) -> "LLMResponse":
            return await _invoke(index + 1, next_messages, next_defs)

        return await middleware.around_llm(msgs, defs, _next)

    return await _invoke(0, messages, tool_defs)


async def _call_tool_with_middlewares(
    *,
    middlewares: list[AgentMiddleware],
    name: str,
    args: dict | list | None,
    execute: Callable[[], Awaitable[str]],
) -> str:
    async def _invoke(index: int) -> str:
        if index >= len(middlewares):
            return await execute()

        middleware = middlewares[index]

        async def _next() -> str:
            return await _invoke(index + 1)

        return await middleware.around_tool(name, args, _next)

    return await _invoke(0)


async def _execute_tool_batch(
    *,
    tool_calls: list[ToolCallRequest],
    tools: "ToolRegistry",
    middlewares: list[AgentMiddleware],
    hook_dispatcher: "HookDispatcher | None" = None,
) -> list[TaskNode]:
    tracer = None
    if os.environ.get("AELOON_TRACE_TOOL_BATCH") == "1":
        try:
            from viztracer import VizTracer

            tracer = VizTracer()
            tracer.start()
        except Exception as exc:
            logger.warning("Failed to start VizTracer for tool batch tracing: {}", exc)

    nodes = build_task_graph(tool_calls, tools)
    pending = {node.index: node for node in nodes}
    running: dict[int, asyncio.Task[str]] = {}

    async def _execute_node(node: TaskNode) -> str:
        async def _do_tool_call() -> str:
            return await tools.execute(node.tool_name, node.arguments)

        # Dispatch BEFORE_TOOL_CALL hook (guard mode — can block or modify)
        if hook_dispatcher is not None:
            try:
                from aeloon.plugins._sdk.hooks import HookEvent

                decision = await hook_dispatcher.dispatch_guard(
                    HookEvent.BEFORE_TOOL_CALL,
                    value=node.arguments,
                    match_value=node.tool_name,
                    tool_name=node.tool_name,
                    tool_call_id=node.call_id,
                )
                if not decision.allow:
                    node.result = f"Error: Tool call blocked — {decision.reason}"
                    node.state = TaskState.DONE
                    return node.result
                if decision.modified_value is not None:
                    node.arguments = decision.modified_value
            except Exception:
                logger.opt(exception=True).debug("BEFORE_TOOL_CALL guard dispatch failed")

        result = await _call_tool_with_middlewares(
            middlewares=middlewares,
            name=node.tool_name,
            args=node.arguments,
            execute=_do_tool_call,
        )

        # Dispatch AFTER_TOOL_CALL hook (fire-and-forget)
        if hook_dispatcher is not None:
            try:
                from aeloon.plugins._sdk.hooks import HookEvent

                await hook_dispatcher.dispatch_notify(
                    HookEvent.AFTER_TOOL_CALL,
                    match_value=node.tool_name,
                    tool_name=node.tool_name,
                    result=result,
                    tool_call_id=node.call_id,
                )
            except Exception:
                logger.opt(exception=True).debug("AFTER_TOOL_CALL hook dispatch failed")

        # Dispatch TOOL_CALL_FAILURE hook if result indicates an error
        if result.startswith("Error") and hook_dispatcher is not None:
            try:
                await hook_dispatcher.dispatch_notify(
                    HookEvent.TOOL_CALL_FAILURE,
                    match_value=node.tool_name,
                    tool_name=node.tool_name,
                    error=result,
                    tool_call_id=node.call_id,
                )
            except Exception:
                logger.opt(exception=True).debug("TOOL_CALL_FAILURE hook dispatch failed")

        return result

    try:
        while pending or running:
            ready = [node for node in pending.values() if not node.deps]
            for node in ready:
                node.state = TaskState.RUNNING
                running[node.index] = asyncio.create_task(_execute_node(node))
                pending.pop(node.index)

            if not running:
                raise RuntimeError("deadlock detected in tool task graph")

            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            finished_indexes = [index for index, task in running.items() if task in done]

            for index in finished_indexes:
                task = running.pop(index)
                node = nodes[index]
                try:
                    node.result = await task
                    node.state = TaskState.DONE
                except asyncio.CancelledError:
                    node.state = TaskState.CANCELLED
                    raise
                except Exception as exc:
                    node.state = TaskState.FAILED
                    node.error = str(exc)
                    node.result = f"Error executing {node.tool_name}: {exc}"

                for dependent_index in node.dependents:
                    nodes[dependent_index].deps.discard(index)
    except asyncio.CancelledError:
        for task in running.values():
            task.cancel()
        await asyncio.gather(*running.values(), return_exceptions=True)
        raise
    finally:
        if tracer is not None:
            try:
                tracer.stop()
                output_path = os.environ.get(
                    "AELOON_TRACE_TOOL_BATCH_OUTPUT",
                    f"trace-tool-batch-{int(time.time() * 1000)}.json",
                )
                tracer.save(output_path)
                logger.info("Saved tool batch trace to {}", output_path)
            except Exception as exc:
                logger.warning("Failed to save VizTracer tool batch trace: {}", exc)

    return nodes


async def run_agent_kernel(
    *,
    provider: "LLMProvider",
    model: str,
    tools: "ToolRegistry",
    messages: list[dict],
    max_iterations: int = 25,
    middlewares: list[AgentMiddleware] | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
    add_assistant_message: Callable[..., list[dict]] | None = None,
    add_tool_result: Callable[[list[dict], str, str, str], list[dict]] | None = None,
    strip_think: Callable[[str | None], str | None] | None = None,
    tool_hint: Callable[[list[ToolCallRequest]], str] | None = None,
    hook_dispatcher: "HookDispatcher | None" = None,
) -> tuple[str | None, list[str], list[dict]]:
    """Execute a reusable tool-augmented LLM loop."""
    add_assistant = add_assistant_message or _default_add_assistant_message
    add_tool = add_tool_result or _default_add_tool_result
    _strip = strip_think or _default_strip_think
    _tool_hint = tool_hint or _default_tool_hint
    _middlewares = middlewares or []

    iteration = 0
    final_content = None
    tools_used: list[str] = []

    while iteration < max_iterations:
        iteration += 1

        tool_defs = tools.get_definitions()

        if on_progress:
            await on_progress(
                "Thinking..." if iteration == 1 else f"Thinking (step {iteration})..."
            )

        async def _do_llm_call(
            current_messages: list[dict], current_tool_defs: list[dict]
        ) -> "LLMResponse":
            return await provider.chat_with_retry(
                messages=current_messages,
                tools=current_tool_defs,
                model=model,
            )

        # Dispatch BEFORE_LLM_CALL hook
        if hook_dispatcher is not None:
            try:
                from aeloon.plugins._sdk.hooks import HookEvent

                await hook_dispatcher.dispatch_notify(
                    HookEvent.BEFORE_LLM_CALL,
                    messages=messages,
                    model=model,
                    tool_count=len(tool_defs),
                )
            except Exception:
                logger.opt(exception=True).debug("BEFORE_LLM_CALL hook dispatch failed")

        response = await _call_llm_with_middlewares(
            middlewares=_middlewares,
            messages=messages,
            tool_defs=tool_defs,
            call_llm=_do_llm_call,
        )

        # Dispatch AFTER_LLM_CALL hook
        if hook_dispatcher is not None:
            try:
                from aeloon.plugins._sdk.hooks import HookEvent

                await hook_dispatcher.dispatch_notify(
                    HookEvent.AFTER_LLM_CALL,
                    content=response.content,
                    has_tool_calls=response.has_tool_calls,
                    finish_reason=response.finish_reason,
                )
            except Exception:
                logger.opt(exception=True).debug("AFTER_LLM_CALL hook dispatch failed")

        if response.has_tool_calls:
            if on_progress:
                thought = _strip(response.content)
                if thought:
                    await on_progress(thought)
                hint = _strip(_tool_hint(response.tool_calls))
                if hint:
                    await on_progress(hint, tool_hint=True)

            tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
            messages = add_assistant(
                messages,
                response.content,
                tool_calls=tool_call_dicts,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )

            for tool_call in response.tool_calls:
                args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                logger.info("Tool call: {}({})", tool_call.name, args_str[:200])

            executed_nodes = await _execute_tool_batch(
                tool_calls=response.tool_calls,
                tools=tools,
                middlewares=_middlewares,
                hook_dispatcher=hook_dispatcher,
            )
            for node in sorted(executed_nodes, key=lambda item: item.index):
                tools_used.append(node.tool_name)
                messages = add_tool(
                    messages,
                    node.call_id,
                    node.tool_name,
                    node.result or f"Error executing {node.tool_name}: unknown failure",
                )
            continue

        clean = _strip(response.content)
        if response.finish_reason == "error":
            logger.error("LLM returned error: {}", (clean or "")[:200])
            final_content = clean or "Sorry, I encountered an error calling the AI model."
            break

        messages = add_assistant(
            messages,
            clean,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        final_content = clean
        break

    if final_content is None and iteration >= max_iterations:
        logger.warning("Max iterations ({}) reached", max_iterations)
        final_content = (
            f"I reached the maximum number of tool call iterations ({max_iterations}) "
            "without completing the task. You can try breaking the task into smaller steps."
        )

    return final_content, tools_used, messages
