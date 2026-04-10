"""Internal task-graph planning for tool-call execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from aeloon.providers.base import ToolCallRequest

if TYPE_CHECKING:
    from aeloon.core.agent.tools.registry import ToolRegistry


class TaskState(str, Enum):
    """Lifecycle state for an internal tool-execution task."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ResourceSpec:
    """Static resource hint used for conflict analysis."""

    kind: str
    key: str
    access: str


@dataclass
class TaskNode:
    """Internal representation of a single tool call in one agent turn."""

    index: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    mode: str
    resources: list[ResourceSpec]
    deps: set[int] = field(default_factory=set)
    dependents: set[int] = field(default_factory=set)
    state: TaskState = TaskState.PENDING
    result: str | None = None
    error: str | None = None


def _normalize_path(path: str) -> str:
    p = Path(path).expanduser()
    try:
        return str(p.resolve(strict=False))
    except Exception:
        return str(p)


def _extract_resources(
    tool_name: str,
    args: dict[str, Any],
    tools: "ToolRegistry",
) -> list[ResourceSpec]:
    path = args.get("path")
    if isinstance(path, str):
        tool = tools.get(tool_name)
        if tool is not None and hasattr(tool, "_resolve"):
            try:
                resolved = getattr(tool, "_resolve")(path)
                return [
                    ResourceSpec(
                        "fs",
                        str(resolved),
                        "write" if tool_name in {"write_file", "edit_file"} else "read",
                    )
                ]
            except Exception:
                pass
        return [
            ResourceSpec(
                "fs",
                _normalize_path(path),
                "write" if tool_name in {"write_file", "edit_file"} else "read",
            )
        ]

    if tool_name == "web_fetch":
        url = args.get("url")
        if isinstance(url, str):
            host = urlparse(url).netloc or url
            return [ResourceSpec("network", host, "read")]

    if tool_name == "web_search":
        return [ResourceSpec("network", "search", "read")]

    if tool_name == "exec":
        return [ResourceSpec("process", "shell", "exclusive")]
    if tool_name == "message":
        return [ResourceSpec("message", "outbound", "exclusive")]
    if tool_name == "spawn":
        return [ResourceSpec("subagent", "spawn", "exclusive")]
    if tool_name == "cron":
        return [ResourceSpec("scheduler", "cron", "exclusive")]

    return [ResourceSpec("unknown", tool_name, "exclusive")]


def _conflicts(left: TaskNode, right: TaskNode) -> bool:
    if left.mode == "exclusive" or right.mode == "exclusive":
        return True

    for lhs in left.resources:
        for rhs in right.resources:
            if lhs.kind != rhs.kind or lhs.key != rhs.key:
                continue
            if lhs.access == "read" and rhs.access == "read":
                continue
            return True
    return False


def build_task_graph(tool_calls: list[ToolCallRequest], tools: "ToolRegistry") -> list[TaskNode]:
    """Compile one LLM tool-call batch into an internal conflict graph."""
    nodes: list[TaskNode] = []
    for index, tool_call in enumerate(tool_calls):
        tool = tools.get(tool_call.name)
        mode = tool.concurrency_mode if tool is not None else "exclusive"
        nodes.append(
            TaskNode(
                index=index,
                call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                mode=mode,
                resources=_extract_resources(tool_call.name, tool_call.arguments, tools),
            )
        )

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if _conflicts(nodes[i], nodes[j]):
                nodes[j].deps.add(i)
                nodes[i].dependents.add(j)

    return nodes
