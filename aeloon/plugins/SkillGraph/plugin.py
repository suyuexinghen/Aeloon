"""SkillGraph compiler plugin and workflow tool registration."""

from __future__ import annotations

import asyncio
import shlex
from typing import TYPE_CHECKING

from aeloon.plugins._sdk import CommandContext, Plugin
from aeloon.plugins.SkillGraph.compiler import (
    SkillCompilerRequest,
    build_skill_compiler_parser,
    compile_skill_to_workspace,
    format_skill_compiler_success,
)
from aeloon.plugins.SkillGraph.tools import ResumeWorkflowTool, WorkflowTool
from aeloon.plugins.SkillGraph.workflow_loader import WorkflowLoader
from aeloon.plugins.SkillGraph.workflow_state import WorkflowStateStore

from .cli import build_skill_compiler_cli_builder

if TYPE_CHECKING:
    from aeloon.plugins._sdk.api import PluginAPI


class SkillGraphPlugin(Plugin):
    """Plugin that compiles skills and exposes compiled workflow tools."""

    def __init__(self) -> None:
        self._api: PluginAPI | None = None
        self._registered_tool_names: set[str] = set()

    def register(self, api: PluginAPI) -> None:
        api.register_cli(
            "skill_compiler",
            build_skill_compiler_cli_builder("skill_compiler"),
            handler=self._handle_command,
            description="Compile one skill into resumable workflow tools",
        )
        self._api = api

    async def activate(self, api: PluginAPI) -> None:
        self._api = api
        self._refresh_tools()

    async def deactivate(self) -> None:
        if self._api is None:
            return
        tools = self._api.runtime.agent_loop.tools
        for tool_name in self._registered_tool_names:
            tools.unregister(tool_name)
        self._registered_tool_names.clear()

    async def _handle_command(self, ctx: CommandContext, args: str) -> str | None:
        assert self._api is not None
        parser = build_skill_compiler_parser()
        try:
            argv = shlex.split(args)
        except ValueError as exc:
            return f"Skill compile failed: {exc}"
        try:
            parsed = parser.parse_args(argv)
        except SystemExit:
            return (
                "Usage: /skill_compiler <skill-path> [--model MODEL] "
                "[--runtime-model MODEL] [--strict-validate]"
            )

        agent_loop = self._api.runtime.agent_loop
        try:
            result = await asyncio.to_thread(
                compile_skill_to_workspace,
                workspace=agent_loop.workspace,
                provider=agent_loop.provider,
                default_model=agent_loop.model,
                request=SkillCompilerRequest(
                    skill_path=parsed.skill_path,
                    model=parsed.model or None,
                    runtime_model=parsed.runtime_model or None,
                    strict_validate=bool(parsed.strict_validate),
                ),
            )
        except Exception as exc:
            return f"Skill compile failed: {exc}"
        refreshed = self._refresh_tools()
        return format_skill_compiler_success(result, refreshed)

    def _refresh_tools(self) -> bool:
        assert self._api is not None
        agent_loop = self._api.runtime.agent_loop
        workspace = agent_loop.workspace
        loader = WorkflowLoader(workspace)
        state_store = WorkflowStateStore(workspace)
        tools = agent_loop.tools

        for tool_name in self._registered_tool_names:
            tools.unregister(tool_name)

        registered: set[str] = set()
        workflows = loader.list_workflows()
        for workflow in workflows:
            tool = WorkflowTool(
                loader=loader,
                workflow_name=workflow.name,
                provider=agent_loop.provider,
                model=agent_loop.model,
                workspace=str(workspace),
                state_store=state_store,
            )
            tools.register(tool)
            registered.add(tool.name)

        if workflows:
            resume_tool = ResumeWorkflowTool(
                loader=loader,
                provider=agent_loop.provider,
                model=agent_loop.model,
                workspace=str(workspace),
                state_store=state_store,
            )
            tools.register(resume_tool)
            registered.add(resume_tool.name)

        self._registered_tool_names = registered
        return bool(workflows)
