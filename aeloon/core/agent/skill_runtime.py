"""Runtime for activating code-backed skills into a tool registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aeloon.core.agent.tools.cron import CronTool
from aeloon.core.agent.tools.registry import ToolRegistry
from aeloon.core.agent.tools.spawn import SpawnTool

if TYPE_CHECKING:
    from aeloon.core.agent.subagent import SubagentManager
    from aeloon.core.config.schema import WebSearchConfig
    from aeloon.services.cron.service import CronService


@dataclass(frozen=True)
class SkillBuildContext:
    """Explicit dependency bag used by code skills during activation."""

    workspace: Path
    web_search_config: "WebSearchConfig"
    web_proxy: str | None
    subagent_manager: "SubagentManager"
    cron_service: "CronService | None"


class CodeSkill(ABC):
    """Base contract for code skills that can register tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable skill name."""
        raise NotImplementedError

    @abstractmethod
    def activate(self, registry: ToolRegistry, ctx: SkillBuildContext) -> bool:
        """Activate this skill into the given registry. Returns whether it was installed."""
        raise NotImplementedError

    @abstractmethod
    def deactivate(self, registry: ToolRegistry) -> None:
        """Deactivate this skill from the given registry."""
        raise NotImplementedError


class SpawnCodeSkill(CodeSkill):
    """Skill that exposes background subagent spawning."""

    @property
    def name(self) -> str:
        return "spawn"

    def activate(self, registry: ToolRegistry, ctx: SkillBuildContext) -> bool:
        registry.register(SpawnTool(manager=ctx.subagent_manager))
        return True

    def deactivate(self, registry: ToolRegistry) -> None:
        registry.unregister("spawn")


class CronCodeSkill(CodeSkill):
    """Skill that exposes cron scheduling when cron service is available."""

    @property
    def name(self) -> str:
        return "cron"

    def activate(self, registry: ToolRegistry, ctx: SkillBuildContext) -> bool:
        if ctx.cron_service is None:
            return False
        registry.register(CronTool(ctx.cron_service))
        return True

    def deactivate(self, registry: ToolRegistry) -> None:
        registry.unregister("cron")


class SkillRuntime:
    """Manages lifecycle of code skills and tool registration."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        context: SkillBuildContext,
        default_active: tuple[str, ...] = ("spawn", "cron"),
    ):
        self._registry = registry
        self._context = context
        self._default_active = default_active
        self._skills: dict[str, CodeSkill] = {}
        self._active: set[str] = set()
        self.register(SpawnCodeSkill())
        self.register(CronCodeSkill())

    def register(self, skill: CodeSkill) -> None:
        """Register a code skill implementation."""
        self._skills[skill.name] = skill

    def activate(self, name: str) -> bool:
        """Activate one skill by name. Returns whether activation happened."""
        if name in self._active:
            return False
        skill = self._skills.get(name)
        if skill is None:
            return False
        if not skill.activate(self._registry, self._context):
            return False
        self._active.add(name)
        return True

    def deactivate(self, name: str) -> bool:
        """Deactivate one skill by name. Returns whether deactivation happened."""
        if name not in self._active:
            return False
        skill = self._skills.get(name)
        if skill is None:
            return False
        skill.deactivate(self._registry)
        self._active.remove(name)
        return True

    def activate_defaults(self) -> None:
        """Activate the default set of code skills."""
        for name in self._default_active:
            self.activate(name)

    @property
    def active_skills(self) -> frozenset[str]:
        """Current active skill names."""
        return frozenset(self._active)
