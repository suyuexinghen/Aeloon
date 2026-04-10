"""Agent core module."""

from __future__ import annotations

from typing import Any

__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore", "SkillsLoader"]


def __getattr__(name: str) -> Any:
    """Lazily expose core agent symbols without eager import cycles."""
    if name == "AgentLoop":
        from aeloon.core.agent.loop import AgentLoop

        return AgentLoop
    if name == "ContextBuilder":
        from aeloon.core.agent.context import ContextBuilder

        return ContextBuilder
    if name == "MemoryStore":
        from aeloon.core.agent.memory import MemoryStore

        return MemoryStore
    if name == "SkillsLoader":
        from aeloon.core.agent.skills import SkillsLoader

        return SkillsLoader
    raise AttributeError(name)
