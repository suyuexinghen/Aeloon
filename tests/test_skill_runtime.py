from __future__ import annotations

from unittest.mock import MagicMock

from aeloon.core.agent.skill_runtime import SkillBuildContext, SkillRuntime
from aeloon.core.agent.tools.factory import register_core_tools
from aeloon.core.agent.tools.registry import ToolRegistry
from aeloon.core.config.schema import ExecToolConfig, WebSearchConfig


def _build_context(tmp_path, *, cron_service):
    return SkillBuildContext(
        workspace=tmp_path,
        web_search_config=WebSearchConfig(),
        web_proxy=None,
        subagent_manager=MagicMock(),
        cron_service=cron_service,
    )


def test_skill_build_context_has_explicit_fields(tmp_path) -> None:
    ctx = _build_context(tmp_path, cron_service=None)

    assert ctx.workspace == tmp_path
    assert isinstance(ctx.web_search_config, WebSearchConfig)
    assert ctx.web_proxy is None
    assert ctx.subagent_manager is not None
    assert ctx.cron_service is None


def test_skill_runtime_activate_defaults_without_cron_service(tmp_path) -> None:
    registry = ToolRegistry()
    runtime = SkillRuntime(registry=registry, context=_build_context(tmp_path, cron_service=None))

    runtime.activate_defaults()

    assert registry.has("spawn")
    assert not registry.has("cron")
    assert runtime.active_skills == frozenset({"spawn"})


def test_skill_runtime_activate_and_deactivate_with_cron(tmp_path) -> None:
    registry = ToolRegistry()
    runtime = SkillRuntime(
        registry=registry,
        context=_build_context(tmp_path, cron_service=MagicMock()),
    )

    runtime.activate_defaults()
    assert registry.has("spawn")
    assert registry.has("cron")
    assert runtime.active_skills == frozenset({"spawn", "cron"})

    assert runtime.deactivate("spawn") is True
    assert not registry.has("spawn")
    assert runtime.active_skills == frozenset({"cron"})

    assert runtime.activate("spawn") is True
    assert registry.has("spawn")
    assert runtime.active_skills == frozenset({"spawn", "cron"})


def test_web_tools_still_registered_by_core_factory(tmp_path) -> None:
    registry = ToolRegistry()
    register_core_tools(
        registry,
        workspace=tmp_path,
        restrict_to_workspace=False,
        exec_config=ExecToolConfig(),
        web_search_config=WebSearchConfig(),
        web_proxy=None,
    )

    runtime = SkillRuntime(registry=registry, context=_build_context(tmp_path, cron_service=None))
    runtime.activate_defaults()

    assert registry.has("web_search")
    assert registry.has("web_fetch")
    assert registry.has("spawn")


def test_skill_runtime_double_activate_is_idempotent(tmp_path) -> None:
    registry = ToolRegistry()
    runtime = SkillRuntime(
        registry=registry,
        context=_build_context(tmp_path, cron_service=MagicMock()),
    )

    assert runtime.activate("spawn") is True
    assert runtime.activate("spawn") is False
    assert runtime.active_skills == frozenset({"spawn"})
