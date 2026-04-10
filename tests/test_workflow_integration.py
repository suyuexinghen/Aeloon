from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aeloon.core.agent.context import ContextBuilder
from aeloon.core.agent.tools.registry import ToolRegistry
from aeloon.core.agent.turn import TurnContext
from aeloon.plugins.SkillGraph.tools import ResumeWorkflowTool, WorkflowTool
from aeloon.plugins.SkillGraph.workflow_loader import WorkflowLoader
from aeloon.plugins.SkillGraph.workflow_state import WorkflowStateStore
from aeloon.providers.base import LLMResponse


def _write_demo_workflow(workspace: Path) -> Path:
    compiled = workspace / "compiled_skills"
    compiled.mkdir(parents=True, exist_ok=True)
    workflow_path = compiled / "demo_workflow.py"
    workflow_path.write_text(
        """
SKILL_META = {
    "name": "demo",
    "description": "Demo compiled workflow",
    "global_inputs": [{"name": "query", "type": "string", "required": True, "description": "Search query"}],
}

def preflight_check(project_dir: str):
    return []

class _Graph:
    def invoke(self, state):
        text = state["global_inputs"].get("query", "")
        return {
            "global_inputs": state["global_inputs"],
            "step_results": {"demo_step": {"query": text}},
            "error": None,
            "final_output": f"demo:{text}",
        }

def build_graph():
    return _Graph()
""",
        encoding="utf-8",
    )
    (compiled / "demo_workflow.manifest.json").write_text(
        json.dumps({"dependencies": []}), encoding="utf-8"
    )
    sandbox = compiled / "demo_workflow.sandbox"
    (sandbox / "skill").mkdir(parents=True, exist_ok=True)
    (sandbox / "bootstrap.json").write_text(
        json.dumps({"status": "ready", "checks": [], "env": {}}), encoding="utf-8"
    )
    (sandbox / "env.json").write_text(json.dumps({}), encoding="utf-8")
    (compiled / "skill_config.json").write_text(
        json.dumps({"runtime": {"model": "test-model", "base_url": "https://example.com"}}),
        encoding="utf-8",
    )
    return workflow_path


def _write_blocking_workflow(workspace: Path) -> Path:
    compiled = workspace / "compiled_skills"
    compiled.mkdir(parents=True, exist_ok=True)
    workflow_path = compiled / "blocking_workflow.py"
    workflow_path.write_text(
        """
SKILL_META = {"name": "blocking", "description": "Blocks until configured", "global_inputs": []}

def preflight_check(project_dir: str):
    return []

class _Graph:
    def invoke(self, state):
        step_results = dict(state.get("step_results") or {})
        step_results.setdefault("prepare", {"ok": True})
        if not state["global_inputs"].get("configured"):
            return {
                "global_inputs": state["global_inputs"],
                "step_results": step_results,
                "error": "configuration missing",
                "final_output": None,
            }
        step_results["resume"] = {"configured": True}
        return {
            "global_inputs": state["global_inputs"],
            "step_results": step_results,
            "error": None,
            "final_output": "workflow-complete",
        }

def build_graph():
    return _Graph()
""",
        encoding="utf-8",
    )
    (compiled / "blocking_workflow.manifest.json").write_text(
        json.dumps({"dependencies": []}), encoding="utf-8"
    )
    sandbox = compiled / "blocking_workflow.sandbox"
    (sandbox / "skill").mkdir(parents=True, exist_ok=True)
    (sandbox / "bootstrap.json").write_text(
        json.dumps({"status": "ready", "checks": [], "env": {}}), encoding="utf-8"
    )
    (sandbox / "env.json").write_text(json.dumps({}), encoding="utf-8")
    return workflow_path


def _write_resumable_workflow(workspace: Path) -> Path:
    compiled = workspace / "compiled_skills"
    compiled.mkdir(parents=True, exist_ok=True)
    workflow_path = compiled / "resumable_workflow.py"
    workflow_path.write_text(
        """
SKILL_META = {"name": "resumable", "description": "Uses explicit resume API", "global_inputs": []}

def preflight_check(project_dir: str):
    return []

class _Graph:
    def invoke(self, state):
        return {"global_inputs": state["global_inputs"], "step_results": {"fallback": True}, "error": None, "final_output": "fallback"}

def build_graph():
    return _Graph()

def run_until_blocked(state):
    return {
        "global_inputs": state["global_inputs"],
        "step_results": {"prepare": {"ran": True}},
        "error": "need user fix",
        "final_output": None,
    }

def resume_from_state(state):
    results = dict(state.get("step_results") or {})
    results["resumed"] = {"ok": bool(state["global_inputs"].get("configured"))}
    return {
        "global_inputs": state["global_inputs"],
        "step_results": results,
        "error": None,
        "final_output": "resumed-ok",
    }
""",
        encoding="utf-8",
    )
    (compiled / "resumable_workflow.manifest.json").write_text(
        json.dumps({"dependencies": []}), encoding="utf-8"
    )
    sandbox = compiled / "resumable_workflow.sandbox"
    (sandbox / "skill").mkdir(parents=True, exist_ok=True)
    (sandbox / "bootstrap.json").write_text(
        json.dumps({"status": "ready", "checks": [], "env": {}}), encoding="utf-8"
    )
    (sandbox / "env.json").write_text(json.dumps({}), encoding="utf-8")
    return workflow_path


def _write_envelope_workflow(workspace: Path) -> Path:
    compiled = workspace / "compiled_skills"
    compiled.mkdir(parents=True, exist_ok=True)
    workflow_path = compiled / "envelope_workflow.py"
    workflow_path.write_text(
        """
SKILL_META = {"name": "envelope", "description": "Returns workflow envelopes", "global_inputs": []}

def preflight_check(project_dir: str):
    return []

class _Graph:
    def invoke(self, state):
        return {"status": "failed", "error": "should not use graph.invoke"}

def build_graph():
    return _Graph()

def run_until_blocked(state):
    return {
        "status": "blocked",
        "current_step": "prepare",
        "global_inputs": state.get("global_inputs", {}),
        "step_results": {"prepare": {"done": True}},
        "block": {
            "type": "user_input_required",
            "message": "Need confirmation",
            "details": {"question": "continue?"},
            "suggested_actions": ["Collect user confirmation and then resume_workflow."],
        },
    }

def resume_from_state(state):
    results = dict(state.get("step_results") or {})
    results["finish"] = {"confirmed": bool(state.get("global_inputs", {}).get("confirmed"))}
    return {
        "status": "completed",
        "current_step": None,
        "global_inputs": state.get("global_inputs", {}),
        "step_results": results,
        "final_output": "envelope-complete",
    }
""",
        encoding="utf-8",
    )
    (compiled / "envelope_workflow.manifest.json").write_text(
        json.dumps({"dependencies": []}), encoding="utf-8"
    )
    sandbox = compiled / "envelope_workflow.sandbox"
    (sandbox / "skill").mkdir(parents=True, exist_ok=True)
    (sandbox / "bootstrap.json").write_text(
        json.dumps({"status": "ready", "checks": [], "env": {}}), encoding="utf-8"
    )
    (sandbox / "env.json").write_text(json.dumps({}), encoding="utf-8")
    return workflow_path


def _build_registry(tmp_path: Path) -> tuple[ToolRegistry, WorkflowStateStore]:
    registry = ToolRegistry()
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="provider result"))
    loader = WorkflowLoader(tmp_path)
    store = WorkflowStateStore(tmp_path)
    for workflow in loader.list_workflows():
        registry.register(
            WorkflowTool(
                loader=loader,
                workflow_name=workflow.name,
                provider=provider,
                model="test-model",
                workspace=str(tmp_path),
                state_store=store,
            )
        )
    if loader.list_workflows():
        registry.register(
            ResumeWorkflowTool(
                loader=loader,
                provider=provider,
                model="test-model",
                workspace=str(tmp_path),
                state_store=store,
            )
        )
    registry.notify_turn_start(TurnContext(channel="cli", chat_id="chat", session_key="cli:chat"))
    return registry, store


def test_workflow_loader_discovers_compiled_workflows(tmp_path: Path) -> None:
    _write_demo_workflow(tmp_path)
    loader = WorkflowLoader(tmp_path)

    workflows = loader.list_workflows()
    assert [workflow.name for workflow in workflows] == ["demo"]
    assert workflows[0].global_inputs[0]["name"] == "query"
    assert "compiled-workflows" in loader.build_summary()
    assert loader.get_workflow(" demo ").metadata.name == "demo"


@pytest.mark.asyncio
async def test_workflow_tool_runs_compiled_workflow(tmp_path: Path) -> None:
    _write_demo_workflow(tmp_path)
    registry, _ = _build_registry(tmp_path)

    result = await registry.execute("run_demo", {"inputs": {"query": "hello"}})
    payload = json.loads(result)
    assert payload["status"] == "completed"
    assert payload["final_output"] == "demo:hello"


@pytest.mark.asyncio
async def test_workflow_block_and_resume_flow(tmp_path: Path) -> None:
    _write_blocking_workflow(tmp_path)
    registry, store = _build_registry(tmp_path)

    first = await registry.execute("run_blocking", {"inputs": {}})
    first_payload = json.loads(first)
    assert first_payload["status"] == "blocked"
    run_id = first_payload["workflow_run_id"]
    saved = store.load("cli:chat", run_id)
    assert saved is not None
    assert saved.status == "blocked"
    assert saved.graph_state["step_results"]["prepare"]["ok"] is True

    second = await registry.execute(
        "resume_workflow",
        {"workflow_run_id": run_id, "inputs": {"configured": True}},
    )
    second_payload = json.loads(second)
    assert second_payload["status"] == "completed"
    assert second_payload["final_output"] == "workflow-complete"
    assert second_payload["step_results"]["prepare"]["ok"] is True
    assert second_payload["step_results"]["resume"]["configured"] is True


@pytest.mark.asyncio
async def test_workflow_tool_preflight_rejects_missing_dependencies(tmp_path: Path) -> None:
    compiled = tmp_path / "compiled_skills"
    compiled.mkdir(parents=True, exist_ok=True)
    (compiled / "strict_workflow.py").write_text(
        """
SKILL_META = {"name": "strict", "description": "Needs preflight", "global_inputs": []}

def preflight_check(project_dir: str):
    return ["CLI tool not found: nonexistent_tool_xyz"]

class _Graph:
    def invoke(self, state):
        return {"global_inputs": state["global_inputs"], "step_results": {}, "error": None, "final_output": "ok"}

def build_graph():
    return _Graph()
""",
        encoding="utf-8",
    )
    (compiled / "strict_workflow.manifest.json").write_text(
        json.dumps({"dependencies": []}), encoding="utf-8"
    )
    sandbox = compiled / "strict_workflow.sandbox"
    (sandbox / "skill").mkdir(parents=True, exist_ok=True)
    (sandbox / "bootstrap.json").write_text(
        json.dumps({"status": "ready", "checks": [], "env": {}}), encoding="utf-8"
    )
    (sandbox / "env.json").write_text(json.dumps({}), encoding="utf-8")

    registry, _ = _build_registry(tmp_path)
    result = await registry.execute("run_strict", {"inputs": {}})
    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert payload["block"]["message"].startswith("Workflow preflight failed")
    assert any("nonexistent_tool_xyz" in item for item in payload["block"]["details"]["preflight"])


@pytest.mark.asyncio
async def test_resume_workflow_uses_latest_blocked_when_id_omitted(tmp_path: Path) -> None:
    _write_resumable_workflow(tmp_path)
    registry, _ = _build_registry(tmp_path)

    first = await registry.execute("run_resumable", {"inputs": {}})
    first_payload = json.loads(first)
    assert first_payload["status"] == "blocked"

    resumed = await registry.execute("resume_workflow", {"inputs": {"configured": True}})
    resumed_payload = json.loads(resumed)
    assert resumed_payload["status"] == "completed"
    assert resumed_payload["final_output"] == "resumed-ok"
    assert resumed_payload["step_results"]["resumed"]["ok"] is True


@pytest.mark.asyncio
async def test_workflow_tool_prefers_resumable_module_api(tmp_path: Path) -> None:
    _write_envelope_workflow(tmp_path)
    registry, _ = _build_registry(tmp_path)

    first = await registry.execute("run_envelope", {"inputs": {}})
    first_payload = json.loads(first)
    assert first_payload["status"] == "blocked"
    assert first_payload["current_step"] == "prepare"
    run_id = first_payload["workflow_run_id"]

    second = await registry.execute(
        "resume_workflow",
        {"workflow_run_id": run_id, "inputs": {"confirmed": True}},
    )
    second_payload = json.loads(second)
    assert second_payload["status"] == "completed"
    assert second_payload["final_output"] == "envelope-complete"
    assert second_payload["step_results"]["finish"]["confirmed"] is True


def test_context_builder_includes_workflow_summary(tmp_path: Path) -> None:
    _write_demo_workflow(tmp_path)
    builder = ContextBuilder(tmp_path)

    prompt = builder.build_system_prompt(session_key="cli:chat")
    assert "# Compiled Workflows" in prompt
    assert "demo" in prompt
    assert "run_demo" in prompt
