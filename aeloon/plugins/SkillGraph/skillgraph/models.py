"""Data models for SkillGraph: the intermediate representation between SKILL.md and LangGraph code."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class StepType(str, Enum):
    TOOL_CALL = "tool_call"
    LLM_GENERATE = "llm_generate"
    CONDITION = "condition"
    DATA_TRANSFORM = "data_transform"


class ExecutionKind(str, Enum):
    SHELL = "shell"
    PYTHON = "python"
    LLM = "llm"
    NOOP = "noop"


class IOField(BaseModel):
    name: str
    description: str = ""
    type: str = "string"
    required: bool = True


class SourceRef(BaseModel):
    path: str
    line: int | None = None
    snippet: str = ""
    score: float = 0.0


class ExecutionSpec(BaseModel):
    kind: ExecutionKind = ExecutionKind.SHELL
    command: str = ""
    argv: list[str] = Field(default_factory=list)
    arg_bindings: dict[str, str] = Field(default_factory=dict)
    cwd: str = ""
    timeout_sec: int = 60
    env: dict[str, str] = Field(default_factory=dict)
    parser: str = "raw"


class GuardSpec(BaseModel):
    kind: str = "env_flag"
    env_var: str = ""
    expected_value: str = ""
    message: str = ""


class Step(BaseModel):
    id: str
    name: str
    description: str = ""
    step_type: StepType
    inputs: list[IOField] = Field(default_factory=list)
    outputs: list[IOField] = Field(default_factory=list)
    cacheable: bool = True
    execution_spec: ExecutionSpec | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    risk_level: str = ""
    guards: list[GuardSpec] = Field(default_factory=list)


class Edge(BaseModel):
    from_step: str
    to_step: str
    description: str = ""


class SkillGraph(BaseModel):
    """The DAG intermediate representation of a skill."""

    skill_name: str
    skill_description: str = ""
    skill_version: str = ""
    steps: list[Step] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    global_inputs: list[IOField] = Field(default_factory=list)
    global_outputs: list[IOField] = Field(default_factory=list)
    analyzer_model: str = ""

    def get_step(self, step_id: str) -> Step | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def upstream_of(self, step_id: str) -> list[str]:
        return [e.from_step for e in self.edges if e.to_step == step_id]

    def downstream_of(self, step_id: str) -> list[str]:
        return [e.to_step for e in self.edges if e.from_step == step_id]

    def entry_nodes(self) -> list[str]:
        upstream_map = {s.id: self.upstream_of(s.id) for s in self.steps}
        return [sid for sid, ups in upstream_map.items() if not ups]

    def exit_nodes(self) -> list[str]:
        downstream_map = {s.id: self.downstream_of(s.id) for s in self.steps}
        return [sid for sid, downs in downstream_map.items() if not downs]

    def topological_layers(self) -> list[list[str]]:
        if not self.steps:
            return []
        in_degree = {s.id: 0 for s in self.steps}
        adj = {s.id: [] for s in self.steps}
        for e in self.edges:
            if e.to_step in in_degree:
                in_degree[e.to_step] += 1
            if e.from_step in adj:
                adj[e.from_step].append(e.to_step)
        layers = []
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        while queue:
            layers.append(sorted(queue))
            nxt = []
            for sid in queue:
                for nb in adj.get(sid, []):
                    in_degree[nb] -= 1
                    if in_degree[nb] == 0:
                        nxt.append(nb)
            queue = nxt
        return layers

    def validate(self) -> list[str]:
        errors = []
        step_ids = {s.id for s in self.steps}
        if len(step_ids) != len(self.steps):
            errors.append("Duplicate step ids detected")
        for e in self.edges:
            if e.from_step not in step_ids:
                errors.append(f"Unknown edge source: {e.from_step}")
            if e.to_step not in step_ids:
                errors.append(f"Unknown edge target: {e.to_step}")
        covered = {sid for layer in self.topological_layers() for sid in layer}
        uncovered = step_ids - covered
        if uncovered:
            errors.append(f"Cycle detected: {uncovered}")
        return errors

    def grounded_tool_step_count(self) -> int:
        return sum(
            1
            for s in self.steps
            if s.step_type == StepType.TOOL_CALL and s.execution_spec and s.execution_spec.command
        )

    def grounded_tool_step_ratio(self) -> float:
        total = sum(1 for s in self.steps if s.step_type == StepType.TOOL_CALL)
        if total == 0:
            return 1.0
        return self.grounded_tool_step_count() / float(total)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SkillGraph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


# ── Runtime dependency models ───────────────────────────────


class RuntimeDependency(BaseModel):
    """A single external resource required at workflow runtime."""

    kind: str  # cli_binary, script_file, env_var, python_package, npm_runtime, llm_service
    name: str
    required: bool = True
    source_step_ids: list[str] = Field(default_factory=list)
    check_command: str = ""  # e.g. "which browser", "test -f scripts/openclaw.sh"


class RuntimeManifest(BaseModel):
    """Structured declaration of everything a compiled workflow needs to run."""

    dependencies: list[RuntimeDependency] = Field(default_factory=list)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RuntimeManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


class SandboxCheck(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class SandboxBootstrapResult(BaseModel):
    skill_slug: str
    sandbox_dir: str
    status: str  # ready | failed
    checks: list[SandboxCheck] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SandboxBootstrapResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
