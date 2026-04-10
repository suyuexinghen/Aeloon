"""Compile report generation for SkillGraph builds."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .manifest import extract_manifest
from .models import RuntimeManifest, SkillGraph, StepType
from .package import SkillPackage
from .validator import ValidationResult


class CompileReport(BaseModel):
    generated_at: str
    skill_slug: str
    skill_root: str
    entry_skill: str
    package_hash: str
    output_path: str
    compilability_kind: str = ""
    compilation_strategy: str = ""
    compilation_confidence: str = ""
    compilability_reason: str = ""
    compilability_signals: list[str] = Field(default_factory=list)
    steps_total: int
    edges_total: int
    tool_steps: int
    llm_steps: int
    grounded_tool_steps: int
    grounded_tool_ratio: float
    argv_backed_tool_steps: int
    binding_backed_tool_steps: int
    shell_string_only_steps: int
    unresolved_tool_steps: int
    high_risk_steps: int
    guarded_high_risk_steps: int
    ungrounded_tool_step_ids: list[str] = Field(default_factory=list)
    runtime_dependencies: list[dict] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def build_report(
    graph: SkillGraph,
    package: SkillPackage,
    output_path: str | Path,
    validation: ValidationResult,
    compilability: dict | None = None,
    runtime_manifest: RuntimeManifest | None = None,
) -> CompileReport:
    tool_steps = [s for s in graph.steps if s.step_type == StepType.TOOL_CALL]
    llm_steps = [s for s in graph.steps if s.step_type == StepType.LLM_GENERATE]
    high_risk = [s for s in graph.steps if s.risk_level == "high"]
    guarded_high_risk = [s for s in high_risk if s.guards]
    ungrounded = [s.id for s in tool_steps if not s.execution_spec or not s.execution_spec.command]
    manifest = runtime_manifest or extract_manifest(graph)
    dep_dicts = [dep.model_dump() for dep in manifest.dependencies]

    return CompileReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        skill_slug=package.slug,
        skill_root=package.skill_root,
        entry_skill=package.entry_skill,
        package_hash=package.package_hash,
        output_path=str(Path(output_path)),
        compilability_kind=str((compilability or {}).get("kind", "")),
        compilation_strategy=str((compilability or {}).get("strategy", "")),
        compilation_confidence=str((compilability or {}).get("confidence", "")),
        compilability_reason=str((compilability or {}).get("reason", "")),
        compilability_signals=list((compilability or {}).get("signals", [])),
        steps_total=len(graph.steps),
        edges_total=len(graph.edges),
        tool_steps=len(tool_steps),
        llm_steps=len(llm_steps),
        grounded_tool_steps=graph.grounded_tool_step_count(),
        grounded_tool_ratio=round(graph.grounded_tool_step_ratio(), 4),
        argv_backed_tool_steps=validation.argv_backed_tool_steps,
        binding_backed_tool_steps=validation.binding_backed_tool_steps,
        shell_string_only_steps=validation.shell_string_only_steps,
        unresolved_tool_steps=validation.unresolved_tool_steps,
        high_risk_steps=len(high_risk),
        guarded_high_risk_steps=len(guarded_high_risk),
        ungrounded_tool_step_ids=ungrounded,
        runtime_dependencies=dep_dicts,
        validation_errors=[f"[{i.code}] {i.message}" for i in validation.errors],
        validation_warnings=[
            f"[{i.code}] {i.message} ({i.step_id or 'graph'})" for i in validation.warnings
        ],
    )
