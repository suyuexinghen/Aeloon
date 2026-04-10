"""Normalization helpers for turning fuzzy analyzer output into stable SkillGraph IR."""

from __future__ import annotations

import re
import shlex

from .models import Edge, ExecutionKind, ExecutionSpec, IOField, SkillGraph, Step, StepType

TYPE_ALIASES = {
    "str": "string",
    "string": "string",
    "text": "string",
    "bool": "bool",
    "boolean": "bool",
    "int": "int",
    "integer": "int",
    "number": "float",
    "float": "float",
    "double": "float",
    "list": "list",
    "array": "list",
    "dict": "dict",
    "object": "dict",
    "map": "dict",
    "json": "dict",
}

COMMAND_TEMPLATE_MARKERS = ("{{", "}}", "{%", "%}")


def normalize_graph(graph: SkillGraph) -> SkillGraph:
    """Return a normalized copy-like graph in place for deterministic codegen."""
    graph.skill_name = _normalize_skill_name(graph.skill_name)
    graph.global_inputs = _normalize_fields(graph.global_inputs)
    graph.global_outputs = _normalize_fields(graph.global_outputs)

    normalized_steps: list[Step] = []
    seen_ids: set[str] = set()
    for index, step in enumerate(graph.steps, start=1):
        step.id = _normalize_step_id(step.id or step.name or f"step_{index}")
        if step.id in seen_ids:
            suffix = 2
            base = step.id
            while f"{base}_{suffix}" in seen_ids:
                suffix += 1
            step.id = f"{base}_{suffix}"
        seen_ids.add(step.id)

        if not step.name.strip():
            step.name = step.id.replace("_", " ").title()
        step.inputs = _normalize_fields(step.inputs)
        step.outputs = _normalize_fields(step.outputs)
        if not step.outputs:
            step.outputs = [
                IOField(
                    name="output",
                    description=f"Default output for step {step.id}",
                    type="string",
                    required=True,
                )
            ]

        if step.execution_spec:
            step.execution_spec = _normalize_execution_spec(step.execution_spec)
        elif step.step_type == StepType.LLM_GENERATE:
            step.execution_spec = ExecutionSpec(kind=ExecutionKind.LLM, parser="text")
        elif step.step_type in {StepType.CONDITION, StepType.DATA_TRANSFORM}:
            step.execution_spec = ExecutionSpec(kind=ExecutionKind.PYTHON, parser="raw")

        normalized_steps.append(step)

    valid_ids = {step.id for step in normalized_steps}
    normalized_edges: list[Edge] = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in graph.edges:
        from_step = _normalize_step_id(edge.from_step)
        to_step = _normalize_step_id(edge.to_step)
        pair = (from_step, to_step)
        if from_step not in valid_ids or to_step not in valid_ids:
            continue
        if from_step == to_step:
            continue
        if pair in seen_edges:
            continue
        seen_edges.add(pair)
        normalized_edges.append(
            Edge(from_step=from_step, to_step=to_step, description=(edge.description or "").strip())
        )

    graph.steps = normalized_steps
    graph.edges = normalized_edges
    return graph


def command_has_template(command: str) -> bool:
    return any(marker in command for marker in COMMAND_TEMPLATE_MARKERS)


def _normalize_fields(fields: list[IOField]) -> list[IOField]:
    normalized: list[IOField] = []
    seen_names: set[str] = set()
    for field in fields:
        name = _normalize_identifier(field.name)
        if not name:
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        normalized.append(
            IOField(
                name=name,
                description=(field.description or "").strip(),
                type=_normalize_type(field.type),
                required=bool(field.required),
            )
        )
    return normalized


def _normalize_execution_spec(spec: ExecutionSpec) -> ExecutionSpec:
    command = (spec.command or "").strip()
    if command_has_template(command):
        command = ""
    argv = [part for part in (spec.argv or []) if str(part).strip()]
    if not argv and command:
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = []

    parser = (spec.parser or "raw").strip() or "raw"
    cwd = (spec.cwd or ".").strip() or "."
    timeout_sec = max(1, int(spec.timeout_sec or 60))
    kind = spec.kind
    if kind == ExecutionKind.SHELL and not command and not argv:
        kind = ExecutionKind.NOOP

    return ExecutionSpec(
        kind=kind,
        command=command,
        argv=argv,
        arg_bindings=dict(spec.arg_bindings or {}),
        cwd=cwd,
        timeout_sec=timeout_sec,
        env=dict(spec.env or {}),
        parser=parser,
    )


def _normalize_skill_name(name: str) -> str:
    cleaned = (name or "skill").strip()
    return cleaned or "skill"


def _normalize_step_id(value: str) -> str:
    normalized = _normalize_identifier(value)
    return normalized or "step"


def _normalize_identifier(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text and text[0].isdigit():
        text = f"n_{text}"
    return text


def _normalize_type(value: str) -> str:
    key = (value or "string").strip().lower()
    return TYPE_ALIASES.get(key, "string")
