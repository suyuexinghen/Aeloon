"""
Analyzer: SKILL.md → SkillGraph JSON (the intermediate representation).

Parses a SKILL.md file, sends it to an LLM for DAG decomposition,
and returns a validated SkillGraph.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter
from openai import OpenAI

from .models import (
    Edge,
    ExecutionKind,
    ExecutionSpec,
    GuardSpec,
    IOField,
    SkillGraph,
    SourceRef,
    Step,
    StepType,
)
from .normalize import command_has_template, normalize_graph

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# SKILL.md Parser
# ────────────────────────────────────────────────────────────


@dataclass
class ParsedSkill:
    name: str
    description: str
    version: str
    content: str
    file_path: str

    @property
    def token_estimate(self) -> int:
        return len(self.content) // 4


def parse_skill_md(path: str | Path) -> ParsedSkill:
    """Parse a SKILL.md file into structured metadata + content."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        post = frontmatter.loads(text)
        meta = dict(post.metadata)
        content = post.content
    except Exception:
        meta = {}
        content = text
        if text.startswith("---"):
            end = text.find("---", 3)
            if end > 0:
                for line in text[3:end].strip().split("\n"):
                    if line.startswith("name:"):
                        meta["name"] = line.split(":", 1)[1].strip().strip("\"'")
                    elif line.startswith("description:"):
                        meta["description"] = line.split(":", 1)[1].strip().strip("\"'")
                content = text[end + 3 :].strip()

    name = meta.get("name", path.parent.name)
    desc = meta.get("description", "")
    if not isinstance(desc, str):
        desc = str(desc) if desc else ""
    version = str(meta.get("version", ""))

    return ParsedSkill(
        name=name,
        description=desc,
        version=version,
        content=content,
        file_path=str(path),
    )


# ────────────────────────────────────────────────────────────
# LLM Analysis Prompt
# ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a skill execution analyst. You analyze AI agent skill definitions (SKILL.md) \
and decompose them into execution DAGs.

## Step Type Classification — STRICT RULES

### `tool_call` — Deterministic tool/CLI invocation. NO LLM reasoning.
Only if: a specific CLI command or API call is fully known from inputs. \
A simple script could execute it without any AI.

### `llm_generate` — Requires LLM reasoning, judgment, or generation.
If ANY: understanding content, making design decisions, generating text/code, \
choosing what to do based on context, synthesizing information.

### `condition` — Programmatic branching (file exists? test passed?).
### `data_transform` — Pure deterministic data manipulation.

Key heuristic: "Can a Python script do this without calling an LLM?" YES → tool_call. NO → llm_generate.

## Rules
- Each step = one logical unit of work
- No edges between independent steps (enables parallelism)
- Conservative: unsure about dependency → add edge
- DAG (no cycles), snake_case ids, every step has ≥1 output
- For each tool_call step, include the exact shell commands in the description
- If available, include `execution_spec` with normalized command information
- If available, include `source_refs` pointing to supporting files and lines
- If a step is risky, set `risk_level` and include `guards`

## Output: JSON object (no markdown fences)

{
  "skill_name": "string",
  "skill_description": "string",
  "steps": [
    {
      "id": "snake_case_id",
      "name": "Human Name",
      "description": "What this does. For tool_call: include exact shell commands.",
      "step_type": "tool_call|llm_generate|condition|data_transform",
      "inputs": [{"name": "x", "description": "...", "type": "string", "required": true}],
      "outputs": [{"name": "y", "description": "...", "type": "string", "required": true}],
      "cacheable": true,
      "execution_spec": {
        "kind": "shell|python|llm|noop",
        "command": "bash scripts/run.sh --arg x",
        "cwd": ".",
        "timeout_sec": 60,
        "env": {"KEY": "value"},
        "parser": "raw"
      },
      "source_refs": [{"path": "scripts/run.sh", "line": 12, "snippet": "openclaw status", "score": 1.0}],
      "risk_level": "low|high",
      "guards": [{"kind": "env_flag", "env_var": "OPENCLAW_WRAPPER_ALLOW_RISKY", "expected_value": "1", "message": "Required for high-risk command"}]
    }
  ],
  "edges": [{"from_step": "a", "to_step": "b", "description": "data flow"}],
  "global_inputs": [{"name": "x", "description": "...", "type": "string", "required": true}],
  "global_outputs": [{"name": "y", "description": "...", "type": "string", "required": true}]
}
"""

USER_PROMPT_TEMPLATE = """\
Analyze this SKILL.md into an execution DAG.

Be STRICT about step_type: "Can a Python script do this without an LLM?" If no → llm_generate.
For tool_call steps, include the exact shell commands in the description field.
When possible, include execution_spec/source_refs/risk_level/guards.

## Skill: {name}
{description}

## Content

{content}

---
Return ONLY the JSON object.\
"""

COMMAND_PREFIXES = (
    "openclaw ",
    "browser ",
    "bash ",
    "sh ",
    "python ",
    "python3 ",
    "node ",
    "npm ",
    "pnpm ",
    "yarn ",
    "git ",
    "pip ",
    "pip3 ",
    "curl ",
    "wget ",
    "docker ",
    "docker-compose ",
    "go ",
    "govulncheck ",
    "cargo ",
    "make ",
    "./",
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "step",
    "check",
    "run",
    "execute",
    "build",
    "command",
    "commands",
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
ANGLE_ARG_RE = re.compile(r"<([a-zA-Z0-9_-]+)>")
FLAG_VALUE_RE = re.compile(r"--([a-zA-Z0-9_-]+)\s+<([a-zA-Z0-9_-]+)>")
OPTIONAL_TOKEN_RE = re.compile(r"\[([^\[\]]+)\]")
REPEAT_ARG_RE = re.compile(r"<([a-zA-Z0-9_-]+)>\.\.\.")

HIGH_RISK_KEYWORDS = {
    "cron",
    "webhooks",
    "dns",
    "nodes",
    "node",
    "pairing",
    "devices",
    "plugin",
    "plugins",
    "hooks",
    "secrets",
    "sandbox",
    "recreate",
    "camera",
    "screen",
    "location",
}


def _extract_shell_commands(text: str) -> list[tuple[str, int]]:
    """Extract command-like lines and 1-based line numbers from text."""
    commands: list[tuple[str, int]] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        if line.startswith("```"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if line.startswith(('"', "'")):
            line = line.strip("\"'").rstrip(",")
        if line.startswith("`") and line.endswith("`") and len(line) > 2:
            line = line[1:-1].strip()
        if not line:
            continue

        run_match = re.match(r"^(?:\d+\.\s*)?run:\s*(.+)$", line, re.I)
        if run_match:
            cmd = run_match.group(1).strip()
            cmd = re.split(r"\s+#", cmd, maxsplit=1)[0].strip()
            cmd = re.split(r"\s+\(", cmd, maxsplit=1)[0].strip()
            if cmd.startswith(COMMAND_PREFIXES):
                commands.append((cmd, idx))
            continue

        if line.startswith(COMMAND_PREFIXES):
            cmd = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            commands.append((cmd, idx))
    return commands


def _keyword_tokens(step: Step) -> set[str]:
    text = f"{step.id} {step.name} {step.description}".lower()
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    return {t for t in tokens if len(t) >= 3 and t not in STOPWORDS}


def _score_line(line: str, tokens: set[str]) -> float:
    if not tokens:
        return 0.0
    low = line.lower()
    hit = sum(1 for t in tokens if t in low)
    if hit == 0:
        return 0.0
    return float(hit)


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


# ────────────────────────────────────────────────────────────
# Analyzer
# ────────────────────────────────────────────────────────────


class Analyzer:
    """Analyzes SKILL.md → SkillGraph using an LLM."""

    def __init__(
        self,
        model: str = "anthropic/claude-opus-4.6",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_content_chars: int = 15000,
    ):
        resolved_api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("SKILLGRAPH_API_KEY")
        )
        kwargs = {}
        if resolved_api_key:
            kwargs["api_key"] = resolved_api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs) if resolved_api_key else None
        self.api_key = resolved_api_key or ""
        self.model = model
        self.temperature = temperature
        self.max_content_chars = max_content_chars

    def analyze(
        self,
        skill_path: str | Path,
        cache_path: str | Path | None = None,
        use_cache: bool = True,
    ) -> SkillGraph:
        """
        Analyze a SKILL.md and return a SkillGraph.
        If cache_path is given and exists, loads from cache instead of calling LLM
        unless use_cache is False.
        """
        # Parse
        skill = parse_skill_md(skill_path)
        skill_file = Path(skill.file_path)

        # Check cache
        if cache_path and use_cache:
            cp = Path(cache_path)
            if cp.exists():
                logger.info(f"Loading cached graph: {cp}")
                graph = SkillGraph.load(cp)
                if self._ground_from_package(graph, skill_file):
                    graph.save(cp)
                    logger.info(f"  Refreshed cache with grounded metadata: {cp}")
                return normalize_graph(graph)

        # Truncate
        content = skill.content
        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars] + "\n\n[... truncated ...]"

        seed_graph = self._pre_analyze(skill)

        prompt = USER_PROMPT_TEMPLATE.format(
            name=skill.name,
            description=skill.description or "N/A",
            content=self._build_llm_content(content, seed_graph),
        )

        if self.client is None:
            raise ValueError("Analyzer requires an API key for LLM graph synthesis")

        logger.info(f"Analyzing: {skill.name} (~{skill.token_estimate} tokens)")

        # Call LLM
        create_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 8192,
        }
        if "claude" not in self.model.lower() and "anthropic" not in self.model.lower():
            create_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**create_kwargs)
        raw = response.choices[0].message.content or ""

        # Parse JSON (handle markdown fences)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[cleaned.index("\n") + 1 :]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(cleaned[start : end + 1])
            else:
                raise ValueError(f"Cannot parse LLM response as JSON:\n{raw[:500]}")

        # Build SkillGraph
        graph = self._build(data, skill)
        graph = self._merge_seed_graph(seed_graph, graph)
        self._ground_from_package(graph, skill_file)
        graph = normalize_graph(graph)

        if usage := response.usage:
            logger.info(
                f"  Tokens: {usage.prompt_tokens}+{usage.completion_tokens}={usage.total_tokens}"
            )

        # Cache
        if cache_path:
            graph.save(cache_path)
            logger.info(f"  Cached: {cache_path}")

        return graph

    def _build_llm_content(self, content: str, seed_graph: SkillGraph) -> str:
        if not seed_graph.steps:
            return content
        seed_json = seed_graph.model_dump_json(indent=2)
        return (
            f"{content}\n\n"
            "## Deterministic extraction seed\n"
            "The following structure was extracted without the LLM. Reuse and refine it rather than inventing a new graph.\n\n"
            f"{seed_json}"
        )

    def _pre_analyze(self, skill: ParsedSkill) -> SkillGraph:
        """Build a deterministic seed graph from headings and explicit commands."""
        lines = skill.content.splitlines()
        global_inputs = [
            IOField(name="project_dir", description="Working project directory", type="string")
        ]
        sections: list[tuple[str, list[str]]] = []
        current_title = "skill_overview"
        current_lines: list[str] = []

        for line in lines:
            match = HEADING_RE.match(line.strip())
            if match:
                if current_lines:
                    sections.append((current_title, current_lines))
                current_title = match.group(2).strip()
                current_lines = []
                continue
            current_lines.append(line)
        if current_lines:
            sections.append((current_title, current_lines))

        steps: list[Step] = []
        edges: list[Edge] = []
        previous_step_id = ""
        commands_seen: set[str] = set()

        for index, (title, body_lines) in enumerate(sections, start=1):
            body = "\n".join(body_lines).strip()
            if self._is_descriptive_section(title, body):
                continue
            commands = _extract_shell_commands(body)
            if commands:
                for cmd_index, (command, _) in enumerate(commands, start=1):
                    if command in commands_seen:
                        continue
                    commands_seen.add(command)
                    step_id = self._seed_step_id(title, index, cmd_index)
                    outputs = [
                        IOField(
                            name=f"{step_id}_output",
                            description=f"Output of command: {command}",
                            type="string",
                        )
                    ]
                    if any(
                        token in command
                        for token in (
                            "install",
                            "link",
                            "setup",
                            "doctor",
                            "status",
                            "navigate",
                            "extract",
                            "act",
                        )
                    ):
                        outputs.append(
                            IOField(
                                name="exit_code",
                                description="Command exit code",
                                type="int",
                                required=False,
                            )
                        )
                    step = Step(
                        id=step_id,
                        name=f"{title}: {command.split()[0]}",
                        description=body or command,
                        step_type=StepType.TOOL_CALL,
                        inputs=self._infer_command_inputs(command),
                        outputs=outputs,
                        execution_spec=ExecutionSpec(
                            kind=ExecutionKind.SHELL,
                            command=command,
                            argv=self._command_argv(command),
                            arg_bindings=self._command_arg_bindings(command),
                            cwd=".",
                            timeout_sec=120
                            if any(word in command for word in ("install", "audit", "build"))
                            else 60,
                            parser="raw",
                        ),
                    )
                    steps.append(step)
                    if previous_step_id:
                        edges.append(
                            Edge(
                                from_step=previous_step_id,
                                to_step=step_id,
                                description="Sequential deterministic command order",
                            )
                        )
                    previous_step_id = step_id
            elif body:
                step_id = self._seed_step_id(title, index, 1)
                step_type = (
                    StepType.CONDITION
                    if any(word in body.lower() for word in ("if ", "when ", "require", "approval"))
                    else StepType.LLM_GENERATE
                )
                outputs = [
                    IOField(
                        name=f"{step_id}_result",
                        description=f"Result for section {title}",
                        type="bool" if step_type == StepType.CONDITION else "string",
                    )
                ]
                steps.append(
                    Step(
                        id=step_id,
                        name=title,
                        description=body,
                        step_type=step_type,
                        inputs=[],
                        outputs=outputs,
                    )
                )
                if previous_step_id:
                    edges.append(
                        Edge(
                            from_step=previous_step_id,
                            to_step=step_id,
                            description="Sequential section order",
                        )
                    )
                previous_step_id = step_id

        return SkillGraph(
            skill_name=skill.name,
            skill_description=skill.description,
            skill_version=skill.version,
            steps=steps,
            edges=edges,
            global_inputs=global_inputs,
            global_outputs=[],
            analyzer_model=f"seed+{self.model}",
        )

    def _seed_step_id(self, title: str, section_index: int, item_index: int) -> str:
        base = re.sub(r"[^a-z0-9_]+", "_", title.lower()).strip("_") or f"section_{section_index}"
        return f"{base}_{section_index}_{item_index}"

    def _is_descriptive_section(self, title: str, body: str) -> bool:
        low_title = title.lower().strip()
        low_body = body.lower().strip()
        if not low_body:
            return True
        if low_title in {
            "skill_overview",
            "browser automation",
            "quick reference",
            "mode comparison",
            "best practices",
            "troubleshooting",
            "non-goals",
            "notes",
            "remarks",
        }:
            return True
        if low_body.startswith("---") or low_body.startswith("name:"):
            return True
        lines = [line for line in body.splitlines() if line.strip()]
        table_lines = [line for line in lines if line.lstrip().startswith("|")]
        if table_lines and len(table_lines) >= max(2, len(lines) // 2):
            return True
        return False

    def _infer_command_inputs(self, command: str) -> list[IOField]:
        inputs: list[IOField] = []
        seen: set[str] = set()

        def add(
            name: str,
            description: str,
            required: bool = True,
            field_type: str = "string",
        ) -> None:
            if name in seen:
                return
            seen.add(name)
            inputs.append(
                IOField(name=name, description=description, type=field_type, required=required)
            )

        args = ANGLE_ARG_RE.findall(command)
        for raw in args:
            token = raw.strip().lower().replace("-", "_")
            if token == "url":
                add("target_url", "URL to navigate to")
            elif token == "action":
                add("action_description", "Natural language action to perform")
            elif token == "instruction":
                add("extract_instruction", "Natural language instruction for extraction")
            elif token == "query":
                add("observe_query", "Query describing which elements to observe")
            else:
                add(token, f"Command argument '{raw}'")

        for raw in REPEAT_ARG_RE.findall(command):
            token = raw.strip().lower().replace("-", "_")
            add(
                self._canonical_input_name(token),
                f"Repeatable command argument '{raw}'",
                field_type="list",
            )

        for flag_name, value_name in FLAG_VALUE_RE.findall(command):
            value_key = value_name.strip().lower().replace("-", "_")
            field_name = self._canonical_input_name(value_key)
            add(field_name, f"Value for --{flag_name}")

        for optional in OPTIONAL_TOKEN_RE.findall(command):
            optional = optional.strip()
            if optional.startswith("{") and optional.endswith("}"):
                add("json_payload", "Optional JSON-like payload", field_type="dict", required=False)

        for optional in OPTIONAL_TOKEN_RE.findall(command):
            optional = optional.strip()
            if not optional or optional.startswith("'") or optional.startswith('"'):
                continue
            for flag_name, value_name in FLAG_VALUE_RE.findall(optional):
                value_key = value_name.strip().lower().replace("-", "_")
                field_name = self._canonical_input_name(value_key)
                add(field_name, f"Optional value for --{flag_name}", required=False)
            for raw in ANGLE_ARG_RE.findall(optional):
                token = raw.strip().lower().replace("-", "_")
                add(
                    self._canonical_input_name(token),
                    f"Optional command argument '{raw}'",
                    required=False,
                )

        if command.startswith("browser navigate"):
            add("target_url", "URL to navigate to")
        elif command.startswith("browser act"):
            add("action_description", "Natural language action to perform")
            add("navigate_output", "Navigation completion output", required=False)
        elif command.startswith("browser extract"):
            add("extract_instruction", "Natural language instruction for extraction")
            add("extract_schema", "Optional JSON schema for structured extraction", required=False)
            add("navigate_output", "Navigation completion output", required=False)
        elif command.startswith("browser observe"):
            add("observe_query", "Query describing which elements to observe")
            add("navigate_output", "Navigation completion output", required=False)
        elif command.startswith("browser screenshot"):
            add("navigate_output", "Navigation completion output", required=False)

        return inputs

    def _command_argv(self, command: str) -> list[str]:
        try:
            import shlex

            return shlex.split(command)
        except Exception:
            return []

    def _command_arg_bindings(self, command: str) -> dict[str, str]:
        bindings: dict[str, str] = {}
        for token in ANGLE_ARG_RE.findall(command):
            raw = token.strip().lower().replace("-", "_")
            bindings[f"<{token}>"] = self._canonical_input_name(raw)
        return bindings

    def _canonical_input_name(self, raw: str) -> str:
        if raw == "url":
            return "target_url"
        if raw == "action":
            return "action_description"
        if raw == "instruction":
            return "extract_instruction"
        if raw == "query":
            return "observe_query"
        return raw

    def _merge_seed_graph(self, seed_graph: SkillGraph, llm_graph: SkillGraph) -> SkillGraph:
        if not seed_graph.steps:
            return llm_graph

        llm_steps = {step.id: step for step in llm_graph.steps}
        merged_steps: list[Step] = []

        for seed_step in seed_graph.steps:
            step = llm_steps.get(seed_step.id)
            if step is None:
                merged_steps.append(seed_step)
                continue
            if seed_step.execution_spec and not step.execution_spec:
                step.execution_spec = seed_step.execution_spec
            if not step.outputs:
                step.outputs = seed_step.outputs
            if not step.inputs:
                step.inputs = seed_step.inputs
            else:
                step.inputs = self._merge_inputs(step.inputs, seed_step.inputs)
            if step.step_type == StepType.LLM_GENERATE and seed_step.execution_spec:
                step.step_type = seed_step.step_type
            if step.execution_spec and step.execution_spec.command:
                step.inputs = self._align_inputs_to_command(step)
            if step.id == "synthesize_results":
                step.inputs = self._rewrite_synthesis_inputs(step.inputs)
            merged_steps.append(step)

        merged_ids = {step.id for step in merged_steps}
        for step in llm_graph.steps:
            if step.id not in merged_ids:
                merged_steps.append(step)

        merged_edges = list(seed_graph.edges)
        existing_edges = {(edge.from_step, edge.to_step) for edge in merged_edges}
        for edge in llm_graph.edges:
            pair = (edge.from_step, edge.to_step)
            if pair not in existing_edges:
                merged_edges.append(edge)
                existing_edges.add(pair)

        llm_graph.steps = merged_steps
        llm_graph.edges = merged_edges
        if not llm_graph.global_inputs:
            llm_graph.global_inputs = seed_graph.global_inputs
        return llm_graph

    def _merge_inputs(self, current: list[IOField], inferred: list[IOField]) -> list[IOField]:
        alias_map = {
            "url": "target_url",
            "target": "target_url",
            "action": "action_description",
            "instruction": "extract_instruction",
            "query": "observe_query",
        }
        by_name = {field.name: field for field in inferred}
        merged: list[IOField] = []
        used: set[str] = set()

        for field in current:
            mapped_name = alias_map.get(field.name, field.name)
            candidate = by_name.get(mapped_name)
            if candidate:
                merged.append(
                    IOField(
                        name=candidate.name,
                        description=field.description or candidate.description,
                        type=candidate.type,
                        required=field.required,
                    )
                )
                used.add(candidate.name)
            elif field.name not in used and field.name != "command_outputs":
                merged.append(field)
                used.add(field.name)

        for field in inferred:
            if field.name not in used:
                merged.append(field)
                used.add(field.name)
        return merged

    def _rewrite_synthesis_inputs(self, inputs: list[IOField]) -> list[IOField]:
        rewritten: list[IOField] = []
        seen: set[str] = set()
        for field in inputs:
            if field.name == "command_outputs":
                replacement = IOField(
                    name="navigate_output",
                    description="Combined output from executed browser commands",
                    type="string",
                    required=False,
                )
                if replacement.name not in seen:
                    rewritten.append(replacement)
                    seen.add(replacement.name)
                continue
            if field.name not in seen:
                rewritten.append(field)
                seen.add(field.name)
        return rewritten

    def _align_inputs_to_command(self, step: Step) -> list[IOField]:
        command = step.execution_spec.command if step.execution_spec else ""
        inferred = self._infer_command_inputs(command)
        if not inferred:
            return step.inputs
        return self._merge_inputs(step.inputs, inferred)

    def _build(self, data: dict, skill: ParsedSkill) -> SkillGraph:
        steps = []
        for s in data.get("steps", []):
            raw_type = s.get("step_type", "llm_generate")
            try:
                step_type = StepType(raw_type)
            except Exception:
                step_type = StepType.LLM_GENERATE

            execution_spec = None
            if isinstance(s.get("execution_spec"), dict):
                try:
                    execution_spec = ExecutionSpec.model_validate(s["execution_spec"])
                except Exception:
                    execution_spec = None

            source_refs = []
            for ref in s.get("source_refs", []):
                if isinstance(ref, dict):
                    try:
                        source_refs.append(SourceRef.model_validate(ref))
                    except Exception:
                        continue

            guards = []
            for g in s.get("guards", []):
                if isinstance(g, dict):
                    try:
                        guards.append(GuardSpec.model_validate(g))
                    except Exception:
                        continue

            steps.append(
                Step(
                    id=s["id"],
                    name=s["name"],
                    description=s.get("description", ""),
                    step_type=step_type,
                    inputs=[IOField(**f) for f in s.get("inputs", [])],
                    outputs=[IOField(**f) for f in s.get("outputs", [])],
                    cacheable=s.get("cacheable", True),
                    execution_spec=execution_spec,
                    source_refs=source_refs,
                    risk_level=str(s.get("risk_level", "")),
                    guards=guards,
                )
            )
        step_ids = {s.id for s in steps}
        edges = [
            Edge(
                from_step=e["from_step"], to_step=e["to_step"], description=e.get("description", "")
            )
            for e in data.get("edges", [])
            if e["from_step"] in step_ids and e["to_step"] in step_ids
        ]
        return SkillGraph(
            skill_name=data.get("skill_name", skill.name),
            skill_description=data.get("skill_description", skill.description),
            skill_version=skill.version,
            steps=steps,
            edges=edges,
            global_inputs=[IOField(**f) for f in data.get("global_inputs", [])],
            global_outputs=[IOField(**f) for f in data.get("global_outputs", [])],
            analyzer_model=self.model,
        )

    def _ground_from_package(self, graph: SkillGraph, skill_file: Path) -> bool:
        """Enrich steps with execution specs, source refs, and risk metadata."""
        skill_root = skill_file.parent
        context_files = self._context_files(skill_root)
        text_cache: dict[Path, str] = {}
        changed = False

        for step in graph.steps:
            before = step.model_dump_json()

            self._attach_source_refs(step, skill_root, context_files, text_cache)

            if step.step_type == StepType.TOOL_CALL:
                command, source_ref = self._select_command_for_step(
                    step, skill_file, skill_root, context_files, text_cache
                )
                current_command = step.execution_spec.command if step.execution_spec else ""
                if command and self._prefer_candidate_command(step, command, current_command):
                    is_builtin = command.startswith("builtin:")
                    kind = ExecutionKind.PYTHON if is_builtin else ExecutionKind.SHELL
                    timeout_sec = (
                        5
                        if is_builtin
                        else (
                            120
                            if re.search(r"\b(install|update|audit|scan|build)\b", command, re.I)
                            else 60
                        )
                    )
                    step.execution_spec = ExecutionSpec(
                        kind=kind,
                        command=command,
                        cwd=".",
                        timeout_sec=timeout_sec,
                        parser="raw",
                    )
                    if source_ref:
                        step.source_refs.insert(0, source_ref)

            if step.step_type == StepType.LLM_GENERATE and not step.execution_spec:
                step.execution_spec = ExecutionSpec(kind=ExecutionKind.LLM, parser="text")
            if (
                step.step_type in {StepType.CONDITION, StepType.DATA_TRANSFORM}
                and not step.execution_spec
            ):
                step.execution_spec = ExecutionSpec(kind=ExecutionKind.PYTHON, parser="raw")

            step.risk_level = self._infer_risk_level(step)

            if step.risk_level == "high" and self._needs_openclaw_guard(step) and not step.guards:
                step.guards = [
                    GuardSpec(
                        kind="env_flag",
                        env_var="OPENCLAW_WRAPPER_ALLOW_RISKY",
                        expected_value="1",
                        message="High-risk operation requires OPENCLAW_WRAPPER_ALLOW_RISKY=1",
                    )
                ]
            if step.risk_level != "high" and step.guards:
                only_openclaw_guard = all(
                    g.env_var == "OPENCLAW_WRAPPER_ALLOW_RISKY" for g in step.guards
                )
                if only_openclaw_guard:
                    step.guards = []

            step.source_refs = self._dedupe_source_refs(step.source_refs, max_items=8)

            if step.model_dump_json() != before:
                changed = True

        if self._repair_contracts(graph):
            changed = True

        tool_steps = sum(1 for s in graph.steps if s.step_type == StepType.TOOL_CALL)
        logger.info(
            "  Grounded tool steps: %d/%d (%.0f%%)",
            graph.grounded_tool_step_count(),
            tool_steps,
            graph.grounded_tool_step_ratio() * 100.0,
        )
        return changed

    def _repair_contracts(self, graph: SkillGraph) -> bool:
        """Repair common IO mismatches introduced by LLM analysis."""
        changed = False
        global_inputs = {i.name for i in graph.global_inputs}
        required_names = {inp.name for s in graph.steps for inp in s.inputs if inp.required}

        has_flags = sorted(n for n in required_names if n.startswith("has_"))
        if has_flags:
            for step in graph.steps:
                sid = step.id.lower()
                if "detect" in sid and "project" in sid:
                    existing = {o.name for o in step.outputs}
                    for name in has_flags:
                        if name in existing:
                            continue
                        step.outputs.append(
                            IOField(
                                name=name,
                                description=f"Inferred ecosystem flag: {name}",
                                type="bool",
                                required=False,
                            )
                        )
                        changed = True

        # Auto-add missing edges when a required input has a unique producer.
        output_producers: dict[str, list[str]] = {}
        for s in graph.steps:
            for o in s.outputs:
                output_producers.setdefault(o.name, []).append(s.id)

        edge_pairs = {(e.from_step, e.to_step) for e in graph.edges}

        for step in graph.steps:
            upstream_ids = graph.upstream_of(step.id)
            upstream_outputs = set()
            for uid in upstream_ids:
                us = graph.get_step(uid)
                if us:
                    upstream_outputs.update(o.name for o in us.outputs)

            for inp in step.inputs:
                if not inp.required:
                    continue
                if inp.name in global_inputs or inp.name in upstream_outputs:
                    continue

                if inp.name == "url":
                    inp.required = False
                    if all(i.name != "command_sequence" for i in step.inputs):
                        step.inputs.append(
                            IOField(
                                name="command_sequence",
                                description="Optional planned command sequence for URL extraction",
                                type="list",
                                required=False,
                            )
                        )
                    if "user_request" in global_inputs and all(
                        i.name != "user_request" for i in step.inputs
                    ):
                        step.inputs.append(
                            IOField(
                                name="user_request",
                                description="Original user request for URL extraction fallback",
                                type="string",
                                required=False,
                            )
                        )
                    changed = True
                    continue

                producers = output_producers.get(inp.name, [])
                if len(producers) != 1:
                    continue
                producer = producers[0]
                if producer == step.id:
                    continue
                if (producer, step.id) in edge_pairs:
                    continue
                if self._path_exists(graph, step.id, producer):
                    continue

                graph.edges.append(
                    Edge(
                        from_step=producer,
                        to_step=step.id,
                        description=f"Auto-added edge for required input '{inp.name}'",
                    )
                )
                edge_pairs.add((producer, step.id))
                upstream_outputs.add(inp.name)
                changed = True

        return changed

    def _path_exists(self, graph: SkillGraph, start: str, goal: str) -> bool:
        if start == goal:
            return True
        stack = [start]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in graph.downstream_of(cur):
                if nxt == goal:
                    return True
                if nxt not in seen:
                    stack.append(nxt)
        return False

    def _context_files(self, skill_root: Path) -> list[Path]:
        files = [p for p in skill_root.rglob("*") if p.is_file()]

        def priority(path: Path) -> tuple[int, str]:
            rel = _relpath(path, skill_root)
            if rel == "SKILL.md":
                return (0, rel)
            if rel.startswith("scripts/"):
                return (1, rel)
            if rel == "setup.json":
                return (2, rel)
            if rel.startswith("references/"):
                return (3, rel)
            if path.name in {"REFERENCE.md", "EXAMPLES.md"}:
                return (4, rel)
            if path.suffix.lower() == ".md":
                return (5, rel)
            if path.suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".ini"}:
                return (6, rel)
            return (9, rel)

        files.sort(key=priority)
        result: list[Path] = []
        for f in files:
            try:
                if f.stat().st_size > 500_000:
                    continue
            except OSError:
                continue
            result.append(f)
            if len(result) >= 120:
                break
        return result

    def _safe_read(self, path: Path, cache: dict[Path, str]) -> str:
        if path in cache:
            return cache[path]
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        if len(text) > 60_000:
            text = text[:60_000]
        cache[path] = text
        return text

    def _attach_source_refs(
        self,
        step: Step,
        skill_root: Path,
        context_files: list[Path],
        text_cache: dict[Path, str],
    ) -> None:
        tokens = _keyword_tokens(step)
        if not tokens:
            return

        candidates: list[SourceRef] = []
        for path in context_files:
            text = self._safe_read(path, text_cache)
            if not text:
                continue

            rel = _relpath(path, skill_root)
            bonus = (
                2.0 if rel.startswith("scripts/") else 1.0 if rel.startswith("references/") else 0.0
            )
            for line_no, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                score = _score_line(stripped, tokens)
                if score <= 0:
                    continue
                candidates.append(
                    SourceRef(
                        path=rel,
                        line=line_no,
                        snippet=stripped[:220],
                        score=score + bonus,
                    )
                )

        if candidates:
            candidates.sort(key=lambda r: r.score, reverse=True)
            step.source_refs.extend(candidates[:6])

    def _select_command_for_step(
        self,
        step: Step,
        skill_file: Path,
        skill_root: Path,
        context_files: list[Path],
        text_cache: dict[Path, str],
    ) -> tuple[str, SourceRef | None]:
        # First, trust explicit command-like lines already in the step description.
        desc_cmds = _extract_shell_commands(step.description)
        if desc_cmds:
            cmd, _ = desc_cmds[0]
            ref = self._locate_command_source(cmd, skill_root, context_files, text_cache)
            if ref is None:
                ref = SourceRef(
                    path=_relpath(skill_file, skill_root), line=None, snippet=cmd[:220], score=1.0
                )
            return cmd, ref

        tokens = _keyword_tokens(step)
        best: tuple[float, str, SourceRef] | None = None

        for path in context_files:
            text = self._safe_read(path, text_cache)
            if not text:
                continue
            rel = _relpath(path, skill_root)
            file_bonus = (
                0.6 if rel.startswith("scripts/") else 0.3 if rel.startswith("references/") else 0.1
            )
            for cmd, line_no in _extract_shell_commands(text):
                match_score = _score_line(cmd, tokens)
                if match_score <= 0:
                    continue
                if len(tokens) >= 3 and match_score < 2.0:
                    continue
                score = match_score + file_bonus
                ref = SourceRef(path=rel, line=line_no, snippet=cmd[:220], score=score)
                if best is None or score > best[0]:
                    best = (score, cmd, ref)

        if best is None:
            heuristic = self._heuristic_command_for_step(step)
            if heuristic:
                ref = self._locate_command_source(heuristic, skill_root, context_files, text_cache)
                return heuristic, ref
            return "", None
        return best[1], best[2]

    def _locate_command_source(
        self,
        command: str,
        skill_root: Path,
        context_files: list[Path],
        text_cache: dict[Path, str],
    ) -> SourceRef | None:
        for path in context_files:
            text = self._safe_read(path, text_cache)
            if not text or command not in text:
                continue
            line_no = self._find_line_no(text, command)
            return SourceRef(
                path=_relpath(path, skill_root), line=line_no, snippet=command[:220], score=1.0
            )
        return None

    def _find_line_no(self, text: str, needle: str) -> int | None:
        for i, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                return i
        return None

    def _prefer_candidate_command(self, step: Step, candidate: str, current: str) -> bool:
        if command_has_template(candidate):
            return False
        if not current.strip() or command_has_template(current):
            return True
        candidate_is_cmd = candidate.startswith(COMMAND_PREFIXES)
        current_is_cmd = current.startswith(COMMAND_PREFIXES)
        if candidate_is_cmd and not current_is_cmd:
            return True

        cand_score = self._command_score(step, candidate)
        curr_score = self._command_score(step, current)
        return cand_score > curr_score + 0.25

    def _command_score(self, step: Step, command: str) -> float:
        score = _score_line(command, _keyword_tokens(step))
        if command.startswith(COMMAND_PREFIXES):
            score += 1.0
        return score

    def _heuristic_command_for_step(self, step: Step) -> str:
        text = f"{step.id} {step.name} {step.description}".lower()
        sid = step.id.lower()

        if "detect" in sid and "project" in sid:
            return "builtin:detect_project_type"
        if "govulncheck" in sid:
            return "govulncheck ./... 2>&1"
        if "injection" in sid:
            return "builtin:scan_injection_patterns"
        if "xss" in sid:
            return "builtin:scan_xss_patterns"
        if "ssl" in sid:
            return "builtin:check_ssl_endpoints"

        if "browserbase" in text and ("env" in text or ".env" in text):
            return 'bash -c \'if [ -f .env ] && grep -q "^BROWSERBASE_API_KEY=" .env && grep -q "^BROWSERBASE_PROJECT_ID=" .env; then echo browserbase; else echo local; fi\''
        if "setup.json" in text and ("check" in text or "read" in text):
            return "cat setup.json"
        if "npm" in text and "install" in text:
            return "npm install"
        if "npm" in text and "link" in text:
            return "npm link"
        if "openclaw" in text and ("version" in text or "availability" in text or "which" in text):
            return "openclaw version"
        if "openclaw" in text and "doctor" in text:
            return "openclaw doctor"
        if "risk" in text and ("classify" in text or "classification" in text):
            return "builtin:classify_openclaw_risk"
        return ""

    def _infer_risk_level(self, step: Step) -> str:
        joined = f"{step.id} {step.name} {step.description}".lower()
        if step.execution_spec and step.execution_spec.command:
            joined += " " + step.execution_spec.command.lower()
        if "openclaw" not in joined:
            return "low"
        return "high" if any(k in joined for k in HIGH_RISK_KEYWORDS) else "low"

    def _needs_openclaw_guard(self, step: Step) -> bool:
        joined = f"{step.id} {step.name} {step.description}".lower()
        if step.execution_spec and step.execution_spec.command:
            joined += " " + step.execution_spec.command.lower()
        return "openclaw" in joined

    def _dedupe_source_refs(self, refs: list[SourceRef], max_items: int = 8) -> list[SourceRef]:
        best: dict[tuple[str, int | None, str], SourceRef] = {}
        for r in refs:
            key = (r.path, r.line, r.snippet)
            if key not in best or r.score > best[key].score:
                best[key] = r
        ordered = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return ordered[:max_items]
