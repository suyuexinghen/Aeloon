"""Assess whether a skill is suitable for deterministic graph compilation."""

from __future__ import annotations

import re
from pathlib import Path

from .package import SkillPackage

COMMAND_PREFIXES = (
    "bash ",
    "sh ",
    "python ",
    "python3 ",
    "npx ",
    "ts-node ",
    "pip ",
    "pip3 ",
    "grep ",
    "find ",
    "sed ",
    "awk ",
    "ls ",
    "mv ",
    "cp ",
    "mkdir ",
    "rm ",
    "du ",
    "sort ",
    "uniq ",
    "curl ",
    "wget ",
    "git ",
    "pytest ",
    "mvn ",
    "gradle ",
    "npm ",
    "pnpm ",
    "yarn ",
    "node ",
    "go ",
    "cargo ",
    "make ",
    "ffmpeg ",
    "ffprobe ",
    "trivy ",
    "browser ",
    "openclaw ",
    "md5 ",
)
WORKFLOW_SECTION_MARKERS = (
    "step ",
    "workflow",
    "instructions",
    "verification",
    "execute",
    "approval",
    "summary",
    "plan",
)
REFERENCE_SECTION_MARKERS = (
    "overview",
    "sources",
    "pitfalls",
    "background",
    "when to use",
    "notes",
    "reference",
)
TOOLKIT_MARKERS = (
    "this skill provides",
    "library for",
    "import the module",
    "class",
    "function",
    "method",
    "api",
    "module",
)
HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$", re.MULTILINE)
FENCED_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)


def assess_compilability(package: SkillPackage, entry_skill: Path) -> dict:
    """Decide whether this skill can be meaningfully compiled into a workflow.

    Returns a dict with:
        compilable: bool
        kind: str
        strategy: workflow | dispatcher | reference
        confidence: low | medium | high
        reason: str
        suggestion: str
        signals: list[str]
    """
    content = entry_skill.read_text(encoding="utf-8") if entry_skill.exists() else ""
    lower = content.lower()

    command_blocks = _count_command_blocks(content)
    python_blocks = _count_python_blocks(content)
    headings = [heading.strip().lower() for heading in HEADING_RE.findall(content)]
    workflow_sections = sum(
        1 for heading in headings if any(marker in heading for marker in WORKFLOW_SECTION_MARKERS)
    )
    reference_sections = sum(
        1 for heading in headings if any(marker in heading for marker in REFERENCE_SECTION_MARKERS)
    )
    library_hits = sum(lower.count(marker) for marker in TOOLKIT_MARKERS)
    imperative_steps = len(
        re.findall(
            r"(?:^|\n)\s*(?:\d+\.|- )\s*\*\*(?:step|create|load|modify|save|verify)",
            lower,
        )
    )

    has_commands = command_blocks > 0
    has_scripts = any(asset.asset_type.value == "script" for asset in package.assets)
    has_config = any(
        asset.path.endswith(("setup.json", "package.json")) for asset in package.assets
    )

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    total_lines = len(lines)
    code_block_lines = _count_code_block_lines(content)
    prose_ratio = 1.0 - (code_block_lines / max(total_lines, 1))
    signals = _build_signals(
        has_scripts=has_scripts,
        has_config=has_config,
        command_blocks=command_blocks,
        python_blocks=python_blocks,
        workflow_sections=workflow_sections,
        reference_sections=reference_sections,
        library_hits=library_hits,
    )

    if not has_commands and not has_scripts and not has_config and prose_ratio > 0.9:
        return {
            "compilable": True,
            "kind": "pure_knowledge",
            "strategy": "reference",
            "confidence": "high",
            "reason": "Skill contains only documentation/prose with no executable commands, scripts, or setup config.",
            "suggestion": "Compile this skill with the reference adapter backend instead of the workflow backend.",
            "signals": signals,
        }

    interactive_markers = [
        "conversation",
        "chat",
        "dialogue",
        "persona",
        "tone of voice",
        "roleplay",
    ]
    interactive_hits = sum(1 for marker in interactive_markers if marker in lower)
    if interactive_hits >= 3 and not has_commands and not has_scripts:
        return {
            "compilable": True,
            "kind": "interactive_only",
            "strategy": "reference",
            "confidence": "high",
            "reason": "Skill describes interactive/conversational behavior without executable steps.",
            "suggestion": "Compile this skill with the reference adapter backend instead of the workflow backend.",
            "signals": signals,
        }

    if total_lines < 10 and not has_commands:
        return {
            "compilable": False,
            "kind": "too_short",
            "strategy": "workflow",
            "confidence": "low",
            "reason": f"Skill content is too short ({total_lines} lines) with no executable commands.",
            "suggestion": "Add concrete commands or workflow steps before attempting compilation.",
            "signals": signals,
        }

    dispatcher_score = 0
    workflow_score = 0
    reference_score = 0

    if has_scripts:
        dispatcher_score += 2
    if python_blocks >= 2:
        dispatcher_score += 2
    elif python_blocks == 1:
        dispatcher_score += 1
    if library_hits >= 2:
        dispatcher_score += 2
    elif library_hits == 1:
        dispatcher_score += 1

    if has_commands:
        workflow_score += 2
    if workflow_sections >= 2:
        workflow_score += 2
    elif workflow_sections == 1:
        workflow_score += 1
    if imperative_steps >= 3:
        workflow_score += 1

    if reference_sections >= 2:
        reference_score += 2
    elif reference_sections == 1:
        reference_score += 1
    if prose_ratio > 0.8 and not has_scripts and not has_commands:
        reference_score += 2

    if dispatcher_score >= workflow_score + 1 and dispatcher_score >= 3:
        return {
            "compilable": True,
            "kind": "toolkit_dispatcher",
            "strategy": "dispatcher",
            "confidence": "high" if dispatcher_score >= workflow_score + 2 else "medium",
            "reason": "Skill looks like a reusable toolkit/library with scripts or API examples; it should lower to a dispatcher artifact instead of a linear workflow.",
            "suggestion": "Prefer capability extraction and dispatcher lowering for this skill family.",
            "signals": signals,
        }

    if python_blocks >= 3 and command_blocks == 0 and dispatcher_score >= workflow_score:
        return {
            "compilable": True,
            "kind": "toolkit_dispatcher",
            "strategy": "dispatcher",
            "confidence": "medium",
            "reason": "Skill is dominated by Python/API examples and reads like a reusable library playbook; it is a better fit for dispatcher lowering than workflow synthesis.",
            "suggestion": "Prefer capability extraction and dispatcher lowering for this skill family.",
            "signals": signals,
        }

    if reference_score >= workflow_score + 2 and not has_scripts:
        return {
            "compilable": True,
            "kind": "reference_only",
            "strategy": "reference",
            "confidence": "medium",
            "reason": "Skill is dominated by reference material, mappings, or examples rather than executable task flow.",
            "suggestion": "Compile this skill with the reference adapter backend instead of the workflow backend.",
            "signals": signals,
        }

    if (
        has_scripts
        and python_blocks >= max(2, command_blocks + 1)
        and dispatcher_score >= workflow_score
    ):
        return {
            "compilable": True,
            "kind": "toolkit_dispatcher",
            "strategy": "dispatcher",
            "confidence": "medium",
            "reason": "Skill ships local scripts and is dominated by Python/API examples over shell orchestration; it is a better fit for dispatcher lowering.",
            "suggestion": "Prefer capability extraction and dispatcher lowering for this skill family.",
            "signals": signals,
        }

    if has_commands and has_scripts:
        kind = "tool_workflow"
    elif has_commands:
        kind = "mixed_workflow"
    elif has_scripts:
        kind = "tool_workflow"
    else:
        kind = "mixed_workflow"

    return {
        "compilable": True,
        "kind": kind,
        "strategy": "workflow",
        "confidence": "high" if workflow_score >= dispatcher_score else "medium",
        "reason": f"Skill has {command_blocks} shell-like command blocks, {'scripts' if has_scripts else 'no scripts'}, {'config' if has_config else 'no config'}.",
        "suggestion": "",
        "signals": signals,
    }


def _build_signals(
    *,
    has_scripts: bool,
    has_config: bool,
    command_blocks: int,
    python_blocks: int,
    workflow_sections: int,
    reference_sections: int,
    library_hits: int,
) -> list[str]:
    signals: list[str] = []
    if has_scripts:
        signals.append("bundled_scripts")
    if has_config:
        signals.append("config_files")
    if command_blocks:
        signals.append(f"shell_blocks={command_blocks}")
    if python_blocks:
        signals.append(f"python_blocks={python_blocks}")
    if workflow_sections:
        signals.append(f"workflow_sections={workflow_sections}")
    if reference_sections:
        signals.append(f"reference_sections={reference_sections}")
    if library_hits:
        signals.append(f"toolkit_markers={library_hits}")
    return signals


def _count_command_blocks(content: str) -> int:
    """Count fenced code blocks that look like shell commands."""
    count = 0
    for language, block in _iter_fenced_blocks(content):
        if language and language not in {"bash", "sh", "shell", "zsh", "console"}:
            continue
        lines = [
            line.strip()
            for line in block.strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if lines and any(_looks_like_shell_command(line) for line in lines):
            count += 1
    return count


def _count_python_blocks(content: str) -> int:
    count = 0
    for language, block in _iter_fenced_blocks(content):
        if language not in {"python", "py"}:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            count += 1
    return count


def _count_code_block_lines(content: str) -> int:
    """Count total lines inside fenced code blocks."""
    total = 0
    for _language, block in _iter_fenced_blocks(content):
        total += len([line for line in block.splitlines() if line.strip()])
    return total


def _iter_fenced_blocks(content: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for match in FENCED_BLOCK_RE.finditer(content):
        language = match.group("lang").strip().lower()
        blocks.append((language, match.group("body")))
    return blocks


def _looks_like_shell_command(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "//")):
        return False
    if stripped.startswith(COMMAND_PREFIXES):
        return True
    return bool(re.match(r"^(?:\./|[A-Za-z0-9_.-]+\s+-[A-Za-z])", stripped))
