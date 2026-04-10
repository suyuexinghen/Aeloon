"""Extract a RuntimeManifest from a SkillGraph — declares all external runtime dependencies."""

from __future__ import annotations

import re
import shlex

from .models import ExecutionKind, RuntimeDependency, RuntimeManifest, SkillGraph, StepType

# Tokens that are shell builtins / not real external binaries
_BUILTIN_TOKENS = frozenset(
    {
        "echo",
        "cat",
        "test",
        "true",
        "false",
        "cd",
        "export",
        "source",
        ".",
        "set",
        "unset",
        "read",
        "printf",
        "eval",
        "exec",
        "exit",
        "return",
        "shift",
        "trap",
        "wait",
        "jobs",
        "fg",
        "bg",
        "kill",
        "umask",
        "alias",
        "unalias",
        "type",
        "builtin",
        "command",
        "local",
        "declare",
        "typeset",
        "readonly",
        "for",
        "do",
        "done",
        "if",
        "then",
        "fi",
        "elif",
        "else",
        "while",
        "until",
        "case",
        "esac",
    }
)

_SCRIPT_PATH_RE = re.compile(r"(?:^|/)(?:scripts|\.)/[a-zA-Z0-9_.-]+(?:\.sh|\.py)$")
_ARTIFACT_PATH_HINTS = (
    "screenshot-",
    "/screenshots/",
    "/output/",
    "/outputs/",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".jsonl",
    ".csv",
)


def extract_manifest(graph: SkillGraph) -> RuntimeManifest:
    """Analyse a SkillGraph and produce a RuntimeManifest listing all external dependencies."""
    deps: dict[tuple[str, str], RuntimeDependency] = {}

    def _add(kind: str, name: str, step_id: str, required: bool = True) -> None:
        key = (kind, name)
        if key in deps:
            if step_id not in deps[key].source_step_ids:
                deps[key].source_step_ids.append(step_id)
            return
        check = _make_check_command(kind, name)
        deps[key] = RuntimeDependency(
            kind=kind,
            name=name,
            required=required,
            source_step_ids=[step_id],
            check_command=check,
        )

    for step in graph.steps:
        # --- CLI binaries and script files from execution specs ---
        spec = step.execution_spec
        if spec:
            tokens: list[str] = []
            if spec.kind == ExecutionKind.SHELL and spec.argv:
                tokens = list(spec.argv)
            elif spec.kind == ExecutionKind.SHELL and spec.command:
                try:
                    tokens = shlex.split(spec.command)
                except ValueError:
                    tokens = spec.command.split()
            elif spec.kind == ExecutionKind.PYTHON:
                for token in spec.argv or []:
                    if (
                        token.endswith(".py")
                        or token.startswith("./")
                        or token.startswith("scripts/")
                    ):
                        _add("script_file", token, step.id)

            if tokens:
                binary = tokens[0]
                # Detect script file dependencies embedded in argv
                for token in tokens:
                    if _looks_like_artifact_output(token):
                        continue
                    if token.startswith("$") or token.startswith("${"):
                        continue
                    if _SCRIPT_PATH_RE.search(token) or token.endswith(".py"):
                        _add("script_file", token, step.id)
                # Detect CLI binary dependency
                if binary not in _BUILTIN_TOKENS and not binary.startswith("builtin:"):
                    if _looks_like_artifact_output(binary):
                        pass
                    elif binary.endswith(".py"):
                        _add("script_file", binary, step.id)
                    elif binary.startswith("./") or binary.startswith("scripts/"):
                        _add("script_file", binary, step.id)
                    else:
                        _add("cli_binary", binary, step.id)

            # npm runtime: if command is npm install/link, project needs package.json + node
            if spec.argv and len(spec.argv) >= 2 and spec.argv[0] == "npm":
                sub = spec.argv[1]
                if sub in ("install", "link", "ci"):
                    _add("npm_runtime", "package.json", step.id)

            # Env vars declared in execution_spec.env (currently a dead field, but model it)
            for env_name in spec.env or {}:
                _add("env_var", env_name, step.id)

        # --- Guard env vars ---
        for guard in step.guards:
            if guard.kind == "env_flag" and guard.env_var:
                _add("env_var", guard.env_var, step.id)

        # --- LLM service ---
        if step.step_type == StepType.LLM_GENERATE:
            _add("llm_service", "openai_compatible_api", step.id)

    # --- Fixed python package dependencies ---
    _add("python_package", "langgraph", "_framework")
    _add("python_package", "langchain_openai", "_framework")
    _add("python_package", "langchain_core", "_framework")

    sorted_deps = sorted(deps.values(), key=lambda d: (d.kind, d.name))
    return RuntimeManifest(dependencies=sorted_deps)


def _make_check_command(kind: str, name: str) -> str:
    if kind == "cli_binary":
        return f"which {name}"
    if kind == "script_file":
        return f"test -f {name}"
    if kind == "env_var":
        return f'test -n "${name}"'
    if kind == "python_package":
        return f'python3 -c "import {name}"'
    if kind == "npm_runtime":
        return f"test -f {name}"
    if kind == "llm_service":
        return ""
    return ""


def _looks_like_artifact_output(token: str) -> bool:
    """Heuristic to ignore output artifact paths that are not runtime prerequisites."""
    low = token.lower()
    if any(hint in low for hint in _ARTIFACT_PATH_HINTS):
        return True
    if "yyyy" in low or "mm" in low or "dd" in low or "thh" in low:
        return True
    return False
