"""Code generator: SkillGraph -> standalone resumable LangGraph workflow."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

from .manifest import extract_manifest
from .models import ExecutionKind, SkillGraph, Step, StepType
from .normalize import command_has_template, normalize_graph


def generate(
    graph: SkillGraph,
    output_path: str | Path,
    api_key: str = "",
    base_url: str = "https://openrouter.ai/api/v1",
    llm_model: str = "openai/gpt-5.4",
) -> Path:
    """
    Generate a standalone LangGraph Python file from a SkillGraph.

    Also ensures a sibling `skill_config.json` exists for runtime config.

    Args:
        graph: Analyzed SkillGraph DAG
        output_path: Where to write the .py file
        api_key: API key for LLM nodes
        base_url: OpenAI-compatible base URL
        llm_model: Model name for LLM nodes

    Returns:
        Path to the generated file
    """
    output_path = Path(output_path)
    graph = normalize_graph(graph)
    manifest = extract_manifest(graph)
    code = _build_code(graph, base_url, llm_model, output_path.stem)
    _verify_generated_code(graph, code)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code, encoding="utf-8")

    # Write runtime manifest
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest.save(manifest_path)

    # Runtime config file (no secrets are written by default)
    config_path = output_path.parent / "skill_config.json"
    if not config_path.exists():
        config_data = {
            "runtime": {
                "api_key": "",
                "base_url": base_url,
                "model": llm_model,
            }
        }
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    return output_path


def _verify_generated_code(graph: SkillGraph, code: str) -> None:
    """Fail fast if generated code is syntactically invalid or graph output wiring is unsafe."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Generated workflow code is invalid Python: {exc}") from exc

    if '"final_output": text' in code:
        raise ValueError(
            "Generated workflow still emits final_output from a non-terminal llm_generate template. "
            "Compile aborted because generated state updates are unsafe."
        )


def _build_code(sg: SkillGraph, base_url: str, llm_model: str, module_stem: str) -> str:
    """Assemble the full Python source."""

    tool_steps = [s for s in sg.steps if s.step_type != StepType.LLM_GENERATE]
    llm_steps = [s for s in sg.steps if s.step_type == StepType.LLM_GENERATE]

    # Build adjacency
    upstream = {s.id: sg.upstream_of(s.id) for s in sg.steps}
    downstream = {s.id: sg.downstream_of(s.id) for s in sg.steps}
    edge_desc = {(e.from_step, e.to_step): e.description for e in sg.edges}
    entries = sg.entry_nodes()

    # Generate node functions
    tool_fns = "\n".join(_gen_tool_node(s, upstream) for s in tool_steps)
    llm_fns = "\n".join(_gen_llm_node(s, upstream, sg) for s in llm_steps)
    conditional_steps = [
        s for s in sg.steps if s.step_type == StepType.CONDITION and len(downstream[s.id]) > 1
    ]
    conditional_pairs: set[tuple[str, str]] = set()
    conditional_wires: list[str] = []
    router_fns_list: list[str] = []

    for s in conditional_steps:
        rules = []
        for dst in downstream[s.id]:
            conditional_pairs.add((s.id, dst))
            rules.append(
                {
                    "label": f"to_{dst}",
                    "target": dst,
                    "description": edge_desc.get((s.id, dst), ""),
                }
            )
        if rules:
            router_fns_list.append(_gen_router_fn(s.id, rules))
            path_map = {r["label"]: r["target"] for r in rules}
            conditional_wires.append(
                f'    graph.add_conditional_edges("{s.id}", route_{s.id}, {repr(path_map)})'
            )

    router_fns = "\n".join(router_fns_list)
    node_order = [sid for layer in sg.topological_layers() for sid in layer]
    exit_nodes = sg.exit_nodes()

    # Generate wiring
    # Use direct edges by default. Condition steps with multiple outgoing
    # branches are wired via add_conditional_edges.
    wiring_lines = []
    for s in sg.steps:
        wiring_lines.append(f'    graph.add_node("{s.id}", node_{s.id})')
    wiring_lines.append("")

    # Entry nodes
    for nid in entries:
        wiring_lines.append(f'    graph.add_edge(START, "{nid}")')

    # Internal dependency edges
    seen_edges: set[tuple[str, str]] = set()
    for e in sg.edges:
        pair = (e.from_step, e.to_step)
        if pair in conditional_pairs:
            continue
        if e.from_step == e.to_step:
            # Guard against accidental self-loop edges from weak analyzers.
            continue
        if pair in seen_edges:
            continue
        seen_edges.add(pair)
        wiring_lines.append(f'    graph.add_edge("{e.from_step}", "{e.to_step}")')

    # Conditional branch edges
    wiring_lines.extend(conditional_wires)

    # Exit nodes
    for s in sg.steps:
        if not downstream[s.id]:
            wiring_lines.append(f'    graph.add_edge("{s.id}", END)')

    wiring = "\n".join(wiring_lines)

    # CLI args for global inputs
    cli_arg_lines: list[str] = []
    gi_set_lines: list[str] = [
        "    runtime_config = _load_runtime_config(args.config)",
        '    global_inputs = {"project_dir": project_dir, "_runtime_config": runtime_config}',
    ]
    for gi in sg.global_inputs:
        name = gi.name
        if name == "project_dir":
            continue
        arg = f"--{name.replace('_', '-')}"
        help_text = (gi.description or name).replace('"', "'")

        if gi.type in {"list"}:
            cli_arg_lines.append(
                f'    parser.add_argument("{arg}", nargs="*", default=None, help="{help_text}")'
            )
            gi_set_lines.append(
                f'    if args.{name} is not None:\n        global_inputs["{name}"] = args.{name}'
            )
        elif gi.type in {"bool", "boolean"}:
            cli_arg_lines.append(
                f'    parser.add_argument("{arg}", action="store_true", help="{help_text}")'
            )
            gi_set_lines.append(f'    if args.{name}:\n        global_inputs["{name}"] = True')
        elif gi.type == "int":
            cli_arg_lines.append(
                f'    parser.add_argument("{arg}", type=int, default=None, help="{help_text}")'
            )
            gi_set_lines.append(
                f'    if args.{name} is not None:\n        global_inputs["{name}"] = args.{name}'
            )
        elif gi.type == "float":
            cli_arg_lines.append(
                f'    parser.add_argument("{arg}", type=float, default=None, help="{help_text}")'
            )
            gi_set_lines.append(
                f'    if args.{name} is not None:\n        global_inputs["{name}"] = args.{name}'
            )
        elif gi.type == "dict":
            cli_arg_lines.append(
                f'    parser.add_argument("{arg}", default=None, help="{help_text} (JSON string)")'
            )
            gi_set_lines.append(f"    if args.{name} is not None:")
            gi_set_lines.append("        try:")
            gi_set_lines.append(f'            global_inputs["{name}"] = json.loads(args.{name})')
            gi_set_lines.append("        except Exception:")
            gi_set_lines.append(f'            global_inputs["{name}"] = args.{name}')
        else:
            cli_arg_lines.append(
                f'    parser.add_argument("{arg}", default=None, help="{help_text}")'
            )
            gi_set_lines.append(
                f'    if args.{name} is not None:\n        global_inputs["{name}"] = args.{name}'
            )

    cli_args_block = "\n".join(cli_arg_lines)
    gi_set_block = "\n".join(gi_set_lines)
    workflow_name = module_stem.removesuffix("_workflow").replace("-", "_")
    skill_meta = {
        "name": workflow_name,
        "description": sg.skill_description,
        "global_inputs": [field.model_dump() for field in sg.global_inputs],
    }

    code = f'''\
#!/usr/bin/env python3
"""
Auto-generated LangGraph workflow: {sg.skill_name}

Steps: {len(sg.steps)} ({len(tool_steps)} tool_call, {len(llm_steps)} llm_generate)
Edges: {len(sg.edges)}

Usage:
    # Standalone
    python {sg.skill_name.replace("-", "_")}_workflow.py --project /path/to/project
    python {sg.skill_name.replace("-", "_")}_workflow.py --config ./skill_config.json

    # Programmatic
    from {sg.skill_name.replace("-", "_")}_workflow import run_until_blocked
    result = run_until_blocked({{"global_inputs": {{"project_dir": "."}}, "step_results": {{}}, "error": None, "final_output": None}})
"""

import argparse
import asyncio
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────

DEFAULT_BASE_URL = "{base_url}"
DEFAULT_LLM_MODEL = "{llm_model}"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill_config.json")
SKILL_META = {repr(skill_meta)}

def _load_runtime_config(config_path: str | None) -> dict:
    path = config_path or os.getenv("SKILLGRAPH_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    path = os.path.abspath(os.path.expanduser(path))

    file_cfg = {{}}
    if os.path.exists(path):
        try:
            file_cfg = json.loads(open(path, "r", encoding="utf-8").read() or "{{}}")
        except Exception as e:
            logger.warning("[config] failed to parse %%s: %%s", path, e)

    runtime = file_cfg.get("runtime", file_cfg) if isinstance(file_cfg, dict) else {{}}
    if not isinstance(runtime, dict):
        runtime = {{}}

    api_key = (
        runtime.get("api_key")
        or os.getenv("SKILLGRAPH_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    base_url = runtime.get("base_url") or os.getenv("SKILLGRAPH_BASE_URL") or DEFAULT_BASE_URL
    model = runtime.get("model") or os.getenv("SKILLGRAPH_RUNTIME_MODEL") or DEFAULT_LLM_MODEL

    return {{
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "config_path": path,
    }}

def _runtime_cfg(state) -> dict:
    gi = state.get("global_inputs", {{}})
    cfg = gi.get("_runtime_config", {{}})
    if isinstance(cfg, dict):
        return cfg
    return _load_runtime_config(None)

def _runtime_llm_callable(state):
    gi = state.get("global_inputs", {{}})
    call = gi.get("_llm_callable")
    return call if callable(call) else None

MANIFEST_FILENAME = "{module_stem}.manifest.json"
SANDBOX_DIRNAME = "{module_stem}.sandbox"

def _sandbox_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), SANDBOX_DIRNAME)

def _sandbox_skill_dir() -> str:
    return os.path.join(_sandbox_dir(), "skill")

def _sandbox_env() -> dict:
    path = os.path.join(_sandbox_dir(), "env.json")
    if not os.path.exists(path):
        return {{}}
    try:
        return json.loads(open(path, "r", encoding="utf-8").read() or "{{}}")
    except Exception:
        return {{}}

def preflight_check(project_dir: str) -> list[str]:
    """Verify runtime dependencies before execution. Returns list of failure messages."""
    import shutil
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MANIFEST_FILENAME)
    sandbox_dir = _sandbox_dir()
    if not os.path.isdir(sandbox_dir):
        return ["Skill sandbox not found: " + sandbox_dir]
    bootstrap_path = os.path.join(sandbox_dir, "bootstrap.json")
    if not os.path.exists(bootstrap_path):
        return ["Skill sandbox bootstrap metadata not found: " + bootstrap_path]
    try:
        bootstrap = json.loads(open(bootstrap_path, "r", encoding="utf-8").read() or "{{}}")
    except Exception as exc:
        return [f"Failed to read sandbox bootstrap metadata: {{exc}}"]
    if bootstrap.get("status") != "ready":
        failures = []
        for check in bootstrap.get("checks", []):
            if not check.get("ok", False):
                failures.append(f"Sandbox bootstrap failed: {{check.get('name')}} — {{check.get('detail', '')}}")
        return failures or ["Skill sandbox bootstrap is not ready"]
    if not os.path.exists(manifest_path):
        return ["Runtime manifest not found: " + manifest_path]
    try:
        manifest = json.loads(open(manifest_path, "r", encoding="utf-8").read())
    except Exception as exc:
        return [f"Failed to read runtime manifest: {{exc}}"]
    sandbox_env = _sandbox_env()
    sandbox_bin = sandbox_env.get("SANDBOX_BIN_DIR", os.path.join(sandbox_dir, "bin"))
    sandbox_skill = _sandbox_skill_dir()
    failures = []
    for dep in manifest.get("dependencies", []):
        if not dep.get("required", True):
            continue
        kind = dep.get("kind", "")
        name = dep.get("name", "")
        if kind == "cli_binary":
            found = _which_with_sandbox(name, sandbox_bin, sandbox_skill)
            if not found:
                failures.append(f"CLI tool not found: {{name}} (checked sandbox/bin and system PATH)")
        elif kind == "script_file":
            if not os.path.isfile(os.path.join(sandbox_skill, name)):
                failures.append(f"Required script not found in sandbox: {{name}}")
        elif kind == "env_var":
            if not os.getenv(name):
                failures.append(f"Required environment variable not set: {{name}}")
        elif kind == "python_package":
            try:
                __import__(name)
            except ImportError:
                failures.append(f"Required Python package not installed: {{name}}")
        elif kind == "npm_runtime":
            if not os.path.isfile(os.path.join(sandbox_skill, name)):
                failures.append(f"npm runtime file not found in sandbox: {{name}}")
    return failures

def _which_with_sandbox(name: str, sandbox_bin: str, sandbox_skill: str) -> str:
    import shutil as _shutil
    node_bin = os.path.join(sandbox_skill, "node_modules", ".bin")
    extra = sandbox_bin + os.pathsep + node_bin
    old_path = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = extra + os.pathsep + old_path
        return _shutil.which(name) or ""
    finally:
        os.environ["PATH"] = old_path

# ── State ───────────────────────────────────────────────────

def _merge(a: dict, b: dict) -> dict:
    merged = dict(a)
    merged.update(b)
    return merged

class SkillState(TypedDict):
    global_inputs: dict[str, Any]
    step_results: Annotated[dict[str, Any], _merge]
    error: str | None
    final_output: str | None

# ── Helpers ─────────────────────────────────────────────────

def _shx(cmd: str, cwd: str | None = None, timeout: int = 60) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        out = r.stdout + ("\\n[stderr]\\n" + r.stderr if r.stderr else "")
        return r.returncode, out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return 124, f"(timed out after {{timeout}}s)"
    except Exception as e:
        return 1, f"(error: {{e}})"

def _run_argv(argv: list[str], cwd: str | None = None, timeout: int = 60) -> tuple[int, str]:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        out = r.stdout + ("\\n[stderr]\\n" + r.stderr if r.stderr else "")
        return r.returncode, out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return 124, f"(timed out after {{timeout}}s)"
    except Exception as e:
        return 1, f"(error: {{e}})"

def _sh(cmd: str, cwd: str | None = None, timeout: int = 60) -> str:
    _, out = _shx(cmd, cwd=cwd, timeout=timeout)
    return out

def _resolve_argv_value(token: str, inp: dict[str, Any]) -> str:
    if token.startswith("<") and token.endswith(">") and len(token) > 2:
        name = token[1:-1].strip().lower().replace("-", "_")
        alias_map = {{
            "url": "target_url",
            "action": "action_description",
            "instruction": "extract_instruction",
            "query": "observe_query",
        }}
        value = inp.get(name)
        if value is None and name in alias_map:
            value = inp.get(alias_map[name])
        if value is None:
            return token
        return str(value)
    return str(token)

def _bind_argv(argv: list[str], bindings: dict[str, str], inp: dict[str, Any]) -> list[str]:
    resolved: list[str] = []
    for token in argv:
        source = bindings.get(token)
        if source:
            value = inp.get(source)
            resolved.append(token if value is None else str(value))
            continue
        resolved.append(_resolve_argv_value(token, inp))
    return resolved

def _shx_argv(argv: list[str], inp: dict[str, Any], cwd: str | None = None, timeout: int = 60) -> tuple[int, str]:
    resolved = [_resolve_argv_value(part, inp) for part in argv]
    try:
        r = subprocess.run(resolved, shell=False, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        out = r.stdout + ("\\n[stderr]\\n" + r.stderr if r.stderr else "")
        return r.returncode, out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return 124, f"(timed out after {{timeout}}s)"
    except Exception as e:
        return 1, f"(error: {{e}})"

def _call_runtime_llm(llm_callable, system_prompt: str, user_prompt: str) -> str:
    result = llm_callable(system_prompt, user_prompt)
    if inspect.isawaitable(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return asyncio.run_coroutine_threadsafe(result, loop).result()
        return asyncio.run(result)
    return result

def _inputs(spec: list[dict], state: SkillState, upstreams: list[str]) -> dict:
    kw = {{}}
    gi = state.get("global_inputs", {{}})
    sr = state.get("step_results", {{}})
    for s in spec:
        n = s["name"]
        if n in gi:
            kw[n] = gi[n]
            continue
        for u in upstreams:
            if n in sr.get(u, {{}}):
                kw[n] = sr[u][n]
                break
        if n not in kw and not s.get("required", True):
            kw[n] = [] if s.get("type") == "list" else False if s.get("type") == "bool" else None
    return kw

def _missing_required(spec: list[dict], inp: dict) -> list[str]:
    return [s["name"] for s in spec if s.get("required", True) and s["name"] not in inp]

def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "ok", "approved", "allow"):
            return True
        if v in ("false", "0", "no", "n", "deny", "denied"):
            return False
    return None

def _choose_route(step_data: dict, rules: list[dict]) -> str:
    if not rules:
        return ""

    bools = {{}}
    for k, v in (step_data or {{}}).items():
        b = _to_bool(v)
        if b is not None:
            bools[str(k).lower()] = b

    # 1) Explicit pattern in edge description: conditional on var=true/false
    for r in rules:
        desc = (r.get("description") or "").lower()
        m = re.search(r"(?:conditional on|if)\\s+([a-zA-Z0-9_]+)\\s*(?:=|==)\\s*(true|false)", desc)
        if not m:
            continue
        var = m.group(1).lower()
        expect = m.group(2) == "true"
        if bools.get(var) is expect:
            return r["label"]

    # 2) Heuristic keywords per bool value
    pos_words = ("true", "proceed", "run", "needed", "required", "approved", "success", "high-risk", "high risk")
    neg_words = ("false", "skip", "already", "not", "fail", "failed", "low-risk", "low risk", "abort", "deny")

    for var, val in bools.items():
        best_label = ""
        best_score = -10**9
        for r in rules:
            desc = (r.get("description") or "").lower()
            if len(bools) > 1 and var not in desc:
                continue
            score = 0
            for w in pos_words:
                if w in desc:
                    score += 1
            for w in neg_words:
                if w in desc:
                    score -= 1
            if not val:
                score = -score
            if score > best_score:
                best_score = score
                best_label = r["label"]
        if best_label and best_score != 0:
            return best_label

    return rules[0]["label"]

# ── Tool Nodes ──────────────────────────────────────────────

{tool_fns}

# ── LLM Nodes ──────────────────────────────────────────────

{llm_fns}

# ── Routers ────────────────────────────────────────────────

{router_fns}

NODE_ORDER = {repr(node_order)}
EXIT_NODE_IDS = {repr(exit_nodes)}
NODE_FUNCTIONS = {{
{textwrap.indent(chr(10).join([f'"{s.id}": node_{s.id},' for s in sg.steps]), "    ")}
}}

# ── Graph ───────────────────────────────────────────────────

def build_graph():
    """Build and compile the LangGraph. Returns a CompiledStateGraph."""
    graph = StateGraph(SkillState)

{wiring}

    return graph.compile()

def _safe_jsonish(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {{str(k): _safe_jsonish(v) for k, v in value.items() if k != "_llm_callable"}}
    if isinstance(value, list):
        return [_safe_jsonish(v) for v in value]
    if isinstance(value, tuple):
        return [_safe_jsonish(v) for v in value]
    return repr(value)

def _envelope_from_result(result: dict, status: str = "completed", current_step: str | None = None, block: dict | None = None) -> dict:
    payload = {{
        "status": status,
        "current_step": current_step,
        "global_inputs": result.get("global_inputs", {{}}),
        "step_results": result.get("step_results", {{}}),
        "final_output": result.get("final_output"),
    }}
    if result.get("error") and not block:
        block = {{
            "message": result.get("error"),
            "details": {{"error": result.get("error")}},
            "suggested_actions": [
                "Inspect the failing step, repair the issue with normal tools, then resume the workflow.",
            ],
        }}
    if block:
        payload["block"] = block
    if result.get("error") and status == "completed":
        payload["status"] = "blocked"
    return payload

def _empty_state(state: SkillState | None = None) -> SkillState:
    base = dict(state or {{}})
    base.setdefault("global_inputs", {{}})
    base.setdefault("step_results", {{}})
    base.setdefault("error", None)
    base.setdefault("final_output", None)
    return base

def _apply_update(state: SkillState, update: dict) -> SkillState:
    merged = _empty_state(state)
    if not update:
        return merged
    if update.get("global_inputs"):
        gi = dict(merged.get("global_inputs", {{}}))
        gi.update(update.get("global_inputs", {{}}))
        merged["global_inputs"] = gi
    if update.get("step_results"):
        sr = dict(merged.get("step_results", {{}}))
        sr.update(update.get("step_results", {{}}))
        merged["step_results"] = sr
    if "error" in update:
        merged["error"] = update.get("error")
    if update.get("final_output") is not None:
        merged["final_output"] = update.get("final_output")
    return merged

def _completed_steps(state: SkillState) -> set[str]:
    return set((state.get("step_results") or {{}}).keys())

def _all_exit_nodes_done(state: SkillState) -> bool:
    completed = _completed_steps(state)
    return all(step_id in completed for step_id in EXIT_NODE_IDS)

def _pending_report(state: SkillState) -> dict:
    report = []
    completed = _completed_steps(state)
    for step_id in NODE_ORDER:
        if step_id in completed:
            continue
        node_fn = NODE_FUNCTIONS[step_id]
        try:
            update = node_fn(_empty_state(state))
        except Exception as exc:
            report.append({{"step_id": step_id, "status": "error", "detail": str(exc)}})
            continue
        if not update:
            report.append({{"step_id": step_id, "status": "waiting"}})
            continue
        if update.get("error"):
            report.append({{"step_id": step_id, "status": "blocked", "detail": update.get("error")}})
            continue
        report.append({{"step_id": step_id, "status": "ready"}})
    return {{"pending": report}}

def _blocked_envelope(state: SkillState, current_step: str | None, message: str, details: dict | None = None) -> dict:
    return _envelope_from_result(
        state,
        status="blocked",
        current_step=current_step,
        block={{
            "message": message,
            "details": _safe_jsonish(details or {{}}),
            "suggested_actions": [
                "Inspect the report, repair the issue with normal agent tools, then resume the workflow.",
            ],
        }},
    )

def run_until_blocked(state: SkillState) -> dict:
    state = _empty_state(state)
    failures = preflight_check(state.get("global_inputs", {{}}).get("project_dir", "."))
    if failures:
        return _blocked_envelope(
            state,
            None,
            "Workflow preflight failed because required runtime dependencies are missing.",
            {{"preflight": failures}},
        )
    progressed = True
    while progressed:
        progressed = False
        for step_id in NODE_ORDER:
            if step_id in _completed_steps(state):
                continue
            node_fn = NODE_FUNCTIONS[step_id]
            update = node_fn(state)
            if not update:
                continue
            if update.get("error"):
                state = _apply_update(state, update)
                return _blocked_envelope(state, step_id, str(update.get("error")), {{"update": update}})
            previous_completed = _completed_steps(state)
            state = _apply_update(state, update)
            if _completed_steps(state) != previous_completed or update.get("final_output") is not None:
                progressed = True
        if _all_exit_nodes_done(state):
            return _envelope_from_result(state, status="completed")
    pending = [step_id for step_id in NODE_ORDER if step_id not in _completed_steps(state)]
    details = {{"pending_steps": pending}}
    details.update(_pending_report(state))
    return _blocked_envelope(
        state,
        pending[0] if pending else None,
        "Workflow made no further progress and requires agent intervention.",
        details,
    )

def resume_from_state(state: SkillState) -> dict:
    resumed = _empty_state(state)
    resumed["error"] = None
    return run_until_blocked(resumed)

# ── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="{sg.skill_name}")
    parser.add_argument("--project", default=".", help="Project directory")
    parser.add_argument("--config", default=os.getenv("SKILLGRAPH_CONFIG_PATH", "skill_config.json"), help="Path to runtime config JSON")
{cli_args_block}
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project)
{gi_set_block}
    print(f"\\n=== {sg.skill_name} | {len(tool_steps)} tool + {len(llm_steps)} llm steps ===\\n")

    failures = preflight_check(project_dir)
    if failures:
        print("[PREFLIGHT FAILED] The following runtime dependencies are missing:")
        for f in failures:
            print(f"  - {{f}}")
        sys.exit(2)

    t0 = time.time()
    result = run_until_blocked({{
        "global_inputs": global_inputs,
        "step_results": {{}}, "error": None, "final_output": None,
    }})

    if result.get("status") == "blocked":
        print("[BLOCKED] Workflow needs repair before it can continue.")
        block = result.get("block") or {{}}
        if block.get("message"):
            print(block.get("message"))
        details = (block.get("details") or {{}}).get("missing_dependencies") or []
        for item in details:
            print(f"  - {{item}}")
        sys.exit(2)

    if err := result.get("error"):
        print(f"[ERROR] {{err}}\\nFALLBACK: revert to normal agent execution.")
        sys.exit(1)

    print(f"[OK] {{time.time()-t0:.1f}}s | LLM rounds saved: {len(tool_steps)} of {
        len(sg.steps)
    } steps")
    if out := result.get("final_output"):
        print("\\n" + out)

if __name__ == "__main__":
    main()
'''
    return code


# ── Node generators ─────────────────────────────────────────


def _gen_router_fn(step_id: str, rules: list[dict[str, str]]) -> str:
    rules_repr = repr(rules)
    return f'''
def route_{step_id}(state: SkillState) -> str:
    step_data = state.get("step_results", {{}}).get("{step_id}", {{}})
    return _choose_route(step_data, {rules_repr})
'''


def _gen_tool_node(step: Step, upstream: dict[str, list[str]]) -> str:
    """Generate a tool_call node function."""
    spec = repr([{"name": f.name, "type": f.type, "required": f.required} for f in step.inputs])
    ups = repr(upstream.get(step.id, []))
    body = _tool_body(step)
    name_literal = step_name_literal(step)
    comment_text = step_comment_text(step)

    return f'''
def node_{step.id}(state: SkillState) -> dict:
    # {comment_text}
    if state.get("error"):
        return {{}}
    if "{step.id}" in state.get("step_results", {{}}):
        return {{}}
    input_spec = {spec}
    inp = _inputs(input_spec, state, {ups})
    missing = _missing_required(input_spec, inp)
    if missing:
        logger.debug("[wait] {step.id} missing required inputs: %s", ", ".join(missing))
        return {{}}
    logger.info("[tool] %s", {name_literal})
    try:
        sandbox_dir = os.path.expanduser(state.get("global_inputs", {{}}).get("sandbox_dir", _sandbox_dir()))
        sandbox_env = _sandbox_env()
        sandbox_skill_dir = os.path.join(sandbox_dir, "skill")
        _sandbox_bin = sandbox_env.get("SANDBOX_BIN_DIR", os.path.join(sandbox_dir, "bin"))
        _node_bin = os.path.join(sandbox_skill_dir, "node_modules", ".bin")
        _extra_path = _sandbox_bin + os.pathsep + _node_bin
        if _extra_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _extra_path + os.pathsep + os.environ.get("PATH", "")
{textwrap.indent(body, "        ")}
        logger.info("[tool] done: %s", {name_literal})
        return {{"step_results": {{"{step.id}": result}}}}
    except Exception as e:
        return {{"error": f"{step.id}: {{e}}"}}
'''


def _tool_body(step: Step) -> str:
    """Generate the implementation body for a tool_call step.

    Extracts shell commands from the step description, or generates
    reasonable defaults based on the step's semantics.
    """
    desc = step.description
    outputs = {o.name: o for o in step.outputs}
    out_keys = list(outputs.keys())

    if step.step_type == StepType.CONDITION:
        return _condition_body(step)
    if step.step_type == StepType.DATA_TRANSFORM:
        return _data_transform_body(step)

    # Prefer grounded execution spec when available.
    if step.execution_spec:
        if step.execution_spec.kind == ExecutionKind.SHELL and step.execution_spec.argv:
            timeout = max(1, int(step.execution_spec.timeout_sec or 60))
            outputs_meta = [(o.name, o.type) for o in step.outputs]
            return f"""\
argv = {repr(step.execution_spec.argv)}
bindings = {repr(step.execution_spec.arg_bindings)}
resolved_argv = _bind_argv(argv, bindings, inp)
for idx, token in enumerate(resolved_argv):
    if token.startswith("scripts/") or token.startswith("./") or token.endswith(".py"):
        resolved_argv[idx] = os.path.join(sandbox_skill_dir, token.lstrip("./"))
exit_code, out = _run_argv(resolved_argv, cwd=sandbox_skill_dir, timeout={timeout})
if exit_code != 0:
    return {{"error": f"{step.id}: command failed (exit {{exit_code}}): {{out[:500]}}"}}
result = {{}}
for name, typ in {repr(outputs_meta)}:
    if typ in ("bool", "boolean"):
        result[name] = exit_code == 0
    elif typ == "int":
        result[name] = int(exit_code)
    elif typ == "dict":
        result[name] = {{"raw": out, "exit_code": exit_code}}
    else:
        result[name] = out
if not result:
    result = {{"output": out}}"""
        if step.execution_spec.kind == ExecutionKind.SHELL and step.execution_spec.command.strip():
            if command_has_template(step.execution_spec.command):
                out_name = out_keys[0] if out_keys else "output"
                return f'''\
result = {{"{out_name}": "Skipped: unresolved command template in execution_spec."}}'''
            timeout = max(1, int(step.execution_spec.timeout_sec or 60))
            commands = [ln.strip() for ln in step.execution_spec.command.splitlines() if ln.strip()]
            outputs_meta = [(o.name, o.type) for o in step.outputs]
            if len(commands) > 1:
                return f"""\
cmds = {repr(commands)}
exit_code = 0
parts = []
for cmd in cmds:
    rc, out = _shx(cmd, cwd=sandbox_skill_dir, timeout={timeout})
    if rc != 0:
        exit_code = rc
    if out and out != "(no output)":
        parts.append(out)
merged = "\\n---\\n".join(parts) if parts else "No findings."
result = {{}}
for name, typ in {repr(outputs_meta)}:
    lname = name.lower()
    if typ in ("bool", "boolean"):
        result[name] = exit_code == 0
    elif typ == "int":
        result[name] = int(exit_code)
    elif typ == "dict":
        result[name] = {{"raw": merged, "exit_code": exit_code}}
    else:
        result[name] = merged
if not result:
    result = {{"output": merged}}"""

            return f"""\
exit_code, out = _shx({json.dumps(commands[0])}, cwd=sandbox_skill_dir, timeout={timeout})
if exit_code != 0:
    return {{"error": f"{step.id}: command failed (exit {{exit_code}}): {{out[:500]}}"}}
result = {{}}
for name, typ in {repr(outputs_meta)}:
    lname = name.lower()
    if typ in ("bool", "boolean"):
        result[name] = exit_code == 0
    elif typ == "int":
        result[name] = int(exit_code)
    elif typ == "dict":
        result[name] = {{"raw": out, "exit_code": exit_code}}
    else:
        result[name] = out
if not result:
    result = {{"output": out}}"""

        if step.execution_spec.kind == ExecutionKind.NOOP:
            out_name = out_keys[0] if out_keys else "output"
            return f'''\
result = {{"{out_name}": "Skipped (no-op step)."}}'''

    # If the description contains shell commands (lines starting with common CLI patterns),
    # extract and execute them
    lines = desc.split("\n")
    shell_cmds = []
    for line in lines:
        line = line.strip()
        if line.startswith(
            (
                "grep ",
                "find ",
                "git ",
                "npm ",
                "pip",
                "go ",
                "cargo ",
                "docker ",
                "curl ",
                "openssl ",
                "ls ",
                "cat ",
                "echo ",
            )
        ):
            shell_cmds.append(line)

    if shell_cmds:
        # Execute extracted commands
        cmd_list_repr = repr(shell_cmds)
        out_name = out_keys[0] if out_keys else "output"
        return f'''\
cmds = {cmd_list_repr}
parts = []
for cmd in cmds:
    out = _sh(cmd, cwd=project_dir)
    if out and out != "(no output)":
        parts.append(out)
result = {{"{out_name}": "\\n---\\n".join(parts) if parts else "No findings."}}'''

    # Fallback: generate a generic implementation based on common step patterns
    sid = step.id.lower()

    if "detect" in sid and "project" in sid:
        return """\
manifests = {
    "package.json": "node", "package-lock.json": "node",
    "requirements.txt": "python", "pyproject.toml": "python",
    "go.mod": "go", "Cargo.toml": "rust", "Gemfile": "ruby", "pom.xml": "java",
}
found, langs = [], set()
for fn, lang in manifests.items():
    if os.path.exists(os.path.join(project_dir, fn)):
        found.append(fn); langs.add(lang)
result = {"project_files": found, "detected_languages": list(langs),
          "has_node": "node" in langs, "has_python": "python" in langs, "has_go": "go" in langs}"""

    if "npm" in sid and "audit" in sid:
        return """\
if not inp.get("has_node", False):
    result = {"npm_audit_results": "Skipped: not a Node.js project."}
else:
    result = {"npm_audit_results": _sh("npm audit --audit-level=high 2>&1", cwd=project_dir, timeout=120)}"""

    if "pip" in sid and "audit" in sid:
        return """\
if not inp.get("has_python", False):
    result = {"pip_audit_results": "Skipped: not a Python project."}
else:
    result = {"pip_audit_results": _sh("pip-audit 2>&1", cwd=project_dir, timeout=120)}"""

    if "govulncheck" in sid:
        return """\
if not inp.get("has_go", False):
    result = {"go_vuln_results": "Skipped: not a Go project."}
else:
    result = {"go_vuln_results": _sh("govulncheck ./... 2>&1", cwd=project_dir, timeout=120)}"""

    if "navigate" in sid and "browser" in sid:
        return r"""
url = str(inp.get("url", "") or "").strip()
if not url:
    seq = inp.get("command_sequence")
    if isinstance(seq, list):
        for item in seq:
            m = re.search(r"https?://[^\s'\"]+", str(item))
            if m:
                url = m.group(0)
                break
    elif isinstance(seq, str):
        m = re.search(r"https?://[^\s'\"]+", seq)
        if m:
            url = m.group(0)
if not url:
    m = re.search(r"https?://[^\s'\"]+", str(inp.get("user_request", "")))
    if m:
        url = m.group(0)

if not url:
    result = {"navigate_result": "Skipped: no URL available."}
else:
    result = {"navigate_result": _sh(f"browser navigate '{url}'", cwd=project_dir, timeout=90)}"""

    if "injection" in sid:
        return r"""
patterns = [
    ("SQL concat", r"(SELECT|INSERT|UPDATE|DELETE).*(\+|f\"|format\()"),
    ("Command exec", r"(exec\(|spawn\(|system\(|subprocess\.)"),
    ("Shell invoke", r"(os\.system\(|Runtime\.getRuntime\()"),
]
parts = []
for name, pat in patterns:
    out = _sh(f"grep -RInE '{pat}' --include='*.{{js,ts,py,go,java,php,rb,sh}}' . 2>/dev/null | grep -v 'node_modules\\|.git/' | head -20", cwd=project_dir)
    if out and out != "(no output)":
        parts.append(f"[{name}]\n{out}")
result = {"injection_matches": "\n\n".join(parts) if parts else "No obvious injection patterns."}"""

    if "xss" in sid:
        return r"""
pat = r"(innerHTML\s*=|dangerouslySetInnerHTML|v-html|document\.write\(|eval\()"
out = _sh(f"grep -RInE '{pat}' --include='*.{{js,ts,tsx,jsx,vue,html}}' . 2>/dev/null | grep -v 'node_modules\\|.git/' | head -30", cwd=project_dir)
result = {"xss_matches": out if out and out != "(no output)" else "No obvious XSS sinks."}"""

    if "gitignore" in sid:
        return """\
findings = []
gi = os.path.join(project_dir, ".gitignore")
if not os.path.exists(gi):
    findings.append("WARNING: No .gitignore file found")
else:
    content = open(gi).read()
    for e in [".env", "node_modules", "*.key", "*.pem"]:
        if e not in content: findings.append(f"MISSING from .gitignore: {e}")
tracked = _sh("git ls-files '*.env' '*.pem' '*.key' 2>/dev/null", cwd=project_dir)
if tracked and tracked != "(no output)": findings.append(f"Tracked sensitive files:\\n{tracked}")
result = {"gitignore_findings": "\\n".join(findings) if findings else "OK"}"""

    if "ssl" in sid:
        return """\
endpoints = inp.get("endpoints") or state.get("global_inputs", {}).get("endpoints", [])
if not endpoints:
    result = {"ssl_findings": "Skipped: no endpoints."}
else:
    parts = []
    for ep in endpoints[:5]:
        host = ep.replace("https://","").replace("http://","").split("/")[0]
        parts.append(f"[{host}]\\n" + _sh(f"echo | openssl s_client -connect {host}:443 -servername {host} 2>/dev/null | openssl x509 -noout -subject -dates 2>/dev/null", timeout=15))
    result = {"ssl_findings": "\\n\\n".join(parts)}"""

    if "header" in sid and ("http" in sid or "security" in sid):
        return """\
endpoints = inp.get("endpoints") or state.get("global_inputs", {}).get("endpoints", [])
if not endpoints:
    result = {"header_findings": "Skipped: no endpoints."}
else:
    parts = []
    for ep in endpoints[:5]:
        out = _sh(f"curl -sI \\'{ep}\\' --max-time 10 2>/dev/null | grep -i \\'strict-transport\\\\|content-security\\\\|x-frame\\\\|x-content-type\\'", timeout=15)
        parts.append(f"[{ep}]\\n{out or 'No security headers.'}")
    result = {"header_findings": "\\n\\n".join(parts)}"""

    if "permission" in sid or "file_perm" in sid:
        return """\
out = _sh("find . -type f -perm -o=w -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | head -20", cwd=project_dir)
result = {"permission_findings": out if out and out != "(no output)" else "File permissions OK."}"""

    if "classify" in sid and "risk" in sid:
        return """\
cmd = str(inp.get("command", "") or "").strip().lower()
is_docs = bool(inp.get("is_docs_lookup", False))
low_risk = {"install", "setup", "doctor", "status", "reset", "version", "tui", "dashboard", "update", "uninstall", "health", "configure", "completion", "logs", "config", "docs", "qr", "system", "sessions", "directory", "acp", "approvals", "security", "memory", "skills", "agents", "agent", "message", "msg"}
high_risk = {"cron", "browser", "webhooks", "dns", "nodes", "node", "devices", "pairing", "prose", "plugin", "hooks", "secrets", "sandbox"}
token = cmd.split()[0] if cmd else ""
result = {"is_high_risk": (not is_docs) and (token in high_risk)}"""

    if "secret" in sid and "grep" in sid:
        return r"""
patterns = [("AWS Key", r"AKIA[0-9A-Z]\{16\}"), ("Private Key", r"BEGIN.*PRIVATE KEY"),
            ("API Key", r"api[_-]\?key\|api[_-]\?secret"), ("Password", r"password\s*[:=]")]
parts = []
for name, pat in patterns:
    out = _sh(f"grep -rn '{pat}' --include='*.{{js,ts,py,go,env,yml,yaml,json}}' . 2>/dev/null | grep -v 'node_modules\\|.git/' | head -20", cwd=project_dir)
    if out and out != "(no output)": parts.append(f"[{name}]\n{out}")
result = {"secret_matches": "\n\n".join(parts) if parts else "No secrets detected."}"""

    if "git" in sid and "history" in sid:
        return r"""
out = _sh("git log -p --all -n 100 2>/dev/null | grep -n -i 'api.key\\|password\\|secret\\|token' | head -30", cwd=project_dir)
result = {"history_secret_matches": out}"""

    # Generic: run the description as a shell command if it looks like one, else return placeholder
    out_name = out_keys[0] if out_keys else "output"
    return f'''\
result = {{"{out_name}": "Skipped: no deterministic execution_spec available for step '{step.id}'."}}'''


def _condition_body(step: Step) -> str:
    outputs_meta = [(o.name, o.type) for o in step.outputs]
    desc = (step.description or "").lower()
    return f"""\
desc = {json.dumps(desc)}
outputs_meta = {repr(outputs_meta)}
result = {{}}

if "exit_code" in inp:
    bool_name = next((name for name, typ in outputs_meta if typ in ("bool", "boolean")), "success")
    int_name = next((name for name, typ in outputs_meta if typ == "int"), "")
    result[bool_name] = int(inp.get("exit_code", 1)) == 0
    if int_name:
        result[int_name] = int(inp.get("exit_code", 1))
elif "user_approved" in inp and any(k in desc for k in ("proceed", "approved", "approval")):
    bool_name = next((name for name, typ in outputs_meta if typ in ("bool", "boolean")), "proceed_with_execution")
    result[bool_name] = bool(inp.get("user_approved"))
else:
    base_bool = None
    for _k, _v in inp.items():
        if isinstance(_v, bool):
            base_bool = _v
            break

    for name, typ in outputs_meta:
        if typ in ("bool", "boolean"):
            val = bool(base_bool) if base_bool is not None else bool(inp)
            lname = name.lower()
            if base_bool is not None and "false" in desc and any(t in lname for t in ("need", "required", "missing")):
                val = not base_bool
            if base_bool is not None and any(t in lname for t in ("blocked", "deny", "abort")):
                val = not base_bool
            result[name] = val
        elif typ == "int":
            result[name] = int(inp.get(name, 0) or 0)
        elif typ == "list":
            result[name] = inp.get(name, [])
        else:
            result[name] = inp.get(name)

if not result:
    result = {{"condition": bool(inp)}}"""


def _data_transform_body(step: Step) -> str:
    outputs_meta = [(o.name, o.type) for o in step.outputs]
    desc = (step.description or "").lower()
    return f"""\
desc = {json.dumps(desc)}
result = {{}}

if any(name == "full_command" for name, _ in {repr(outputs_meta)}):
    cmd = str(inp.get("command", "")).strip()
    if not cmd and inp.get("subcommand_details"):
        cmd = str(inp.get("subcommand_details")).strip()
    if "openclaw.sh" in desc and cmd and not cmd.startswith("bash scripts/openclaw.sh"):
        cmd = f"bash scripts/openclaw.sh {{cmd}}"
    if inp.get("is_high_risk") and cmd and "OPENCLAW_WRAPPER_ALLOW_RISKY=" not in cmd:
        cmd = f"OPENCLAW_WRAPPER_ALLOW_RISKY=1 {{cmd}}"
    result["full_command"] = cmd

for name, typ in {repr(outputs_meta)}:
    if name in result:
        continue
    if name in inp:
        result[name] = inp[name]
        continue
    if typ in ("bool", "boolean"):
        result[name] = False
    elif typ == "int":
        result[name] = 0
    elif typ == "list":
        result[name] = []
    elif typ == "dict":
        result[name] = {{}}
    else:
        result[name] = None"""


def escaped_step_description_literal(step: Step) -> str:
    return repr(step.description or step.name)


def step_name_literal(step: Step) -> str:
    return repr(step.name)


def step_comment_text(step: Step) -> str:
    return (step.name or step.id).replace("\n", " ").replace("\r", " ")


def step_prompt_literal(step: Step) -> str:
    return repr(
        f"You are executing '{step.name}'.\nTask: {step.description}\nBe thorough and structured."
    )


def _gen_llm_node(step: Step, upstream: dict[str, list[str]], sg: SkillGraph) -> str:
    """Generate an llm_generate node function."""
    ups = upstream.get(step.id, [])
    is_terminal = step.id in set(sg.exit_nodes())
    spec = repr([{"name": f.name, "type": f.type, "required": f.required} for f in step.inputs])
    name_literal = step_name_literal(step)
    up_names = []
    for uid in ups:
        s = sg.get_step(uid)
        up_names.append(s.name if s else uid)

    # Output parsing
    parse_lines = []
    for o in step.outputs:
        if o.type == "int":
            parse_lines.append(
                '        nums = re.findall(r"\\b(\\d+)\\s+(?:issue|finding|vulnerabilit)", text, re.I)'
            )
            parse_lines.append(f'        output["{o.name}"] = int(nums[0]) if nums else 0')
        elif o.type == "list":
            parse_lines.append(
                f'        output["{o.name}"] = [l.strip().lstrip("- *") for l in text.split("\\n") if l.strip().startswith(("-","*"))][:20]'
            )
        else:
            parse_lines.append(f'        output["{o.name}"] = text')
    parse_code = "\n".join(parse_lines)
    comment_text = step_comment_text(step)

    return f'''
def node_{step.id}(state: SkillState) -> dict:
    # {comment_text}
    if state.get("error"):
        return {{}}
    if "{step.id}" in state.get("step_results", {{}}):
        return {{}}
    input_spec = {spec}
    inp = _inputs(input_spec, state, {repr(ups)})
    missing = _missing_required(input_spec, inp)
    if missing:
        logger.debug("[wait] {step.id} missing required inputs: %s", ", ".join(missing))
        return {{}}
    logger.info("[llm] %s", {name_literal})
    rcfg = _runtime_cfg(state)
    llm_callable = _runtime_llm_callable(state)
    llm = None if llm_callable else ChatOpenAI(
        model=rcfg.get("model") or DEFAULT_LLM_MODEL,
        api_key=rcfg.get("api_key") or "",
        base_url=rcfg.get("base_url") or DEFAULT_BASE_URL,
        temperature=0.0,
    )
    sr = state.get("step_results", {{}})
    ctx = []
    for uid, uname in zip({repr(ups)}, {repr(up_names)}):
        for k, v in sr.get(uid, {{}}).items():
            ctx.append(f"### {{uname}} — {{k}}\\n{{v}}")
    context = "\\n\\n".join(ctx)
    system_prompt = {step_prompt_literal(step)}
    user_prompt = "Data from upstream steps:\\n\\n" + context + "\\n\\n" + {escaped_step_description_literal(step)}
    try:
        if llm_callable:
            text = _call_runtime_llm(llm_callable, system_prompt, user_prompt)
        else:
            text = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]).content
        output = {{}}
{parse_code}
        logger.info("[llm] done: %s", {name_literal})
        result = {{"step_results": {{"{step.id}": output}}}}
        if {repr(is_terminal)}:
            result["final_output"] = text
        return result
    except Exception as e:
        if os.getenv("SKILLGRAPH_LLM_HARD_FAIL", "0") == "1":
            return {{"error": f"{step.id}: {{e}}"}}
        text = "[LLM fallback] " + {name_literal} + f" failed: {{e}}\\n\\nUpstream context (truncated):\\n{{context[:4000]}}"
        output = {{}}
{parse_code}
        logger.warning("[llm] fallback used: %s | %s", {name_literal}, e)
        result = {{"step_results": {{"{step.id}": output}}}}
        if {repr(is_terminal)}:
            result["final_output"] = text
        return result
'''
