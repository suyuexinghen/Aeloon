"""Create and verify per-skill sandbox environments at compile time."""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path

from .manifest import extract_manifest
from .models import RuntimeManifest, SandboxBootstrapResult, SandboxCheck, SkillGraph
from .package import SkillPackage

logger = logging.getLogger(__name__)


def bootstrap_sandbox(
    package: SkillPackage,
    graph: SkillGraph | None,
    output_path: str | Path,
    runtime_manifest: RuntimeManifest | None = None,
) -> SandboxBootstrapResult:
    """Create a self-contained sandbox next to the compiled workflow.

    This does real work: copies the skill package, runs npm install if needed,
    creates bin wrappers, sets executable bits on scripts, and verifies everything.
    """
    output_path = Path(output_path)
    sandbox_dir = output_path.with_suffix(".sandbox")
    skill_src = Path(package.skill_root)
    skill_dst = sandbox_dir / "skill"
    runtime_dir = sandbox_dir / "runtime"
    bin_dir = sandbox_dir / "bin"
    logs_dir = sandbox_dir / "logs"

    # Clean and recreate
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    skill_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_src, skill_dst)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checks: list[SandboxCheck] = []
    env: dict[str, str] = {}
    manifest = runtime_manifest or (extract_manifest(graph) if graph is not None else None)
    if manifest is None:
        raise ValueError(
            "bootstrap_sandbox requires either a graph or an explicit runtime manifest"
        )
    dep_names = {(dep.kind, dep.name) for dep in manifest.dependencies}

    # ── Node/npm bootstrap ──────────────────────────────────
    has_package_json = (skill_dst / "package.json").exists()
    needs_npm = ("npm_runtime", "package.json") in dep_names or has_package_json

    if needs_npm and has_package_json:
        checks.append(
            SandboxCheck(name="copy_package_json", ok=True, detail="package.json present")
        )

        # Actually run npm install inside sandbox/skill
        npm_ok, npm_detail = _run_cmd(["npm", "install", "--no-fund", "--no-audit"], cwd=skill_dst)
        checks.append(SandboxCheck(name="npm_install", ok=npm_ok, detail=npm_detail))

        if npm_ok:
            node_modules_bin = skill_dst / "node_modules" / ".bin"
            if node_modules_bin.exists():
                env["PATH_PREPEND"] = str(node_modules_bin)
                checks.append(
                    SandboxCheck(
                        name="node_modules_bin",
                        ok=True,
                        detail=str(node_modules_bin),
                    )
                )

                # Create bin wrappers in sandbox/bin for every binary in node_modules/.bin
                for entry in node_modules_bin.iterdir():
                    wrapper = bin_dir / entry.name
                    if not wrapper.exists():
                        wrapper.symlink_to(entry.resolve())
                checks.append(
                    SandboxCheck(
                        name="bin_wrappers",
                        ok=True,
                        detail=f"Created {len(list(bin_dir.iterdir()))} bin wrappers",
                    )
                )
            else:
                checks.append(
                    SandboxCheck(
                        name="node_modules_bin",
                        ok=False,
                        detail="npm install succeeded but node_modules/.bin missing",
                    )
                )

    # ── Script files: ensure executable bit ──────────────────
    for dep in manifest.dependencies:
        if dep.kind != "script_file":
            continue
        target = skill_dst / dep.name
        if target.exists():
            mode = target.stat().st_mode
            target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP)
            checks.append(SandboxCheck(name=f"script:{dep.name}", ok=True, detail=str(target)))
        else:
            checks.append(
                SandboxCheck(name=f"script:{dep.name}", ok=False, detail="missing in skill package")
            )

    # ── CLI binary availability ───────────────────────────────
    sandbox_path = f"{bin_dir}:{skill_dst / 'node_modules' / '.bin'}" if needs_npm else str(bin_dir)
    cli_deps = [dep for dep in manifest.dependencies if dep.kind == "cli_binary"]
    missing_clis: list[str] = []
    for dep in cli_deps:
        found = _which_in(dep.name, sandbox_path) or shutil.which(dep.name)
        if found:
            checks.append(SandboxCheck(name=f"cli:{dep.name}", ok=True, detail=found))
        else:
            missing_clis.append(dep.name)

    # For missing CLIs: try npm-installing the skill package into sandbox
    if missing_clis and shutil.which("npm"):
        npm_pkg = _detect_npm_package(skill_dst, graph.skill_name if graph else package.slug)
        if npm_pkg:
            logger.info(
                "Attempting sandbox npm install of %s for missing CLIs: %s", npm_pkg, missing_clis
            )
            ok, detail = _run_cmd(
                ["npm", "install", "--no-fund", "--no-audit", npm_pkg],
                cwd=skill_dst,
            )
            checks.append(SandboxCheck(name=f"npm_install:{npm_pkg}", ok=ok, detail=detail))
            if ok:
                node_bin = skill_dst / "node_modules" / ".bin"
                if node_bin.exists():
                    env["PATH_PREPEND"] = str(node_bin)
                    for entry in node_bin.iterdir():
                        wrapper = bin_dir / entry.name
                        if not wrapper.exists():
                            wrapper.symlink_to(entry.resolve())
                    checks.append(
                        SandboxCheck(
                            name="bin_wrappers_post_install",
                            ok=True,
                            detail=f"Created {len(list(bin_dir.iterdir()))} bin wrappers",
                        )
                    )

        # Re-check missing CLIs after install
        sandbox_path_updated = f"{bin_dir}:{skill_dst / 'node_modules' / '.bin'}"
        for name in missing_clis:
            found = _which_in(name, sandbox_path_updated) or shutil.which(name)
            if found:
                checks.append(SandboxCheck(name=f"cli:{name}", ok=True, detail=found))
            else:
                checks.append(
                    SandboxCheck(
                        name=f"cli:{name}",
                        ok=False,
                        detail="not found after sandbox install attempt",
                    )
                )
    else:
        for name in missing_clis:
            checks.append(
                SandboxCheck(
                    name=f"cli:{name}",
                    ok=False,
                    detail="not found in sandbox or system PATH",
                )
            )

    # ── Finalize env ────────────────────────────────────────
    env.setdefault("PATH_PREPEND", str(bin_dir))
    env["SANDBOX_SKILL_DIR"] = str(skill_dst)
    env["SANDBOX_BIN_DIR"] = str(bin_dir)

    # Status: ready if no hard failures (ignore missing system CLIs as warnings)
    hard_failures = [
        c for c in checks if not c.ok and c.name.startswith(("npm_install", "copy_", "script:"))
    ]
    status = "failed" if hard_failures else "ready"

    (sandbox_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    result = SandboxBootstrapResult(
        skill_slug=package.slug,
        sandbox_dir=str(sandbox_dir),
        status=status,
        checks=checks,
        env=env,
    )
    result.save(sandbox_dir / "bootstrap.json")

    # Log summary
    ok_count = sum(1 for c in checks if c.ok)
    fail_count = sum(1 for c in checks if not c.ok)
    logger.info(
        "Sandbox bootstrap %s for %s: %d ok, %d failed — %s",
        status,
        package.slug,
        ok_count,
        fail_count,
        sandbox_dir,
    )
    for c in checks:
        level = "INFO" if c.ok else "WARNING"
        logger.log(
            logging.getLevelName(level), "  [%s] %s: %s", "OK" if c.ok else "FAIL", c.name, c.detail
        )

    return result


def _run_cmd(argv: list[str], cwd: Path, timeout: int = 120) -> tuple[bool, str]:
    """Run a command and return (success, detail)."""
    try:
        r = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout,
        )
        if r.returncode == 0:
            return True, f"exit 0 ({len(r.stdout)} bytes stdout)"
        return False, f"exit {r.returncode}: {(r.stderr or r.stdout)[:500]}"
    except FileNotFoundError:
        return False, f"command not found: {argv[0]}"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as exc:
        return False, str(exc)


def _detect_npm_package(skill_dst: Path, skill_name: str) -> str:
    """Try to figure out what npm package to install for this skill.

    Strategy: look for command names in SKILL.md metadata, then try
    common npm package name patterns until one resolves on the registry.
    """
    import re

    command_names: list[str] = []

    # 1. SKILL.md frontmatter metadata for command hints
    skill_md = skill_dst / "SKILL.md"
    if skill_md.exists():
        try:
            content = skill_md.read_text(encoding="utf-8")
            meta_match = re.search(
                r'"requires"\s*:\s*\{[^}]*"commands"\s*:\s*\[([^\]]*)\]', content
            )
            if meta_match:
                raw = meta_match.group(1).replace('"', "").replace("'", "").split(",")
                command_names.extend(c.strip() for c in raw if c.strip())
        except Exception:
            pass

    # 2. _meta.json slug
    meta = skill_dst / "_meta.json"
    slug = ""
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            slug = data.get("slug", "")
        except Exception:
            pass

    # Build candidate package names to try
    candidates: list[str] = []
    for cmd in command_names:
        candidates.append(cmd)  # e.g. "agent-browser"
        candidates.append(f"agent-{cmd}")  # e.g. "agent-browser" from "browser"
    if slug:
        candidates.append(slug)
    candidates.append(skill_name)

    # De-dup preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)

    # Try each candidate on npm registry
    for pkg in unique:
        ok, _ = _run_cmd(["npm", "view", pkg, "name"], cwd=skill_dst, timeout=15)
        if ok:
            logger.info("Detected npm package for skill '%s': %s", skill_name, pkg)
            return pkg

    return ""


def _which_in(name: str, extra_path: str) -> str | None:
    """Like shutil.which but with extra PATH directories prepended."""
    original = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = extra_path + os.pathsep + original
        return shutil.which(name)
    finally:
        os.environ["PATH"] = original
