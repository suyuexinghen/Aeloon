"""Skill package discovery and manifest generation."""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class AssetType(str, Enum):
    instruction = "instruction"
    script = "script"
    reference = "reference"
    config = "config"
    template = "template"
    example = "example"
    meta = "meta"
    other = "other"


class SkillAsset(BaseModel):
    path: str
    asset_type: AssetType
    size_bytes: int
    sha256: str


class SkillPackage(BaseModel):
    skill_root: str
    entry_skill: str
    slug: str
    package_hash: str
    assets: list[SkillAsset] = Field(default_factory=list)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SkillPackage":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def resolve_skill_input(skill_path: str | Path) -> tuple[Path, Path, str]:
    """Resolve skill root directory, entry SKILL.md, and stable slug.

    Accepts either:
    - path to a SKILL.md file
    - path to a skill directory containing SKILL.md
    """
    p = Path(skill_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Skill path not found: {p}")

    if p.is_file():
        skill_root = p.parent
        entry_skill = p
    else:
        skill_root = p
        direct = skill_root / "SKILL.md"
        if direct.exists():
            entry_skill = direct
        else:
            candidates = sorted(skill_root.rglob("SKILL.md"))
            if not candidates:
                raise FileNotFoundError(f"No SKILL.md found under: {skill_root}")
            if len(candidates) > 1:
                joined = "\n".join(f"- {c}" for c in candidates[:10])
                raise ValueError(f"Multiple SKILL.md files found; pass one explicitly:\n{joined}")
            entry_skill = candidates[0]

    slug = _slugify(skill_root.name or entry_skill.stem)
    return skill_root, entry_skill, slug


def build_skill_package(skill_path: str | Path) -> SkillPackage:
    """Scan a skill package and build a manifest with file hashes."""
    skill_root, entry_skill, slug = resolve_skill_input(skill_path)

    assets: list[SkillAsset] = []
    for file_path in sorted(skill_root.rglob("*")):
        if not file_path.is_file():
            continue
        if _should_skip_asset(file_path, skill_root):
            continue
        rel = file_path.relative_to(skill_root).as_posix()
        stat = file_path.stat()
        assets.append(
            SkillAsset(
                path=rel,
                asset_type=_classify_asset(rel),
                size_bytes=stat.st_size,
                sha256=_sha256_file(file_path),
            )
        )

    package_hash = _package_hash(assets)
    return SkillPackage(
        skill_root=str(skill_root),
        entry_skill=entry_skill.relative_to(skill_root).as_posix(),
        slug=slug,
        package_hash=package_hash,
        assets=assets,
    )


def _classify_asset(relative_path: str) -> AssetType:
    p = Path(relative_path)
    lower = relative_path.lower()
    name = p.name.lower()

    if lower == "skill.md":
        return AssetType.instruction
    if lower.startswith("scripts/"):
        return AssetType.script
    if lower.startswith("references/") or name == "reference.md":
        return AssetType.reference
    if lower.startswith("templates/"):
        return AssetType.template
    if lower.startswith("examples/") or name == "examples.md":
        return AssetType.example
    if name in {"setup.json", "package.json", "pyproject.toml", "requirements.txt", ".env.example"}:
        return AssetType.config
    if lower.startswith(".clawhub/") or name in {"_meta.json", "origin.json"}:
        return AssetType.meta

    if p.suffix.lower() in {".sh", ".bash", ".py", ".js", ".ts"}:
        return AssetType.script
    if p.suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".ini", ".conf"}:
        return AssetType.config
    if p.suffix.lower() in {".md", ".rst", ".txt"}:
        return AssetType.reference
    return AssetType.other


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _package_hash(assets: list[SkillAsset]) -> str:
    h = hashlib.sha256()
    for a in sorted(assets, key=lambda x: x.path):
        h.update(a.path.encode("utf-8"))
        h.update(a.sha256.encode("utf-8"))
    return h.hexdigest()


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return cleaned or "skill"


def _should_skip_asset(path: Path, skill_root: Path) -> bool:
    rel_parts = path.relative_to(skill_root).parts
    if any(part == "__pycache__" for part in rel_parts):
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    return False
