"""Plugin manifest model and validation utilities.

Provides Pydantic models for parsing ``aeloon.plugin.json`` manifests and
helpers for validating version constraints, binary dependencies, and
required environment variables.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ManifestLoadError(Exception):
    """Raised when a manifest file cannot be loaded or parsed."""


class PluginProvides(BaseModel):
    """Capabilities a plugin declares it will register."""

    model_config = ConfigDict(populate_by_name=True)

    commands: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    middlewares: list[str] = Field(default_factory=list)
    config_schema: str | None = None


class PluginRequires(BaseModel):
    """External requirements a plugin declares."""

    model_config = ConfigDict(populate_by_name=True)

    aeloon_version: str | None = None
    plugins: list[str] = Field(default_factory=list)
    bins: list[str] = Field(default_factory=list)
    env: list[str] = Field(default_factory=list)


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_ENTRY_PATTERN = re.compile(r"^[\w.]+:[\w]+$")


class PluginManifest(BaseModel):
    """Parsed representation of ``aeloon.plugin.json``."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    entry: str  # "module.path:ClassName"
    provides: PluginProvides = Field(default_factory=PluginProvides)
    requires: PluginRequires = Field(default_factory=PluginRequires)
    config: dict[str, Any] | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Enforce reverse-DNS style: at least two dot-separated lowercase segments."""
        if not _ID_PATTERN.match(v):
            msg = (
                f"Plugin id '{v}' must be reverse-DNS style with at least two "
                "dot-separated lowercase segments (e.g. 'aeloon.science')"
            )
            raise ValueError(msg)
        return v

    @field_validator("entry")
    @classmethod
    def validate_entry(cls, v: str) -> str:
        """Enforce ``module.path:ClassName`` format."""
        if not _ENTRY_PATTERN.match(v):
            msg = f"entry '{v}' must match 'module.path:ClassName'"
            raise ValueError(msg)
        return v


def load_manifest(path: Path) -> PluginManifest:
    """Load and validate a manifest from a JSON file path."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ManifestLoadError(f"Manifest not found: {path}") from exc
    except OSError as exc:
        raise ManifestLoadError(f"Cannot read manifest: {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestLoadError(f"Invalid JSON in {path}: {exc}") from exc

    try:
        return PluginManifest.model_validate(data)
    except Exception as exc:
        raise ManifestLoadError(f"Invalid manifest in {path}: {exc}") from exc


def validate_aeloon_version(spec_str: str | None) -> bool:
    """Check *spec_str* against the current Aeloon version.

    Returns ``True`` if the constraint is satisfied or if *spec_str* is
    ``None``/empty.  Uses :mod:`packaging.specifiers` when available;
    falls back to ``True`` (optimistic) if the library is absent.
    """
    if not spec_str:
        return True
    try:
        from importlib.metadata import version as pkg_version

        from packaging.specifiers import SpecifierSet

        current = pkg_version("aeloon")
        return SpecifierSet(spec_str).contains(current)
    except Exception:  # noqa: BLE001 – packaging may not be installed
        return True


def validate_bins(bins: list[str]) -> list[str]:
    """Return list of binaries from *bins* that are **not** found on ``$PATH``."""
    return [b for b in bins if shutil.which(b) is None]


def validate_env(env_vars: list[str]) -> list[str]:
    """Return list of environment variable names that are not set."""
    return [v for v in env_vars if v not in os.environ]
