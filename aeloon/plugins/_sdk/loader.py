"""Plugin loader — dependency resolution, validation, import, and instantiation.

Uses Kahn's algorithm for topological sorting with cycle detection.
"""

from __future__ import annotations

import importlib

from loguru import logger

from aeloon.plugins._sdk.base import Plugin
from aeloon.plugins._sdk.discovery import PluginCandidate
from aeloon.plugins._sdk.manifest import (
    PluginManifest,
    validate_aeloon_version,
    validate_bins,
    validate_env,
)


class CircularDependencyError(Exception):
    """Raised when the plugin dependency graph contains a cycle."""

    def __init__(self, cycle_members: list[str]) -> None:
        self.cycle_members = cycle_members
        super().__init__(f"Circular dependency among: {', '.join(cycle_members)}")


class PluginLoadError(Exception):
    """Raised when a plugin class cannot be imported or instantiated."""


class PluginLoader:
    """Validates manifests, resolves load order, imports and instantiates plugins."""

    def resolve_load_order(self, candidates: list[PluginCandidate]) -> list[PluginCandidate]:
        """Topological sort (Kahn's algorithm) with cycle detection.

        Returns candidates in dependency-safe load order.
        Raises :class:`CircularDependencyError` if a cycle exists.
        Skips candidates whose required plugins are not in the candidate set.
        """
        by_id: dict[str, PluginCandidate] = {c.manifest.id: c for c in candidates}

        in_degree: dict[str, int] = {pid: 0 for pid in by_id}
        dependents: dict[str, list[str]] = {pid: [] for pid in by_id}

        skipped: set[str] = set()
        for pid, candidate in by_id.items():
            for dep in candidate.manifest.requires.plugins:
                if dep not in by_id:
                    logger.error(
                        "Plugin '{}' requires '{}' which is not available; skipping",
                        pid,
                        dep,
                    )
                    skipped.add(pid)
                    break
                in_degree[pid] += 1
                dependents[dep].append(pid)

        # Also skip any plugin that depends on a skipped plugin
        changed = True
        while changed:
            changed = False
            for pid in list(by_id):
                if pid in skipped:
                    continue
                for dep in by_id[pid].manifest.requires.plugins:
                    if dep in skipped:
                        skipped.add(pid)
                        changed = True
                        break

        for sid in skipped:
            in_degree.pop(sid, None)
            dependents.pop(sid, None)

        # Kahn's algorithm
        queue = [pid for pid, deg in in_degree.items() if deg == 0]
        sorted_ids: list[str] = []

        while queue:
            pid = queue.pop(0)
            sorted_ids.append(pid)
            for dep_id in dependents.get(pid, []):
                if dep_id in in_degree:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        queue.append(dep_id)

        remaining = [pid for pid in in_degree if pid not in sorted_ids]
        if remaining:
            raise CircularDependencyError(remaining)

        return [by_id[pid] for pid in sorted_ids]

    def validate_candidate(self, candidate: PluginCandidate) -> list[str]:
        """Validate a candidate's requirements.

        Returns a list of error strings (empty means valid).
        """
        errors: list[str] = []
        manifest = candidate.manifest

        if manifest.requires.aeloon_version:
            if not validate_aeloon_version(manifest.requires.aeloon_version):
                errors.append(
                    f"Aeloon version does not satisfy '{manifest.requires.aeloon_version}'"
                )

        missing_bins = validate_bins(manifest.requires.bins)
        if missing_bins:
            errors.append(f"Missing binaries: {', '.join(missing_bins)}")

        missing_env = validate_env(manifest.requires.env)
        if missing_env:
            errors.append(f"Missing env vars: {', '.join(missing_env)}")

        return errors

    def load_plugin_class(self, manifest: PluginManifest) -> type[Plugin]:
        """Import the plugin class from ``manifest.entry`` (``module.path:ClassName``)."""
        try:
            module_path, class_name = manifest.entry.rsplit(":", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if not (isinstance(cls, type) and issubclass(cls, Plugin)):
                raise PluginLoadError(f"{manifest.entry} is not a Plugin subclass")
            return cls
        except (ImportError, AttributeError, ValueError) as exc:
            raise PluginLoadError(f"Failed to import {manifest.entry}: {exc}") from exc

    def instantiate(self, cls: type[Plugin]) -> Plugin:
        """Create a :class:`Plugin` instance (no-arg constructor)."""
        try:
            return cls()
        except Exception as exc:
            raise PluginLoadError(f"Failed to instantiate {cls.__name__}: {exc}") from exc
