"""Plugin discovery — multi-source scanning with conflict resolution.

Discovers plugins from:

1. **Bundled** — ``aeloon/plugins/`` directory (lowest priority)
2. **Entry points** — ``aeloon.plugins`` setuptools group
3. **Workspace** — ``~/.aeloon/plugins/`` user directory
4. **Extra paths** — explicitly configured directories (highest priority)

Higher-priority sources override lower ones on ID collision.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
from pathlib import Path

from loguru import logger

from aeloon.plugins._sdk.manifest import PluginManifest, load_manifest


class PluginDiscoveryError(Exception):
    """Non-fatal error encountered during plugin discovery."""


@dataclasses.dataclass
class PluginCandidate:
    """A discovered plugin that has not yet been loaded."""

    manifest: PluginManifest
    source: int  # Priority: higher overrides lower
    source_label: str
    path: Path | None  # Filesystem path (None for entry_points)


# Source priority constants
SOURCE_BUNDLED = 10
SOURCE_ENTRY_POINTS = 20
SOURCE_WORKSPACE = 30
SOURCE_EXTRA = 40


class PluginDiscovery:
    """Discovers plugin candidates from multiple sources."""

    def __init__(
        self,
        bundled_dir: Path | None = None,
        workspace_dir: Path | None = None,
        extra_paths: list[Path] | None = None,
    ) -> None:
        self._bundled_dir = bundled_dir
        self._workspace_dir = workspace_dir
        self._extra_paths = extra_paths or []

    def discover_all(self) -> list[PluginCandidate]:
        """Scan all sources, resolve conflicts, return deduplicated candidates."""
        candidates: dict[str, PluginCandidate] = {}

        for candidate in self._scan_bundled():
            self._merge(candidates, candidate)
        for candidate in self._scan_entry_points():
            self._merge(candidates, candidate)
        for candidate in self._scan_workspace():
            self._merge(candidates, candidate)
        for candidate in self._scan_extra_paths():
            self._merge(candidates, candidate)

        return list(candidates.values())

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        candidates: dict[str, PluginCandidate],
        new: PluginCandidate,
    ) -> None:
        existing = candidates.get(new.manifest.id)
        if existing:
            if new.source >= existing.source:
                logger.warning(
                    "Plugin '{}' from {} overrides {} (higher priority)",
                    new.manifest.id,
                    new.source_label,
                    existing.source_label,
                )
                candidates[new.manifest.id] = new
            else:
                logger.debug(
                    "Plugin '{}' from {} ignored (lower priority than {})",
                    new.manifest.id,
                    new.source_label,
                    existing.source_label,
                )
        else:
            candidates[new.manifest.id] = new

    # ------------------------------------------------------------------
    # Source scanners
    # ------------------------------------------------------------------

    def _scan_bundled(self) -> list[PluginCandidate]:
        """Scan ``aeloon/plugins/`` for directories containing ``aeloon.plugin.json``."""
        if not self._bundled_dir:
            return []
        return self._scan_directory(self._bundled_dir, SOURCE_BUNDLED, "bundled")

    def _scan_entry_points(self) -> list[PluginCandidate]:
        """Scan the ``aeloon.plugins`` entry-points group."""
        candidates: list[PluginCandidate] = []
        try:
            eps = importlib.metadata.entry_points(group="aeloon.plugins")
        except Exception:
            return candidates

        for ep in eps:
            try:
                # Entry point value should point to a directory or module
                # Try loading as a module with a ``MANIFEST`` attribute first.
                module = ep.load()
                manifest_path = getattr(module, "MANIFEST_PATH", None)
                if manifest_path:
                    manifest = load_manifest(Path(manifest_path))
                    candidates.append(
                        PluginCandidate(
                            manifest=manifest,
                            source=SOURCE_ENTRY_POINTS,
                            source_label=f"entry_points:{ep.name}",
                            path=Path(manifest_path).parent,
                        )
                    )
            except Exception as exc:
                logger.warning("Failed to load entry point '{}': {}", ep.name, exc)
        return candidates

    def _scan_workspace(self) -> list[PluginCandidate]:
        """Scan ``~/.aeloon/plugins/`` for user-installed plugins."""
        if not self._workspace_dir:
            return []
        return self._scan_directory(self._workspace_dir, SOURCE_WORKSPACE, "workspace")

    def _scan_extra_paths(self) -> list[PluginCandidate]:
        """Scan configured extra directories."""
        candidates: list[PluginCandidate] = []
        for p in self._extra_paths:
            candidates.extend(self._scan_directory(p, SOURCE_EXTRA, f"extra:{p}"))
        return candidates

    # ------------------------------------------------------------------
    # Shared directory scanner
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_directory(
        base_dir: Path,
        source: int,
        label: str,
    ) -> list[PluginCandidate]:
        """Iterate subdirectories looking for ``aeloon.plugin.json``."""
        candidates: list[PluginCandidate] = []
        if not base_dir.is_dir():
            return candidates
        for child in sorted(base_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "aeloon.plugin.json"
            if manifest_path.is_file():
                try:
                    manifest = load_manifest(manifest_path)
                    candidates.append(
                        PluginCandidate(
                            manifest=manifest,
                            source=source,
                            source_label=f"{label}:{child.name}",
                            path=child,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load manifest from {}: {}",
                        manifest_path,
                        exc,
                    )
        return candidates
