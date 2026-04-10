"""Persistent plugin state across restarts.

Stores install metadata and activation flags in ``~/.aeloon/plugin_state.json``.
Recoverable from corruption -- if the file is invalid, starts empty.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from loguru import logger


@dataclasses.dataclass
class PluginState:
    """Persisted metadata for a single plugin."""

    plugin_id: str
    installed_at: str  # ISO 8601
    source: str  # "bundled" | "workspace" | "<archive_name>"
    enabled: bool = True
    version: str = ""


class PluginStateStore:
    """Read/write plugin state from a JSON file.

    Uses atomic writes (temp file + rename) to prevent corruption.
    Gracefully recovers from corrupt or missing files.
    """

    def __init__(self, state_path: Path) -> None:
        self._path = state_path
        self._states: dict[str, PluginState] = {}
        self._load()

    def _load(self) -> None:
        """Load states from disk.  Recovers gracefully from corruption."""
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read plugin state file {}: {}", self._path, exc)
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt plugin state file {}: {}; starting empty", self._path, exc)
            return
        if not isinstance(data, dict):
            logger.warning("Plugin state file {} is not a dict; starting empty", self._path)
            return
        for pid, entry in data.items():
            if isinstance(entry, dict):
                try:
                    self._states[pid] = PluginState(
                        plugin_id=entry.get("plugin_id", pid),
                        installed_at=entry.get("installed_at", ""),
                        source=entry.get("source", "unknown"),
                        enabled=entry.get("enabled", True),
                        version=entry.get("version", ""),
                    )
                except (TypeError, KeyError):
                    logger.warning("Skipping malformed state entry for '{}'", pid)

    def _save(self) -> None:
        """Atomic write via temp file + rename."""
        data: dict[str, dict[str, Any]] = {}
        for pid, state in self._states.items():
            data[pid] = dataclasses.asdict(state)

        self._path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self._path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            tmp_path.replace(self._path)
        except OSError as exc:
            logger.error("Failed to write plugin state to {}: {}", self._path, exc)
            # Clean up temp file if rename failed
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, plugin_id: str) -> PluginState | None:
        """Return state for a plugin, or None if not tracked."""
        return self._states.get(plugin_id)

    def set(self, state: PluginState) -> None:
        """Insert or update state for a plugin, then persist."""
        self._states[state.plugin_id] = state
        self._save()

    def set_enabled(self, plugin_id: str, enabled: bool) -> bool:
        """Toggle the enabled flag.  Returns False if plugin not tracked."""
        state = self._states.get(plugin_id)
        if not state:
            return False
        state.enabled = enabled
        self._save()
        return True

    def remove(self, plugin_id: str) -> None:
        """Remove state entry for a plugin."""
        if plugin_id in self._states:
            del self._states[plugin_id]
            self._save()

    def list_all(self) -> dict[str, PluginState]:
        """Return a shallow copy of all tracked plugin states."""
        return dict(self._states)
