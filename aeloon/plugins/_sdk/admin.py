"""Shared plugin administration helpers for CLI and slash commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aeloon.plugins._sdk.installer import PluginInstaller
from aeloon.plugins._sdk.manifest import load_manifest
from aeloon.plugins._sdk.state_store import PluginState, PluginStateStore


@dataclass(frozen=True)
class PluginInventoryEntry:
    """One discovered plugin entry for CLI listing."""

    id: str
    name: str
    version: str
    status: str
    source: str


@dataclass(frozen=True)
class PluginAdminResult:
    """Simple result wrapper for plugin admin actions."""

    ok: bool
    message: str


def collect_installed_plugin_entries(
    *,
    bundled_dir: Path,
    workspace_dir: Path,
    state_store: PluginStateStore,
) -> list[PluginInventoryEntry]:
    """Scan bundled and workspace plugin directories."""
    entries: list[PluginInventoryEntry] = []

    for scan_dir, source in ((bundled_dir, "bundled"), (workspace_dir, "workspace")):
        if not scan_dir.is_dir():
            continue
        for child in sorted(scan_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "aeloon.plugin.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = load_manifest(manifest_path)
                state = state_store.get(manifest.id)
                enabled = state.enabled if state else True
                status = "deactivated" if not enabled else "ok"
                entries.append(
                    PluginInventoryEntry(
                        id=manifest.id,
                        name=manifest.name,
                        version=manifest.version,
                        status=status,
                        source=source,
                    )
                )
            except Exception as exc:
                entries.append(
                    PluginInventoryEntry(
                        id=child.name,
                        name=child.name,
                        version="?",
                        status=f"broken: {exc}",
                        source=source,
                    )
                )
    return entries


def collect_local_plugin_entries() -> list[PluginInventoryEntry]:
    """Collect plugin inventory from the local Aeloon home."""
    from aeloon.core.config.loader import get_aeloon_home

    aeloon_home = get_aeloon_home()
    return collect_installed_plugin_entries(
        bundled_dir=Path(__file__).resolve().parents[2] / "plugins",
        workspace_dir=aeloon_home / "plugins",
        state_store=PluginStateStore(aeloon_home / "plugin_state.json"),
    )


def suggest_plugin_entries(action: str) -> list[PluginInventoryEntry]:
    """Return action-aware plugin suggestions for CLI prompting."""
    entries = collect_local_plugin_entries()
    if action == "remove":
        return [entry for entry in entries if entry.source == "workspace"]
    if action == "activate":
        workspace_entries = [entry for entry in entries if entry.source == "workspace"]
        return [entry for entry in workspace_entries if entry.status != "ok"] or workspace_entries
    if action == "deactivate":
        workspace_entries = [entry for entry in entries if entry.source == "workspace"]
        return [entry for entry in workspace_entries if entry.status == "ok"] or workspace_entries
    if action == "error":
        return [entry for entry in entries if "broken" in entry.status] or entries
    return entries


def install_plugin_archive(
    *,
    archive: Path,
    workspace_dir: Path,
    state_store: PluginStateStore,
) -> PluginAdminResult:
    """Install one plugin archive and persist enabled state."""
    if not archive.exists():
        return PluginAdminResult(False, f"Archive not found: {archive}")

    installer = PluginInstaller()
    result = installer.install(archive, workspace_dir, verify_import=True)
    if result.status != "ok":
        return PluginAdminResult(False, f"Install failed: {result.error}")

    state_store.set(
        PluginState(
            plugin_id=result.plugin_id,
            installed_at=datetime.now().isoformat(),
            source=archive.name,
            enabled=True,
            version=result.version,
        )
    )
    return PluginAdminResult(
        True,
        (
            f"Installed: {result.name} ({result.plugin_id}) v{result.version}\n"
            f"Path: {result.install_path}\n"
            "Restart required to activate."
        ),
    )


def remove_workspace_plugin(
    *,
    name: str,
    workspace_dir: Path,
    state_store: PluginStateStore,
) -> PluginAdminResult:
    """Remove one workspace plugin and clear persisted state."""
    installer = PluginInstaller()
    if not installer.remove(name, workspace_dir):
        return PluginAdminResult(
            False,
            f"Plugin '{name}' not found in workspace directory. Bundled plugins cannot be removed.",
        )
    state_store.remove(name)
    return PluginAdminResult(True, f"Plugin '{name}' removed. Restart required to take effect.")


def set_plugin_enabled(
    *,
    name: str,
    enabled: bool,
    state_store: PluginStateStore | None,
) -> PluginAdminResult:
    """Toggle persisted enabled state for one plugin."""
    if state_store is None:
        return PluginAdminResult(False, "Plugin manager not available.")
    if state_store.set_enabled(name, enabled):
        action = "activated" if enabled else "deactivated"
        return PluginAdminResult(
            True, f"Plugin '{name}' {action}. Restart required to take effect."
        )
    return PluginAdminResult(False, f"Plugin '{name}' not found in state store.")


def format_runtime_plugin_list(plugin_manager: Any) -> str:
    """Render runtime plugin status lines for slash output."""
    lines = ["Plugins:", "─" * 50]

    for pid, record in sorted(plugin_manager.registry.plugins.items()):
        status = record.status
        version = record.manifest.version
        error_tag = f" — {record.error}" if record.error else ""
        lines.append(f"  {pid} v{version} [{status}]{error_tag}")

    if plugin_manager._state_store:
        registry_ids = set(plugin_manager.registry.plugins)
        for pid, state in sorted(plugin_manager._state_store.list_all().items()):
            if pid in registry_ids:
                continue
            if not state.enabled:
                lines.append(f"  {pid} v{state.version} [deactivated]")
            else:
                lines.append(f"  {pid} v{state.version} [broken]")

    if len(lines) == 2:
        lines.append("  (no plugins)")
    lines.append("")
    lines.append("Use `/plugin error <name>` for details on broken plugins.")
    return "\n".join(lines)


def format_plugin_errors(plugin_manager: Any, name: str) -> str:
    """Render one or all plugin error messages from the runtime registry."""
    if name:
        record = plugin_manager.registry.get_plugin(name)
        if not record:
            return f"Plugin '{name}' not found in registry."
        if not record.error:
            return f"Plugin '{name}' has no error (status: {record.status})."
        return f"Plugin: {name}\nStatus: {record.status}\nError:\n{record.error}"

    errors = [
        f"  {pid}: {record.error}"
        for pid, record in plugin_manager.registry.plugins.items()
        if record.error
    ]
    if not errors:
        return "No plugin errors."
    return "Plugin Errors:\n" + "\n".join(errors)


def format_state_store_report(state_store: PluginStateStore, name: str | None) -> str:
    """Render plugin state store details for CLI output."""
    if name:
        state = state_store.get(name)
        if not state:
            return f"Plugin '{name}' not found in state store."
        return (
            f"Plugin: {name}\n"
            f"Version: {state.version}\n"
            f"Source: {state.source}\n"
            f"Enabled: {state.enabled}"
        )

    states = state_store.list_all()
    if not states:
        return "No plugins tracked."
    return "\n".join(
        f"  {pid} v{state.version} — source={state.source} enabled={state.enabled}"
        for pid, state in sorted(states.items())
    )
