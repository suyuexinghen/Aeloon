"""Plugin archive installation and removal.

Handles ``.zip`` and ``.tar.gz`` archives, validates the plugin manifest,
and optionally tests that the plugin class can be imported.
"""

from __future__ import annotations

import dataclasses
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

from loguru import logger

from aeloon.plugins._sdk.manifest import ManifestLoadError, load_manifest


@dataclasses.dataclass
class InstallResult:
    """Outcome of an install or verify operation."""

    plugin_id: str
    name: str
    version: str
    status: Literal["ok", "broken"]
    error: str | None
    install_path: Path


class PluginInstaller:
    """Install, verify, and remove plugin archives."""

    SUPPORTED_EXTENSIONS = (".zip", ".tar.gz", ".tgz", ".tar.bz2")
    MAX_ARCHIVE_SIZE = 100 * 1024 * 1024  # 100 MB

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def install(
        self, archive_path: Path, target_dir: Path, *, verify_import: bool = True
    ) -> InstallResult:
        """Extract an archive to *target_dir*, validate, and return result.

        Steps:
        1. Detect format and validate size.
        2. Extract to a temp directory.
        3. Validate: exactly one top-level dir with ``aeloon.plugin.json``.
        4. Move into *target_dir*.
        5. Verify manifest and import.
        """
        archive_path = archive_path.resolve()
        if not archive_path.exists():
            return InstallResult(
                plugin_id="",
                name="",
                version="",
                status="broken",
                error=f"Archive not found: {archive_path}",
                install_path=Path(),
            )

        if archive_path.stat().st_size > self.MAX_ARCHIVE_SIZE:
            return InstallResult(
                plugin_id="",
                name="",
                version="",
                status="broken",
                error="Archive exceeds 100 MB size limit",
                install_path=Path(),
            )

        # Extract to temp dir first
        with tempfile.TemporaryDirectory(prefix="aeloon_install_") as tmp:
            tmp_path = Path(tmp)
            try:
                self._extract(archive_path, tmp_path)
            except Exception as exc:
                return InstallResult(
                    plugin_id="",
                    name="",
                    version="",
                    status="broken",
                    error=f"Extraction failed: {exc}",
                    install_path=Path(),
                )

            # Find the single top-level directory
            top_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
            if len(top_dirs) != 1:
                return InstallResult(
                    plugin_id="",
                    name="",
                    version="",
                    status="broken",
                    error=f"Archive must contain exactly one top-level directory, found {len(top_dirs)}",
                    install_path=Path(),
                )

            plugin_tmp = top_dirs[0]
            manifest_path = plugin_tmp / "aeloon.plugin.json"
            if not manifest_path.is_file():
                return InstallResult(
                    plugin_id="",
                    name="",
                    version="",
                    status="broken",
                    error="Plugin directory must contain 'aeloon.plugin.json'",
                    install_path=Path(),
                )

            # Load manifest from temp location
            try:
                manifest = load_manifest(manifest_path)
            except ManifestLoadError as exc:
                return InstallResult(
                    plugin_id="",
                    name="",
                    version="",
                    status="broken",
                    error=f"Invalid manifest: {exc}",
                    install_path=Path(),
                )

            # Move into target dir
            dest = target_dir / plugin_tmp.name
            target_dir.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(plugin_tmp), str(dest))

            # Verify the installed copy
            result = self.verify(dest, verify_import=verify_import)
            if result.status == "ok":
                logger.info("Installed plugin '{}' v{}", manifest.id, manifest.version)
            else:
                logger.warning("Plugin '{}' installed but broken: {}", manifest.id, result.error)
            return result

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify(self, plugin_dir: Path, *, verify_import: bool = True) -> InstallResult:
        """Validate manifest and optionally test import for a plugin directory."""
        manifest_path = plugin_dir / "aeloon.plugin.json"
        if not manifest_path.is_file():
            return InstallResult(
                plugin_id="",
                name="",
                version="",
                status="broken",
                error=f"No aeloon.plugin.json in {plugin_dir}",
                install_path=plugin_dir,
            )

        try:
            manifest = load_manifest(manifest_path)
        except ManifestLoadError as exc:
            return InstallResult(
                plugin_id="",
                name="",
                version="",
                status="broken",
                error=f"Invalid manifest: {exc}",
                install_path=plugin_dir,
            )

        # Try importing the plugin class (optional)
        if verify_import:
            try:
                from aeloon.plugins._sdk.loader import PluginLoader

                loader = PluginLoader()
                cls = loader.load_plugin_class(manifest)
                if cls is None:
                    raise ImportError(f"Entry point {manifest.entry} resolved to None")
            except Exception as exc:
                return InstallResult(
                    plugin_id=manifest.id,
                    name=manifest.name,
                    version=manifest.version,
                    status="broken",
                    error=f"Import failed: {exc}",
                    install_path=plugin_dir,
                )

        return InstallResult(
            plugin_id=manifest.id,
            name=manifest.name,
            version=manifest.version,
            status="ok",
            error=None,
            install_path=plugin_dir,
        )

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(self, plugin_id: str, workspace_dir: Path) -> bool:
        """Remove a workspace-installed plugin by its ID.

        Scans *workspace_dir* for a plugin whose manifest ``id`` matches.
        Returns ``True`` if found and removed, ``False`` otherwise.
        """
        if not workspace_dir.is_dir():
            return False

        for child in workspace_dir.iterdir():
            if not child.is_dir():
                continue
            manifest_path = child / "aeloon.plugin.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = load_manifest(manifest_path)
                if manifest.id == plugin_id:
                    shutil.rmtree(child)
                    logger.info("Removed plugin '{}' from {}", plugin_id, child)
                    return True
            except ManifestLoadError:
                continue
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract(self, archive_path: Path, dest: Path) -> None:
        """Extract an archive to *dest*, rejecting path-traversal entries."""
        name = archive_path.name.lower()
        if name.endswith(".zip"):
            self._extract_zip(archive_path, dest)
        elif name.endswith((".tar.gz", ".tgz")):
            self._extract_tar(archive_path, dest, mode="r:gz")
        elif name.endswith(".tar.bz2"):
            self._extract_tar(archive_path, dest, mode="r:bz2")
        else:
            msg = f"Unsupported archive format: {archive_path.suffix}"
            raise ValueError(msg)

    @staticmethod
    def _extract_zip(archive_path: Path, dest: Path) -> None:
        """Extract a zip archive with path-traversal protection."""
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                # Reject path traversal
                if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                    msg = f"Unsafe path in archive: {info.filename}"
                    raise ValueError(msg)
            zf.extractall(dest)

    @staticmethod
    def _extract_tar(archive_path: Path, dest: Path, mode: str = "r:gz") -> None:
        """Extract a tar archive with path-traversal protection."""
        with tarfile.open(archive_path, mode) as tf:
            for member in tf.getmembers():
                # Reject absolute paths and traversal
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    msg = f"Unsafe path in archive: {member.name}"
                    raise ValueError(msg)
            tf.extractall(dest, filter="data")
