"""Tests for PluginInstaller."""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

from aeloon.plugins._sdk.installer import PluginInstaller

MANIFEST_TEMPLATE = {
    "id": "aeloon.demoplugin",
    "name": "Demo Plugin",
    "version": "0.1.0",
    "description": "A test plugin",
    "author": "Test",
    "entry": "aeloon.plugins.demoplugin.plugin:DemoPlugin",
}


def _write_manifest(directory: Path, manifest: dict | None = None) -> Path:
    """Write a manifest into a directory and return the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    data = manifest or MANIFEST_TEMPLATE
    (directory / "aeloon.plugin.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return directory


def _create_zip_plugin(tmp_path: Path, manifest: dict | None = None) -> Path:
    """Create a zip archive containing a single plugin directory."""
    plugin_dir = tmp_path / "build" / "demoplugin"
    _write_manifest(plugin_dir, manifest)

    archive = tmp_path / "demoplugin.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in plugin_dir.rglob("*"):
            zf.write(f, f.relative_to(tmp_path / "build"))
    return archive


def _create_tar_plugin(tmp_path: Path, manifest: dict | None = None) -> Path:
    """Create a tar.gz archive containing a single plugin directory."""
    plugin_dir = tmp_path / "build" / "demoplugin"
    _write_manifest(plugin_dir, manifest)

    archive = tmp_path / "demoplugin.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(str(plugin_dir), arcname="demoplugin")
    return archive


class TestPluginInstallerVerify:
    def test_verify_valid_plugin(self, tmp_path: Path) -> None:
        plugin_dir = _write_manifest(tmp_path / "demoplugin")
        installer = PluginInstaller()
        result = installer.verify(plugin_dir, verify_import=False)
        assert result.status == "ok"
        assert result.plugin_id == "aeloon.demoplugin"
        assert result.version == "0.1.0"

    def test_verify_no_manifest(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "empty_plugin"
        plugin_dir.mkdir()
        installer = PluginInstaller()
        result = installer.verify(plugin_dir)
        assert result.status == "broken"
        assert "No aeloon.plugin.json" in result.error

    def test_verify_bad_manifest(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "bad_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "aeloon.plugin.json").write_text("{bad json", encoding="utf-8")
        installer = PluginInstaller()
        result = installer.verify(plugin_dir)
        assert result.status == "broken"

    def test_verify_invalid_id(self, tmp_path: Path) -> None:
        bad_manifest = {**MANIFEST_TEMPLATE, "id": "not-valid"}
        plugin_dir = _write_manifest(tmp_path / "bad_id", bad_manifest)
        installer = PluginInstaller()
        result = installer.verify(plugin_dir)
        assert result.status == "broken"


class TestPluginInstallerInstall:
    def test_install_zip(self, tmp_path: Path) -> None:
        archive = _create_zip_plugin(tmp_path)
        target = tmp_path / "plugins"
        installer = PluginInstaller()

        result = installer.install(archive, target, verify_import=False)
        assert result.status == "ok"
        assert result.plugin_id == "aeloon.demoplugin"
        assert (target / "demoplugin" / "aeloon.plugin.json").exists()

    def test_install_tar_gz(self, tmp_path: Path) -> None:
        archive = _create_tar_plugin(tmp_path)
        target = tmp_path / "plugins"
        installer = PluginInstaller()

        result = installer.install(archive, target, verify_import=False)
        assert result.status == "ok"
        assert result.plugin_id == "aeloon.demoplugin"

    def test_install_nonexistent_archive(self, tmp_path: Path) -> None:
        installer = PluginInstaller()
        result = installer.install(tmp_path / "missing.zip", tmp_path / "plugins")
        assert result.status == "broken"
        assert "not found" in result.error.lower()

    def test_install_no_manifest_in_archive(self, tmp_path: Path) -> None:
        # Create zip with a dir but no manifest
        build = tmp_path / "build" / "empty"
        build.mkdir(parents=True)
        archive = tmp_path / "empty.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("empty/dummy.txt", "hello")

        installer = PluginInstaller()
        result = installer.install(archive, tmp_path / "plugins")
        assert result.status == "broken"
        assert "aeloon.plugin.json" in result.error

    def test_install_bad_manifest_in_archive(self, tmp_path: Path) -> None:
        build = tmp_path / "build" / "badplugin"
        build.mkdir(parents=True)
        (build / "aeloon.plugin.json").write_text("{invalid", encoding="utf-8")
        archive = tmp_path / "bad.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for f in build.rglob("*"):
                zf.write(f, f.relative_to(tmp_path / "build"))

        installer = PluginInstaller()
        result = installer.install(archive, tmp_path / "plugins")
        assert result.status == "broken"
        assert "manifest" in result.error.lower()

    def test_install_multiple_top_dirs_rejected(self, tmp_path: Path) -> None:
        build = tmp_path / "build"
        (build / "dir_a").mkdir(parents=True)
        (build / "dir_b").mkdir(parents=True)
        archive = tmp_path / "multi.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("dir_a/dummy.txt", "a")
            zf.writestr("dir_b/dummy.txt", "b")

        installer = PluginInstaller()
        result = installer.install(archive, tmp_path / "plugins")
        assert result.status == "broken"
        assert "exactly one" in result.error.lower()

    def test_install_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "plugins"
        archive = _create_zip_plugin(tmp_path)

        installer = PluginInstaller()
        r1 = installer.install(archive, target, verify_import=False)
        assert r1.status == "ok"

        # Install again — should overwrite
        archive2 = _create_zip_plugin(tmp_path)
        r2 = installer.install(archive2, target, verify_import=False)
        assert r2.status == "ok"


class TestPluginInstallerRemove:
    def test_remove_workspace_plugin(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        plugin_dir = _write_manifest(workspace / "demoplugin")

        installer = PluginInstaller()
        assert installer.remove("aeloon.demoplugin", workspace) is True
        assert not plugin_dir.exists()

    def test_remove_nonexistent_plugin(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        installer = PluginInstaller()
        assert installer.remove("aeloon.missing", workspace) is False

    def test_remove_no_workspace_dir(self, tmp_path: Path) -> None:
        installer = PluginInstaller()
        assert installer.remove("aeloon.test", tmp_path / "nonexistent") is False

    def test_remove_skips_dirs_without_manifest(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        (workspace / "not_a_plugin").mkdir(parents=True)
        installer = PluginInstaller()
        assert installer.remove("aeloon.test", workspace) is False
