"""Tests for PluginStateStore."""

from __future__ import annotations

import json
from pathlib import Path

from aeloon.plugins._sdk.state_store import PluginState, PluginStateStore


def _make_state(**overrides):
    defaults = {
        "plugin_id": "aeloon.test",
        "installed_at": "2026-01-01T00:00:00",
        "source": "workspace",
        "enabled": True,
        "version": "1.0.0",
    }
    defaults.update(overrides)
    return PluginState(**defaults)


class TestPluginState:
    def test_defaults(self) -> None:
        s = PluginState(
            plugin_id="aeloon.test",
            installed_at="2026-01-01",
            source="bundled",
        )
        assert s.enabled is True
        assert s.version == ""


class TestPluginStateStore:
    def test_empty_on_missing_file(self, tmp_path: Path) -> None:
        store = PluginStateStore(tmp_path / "nonexistent.json")
        assert store.list_all() == {}

    def test_set_and_get(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store = PluginStateStore(path)
        state = _make_state()
        store.set(state)

        # Re-read from disk
        store2 = PluginStateStore(path)
        got = store2.get("aeloon.test")
        assert got is not None
        assert got.plugin_id == "aeloon.test"
        assert got.version == "1.0.0"
        assert got.enabled is True

    def test_set_enabled(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store = PluginStateStore(path)
        store.set(_make_state())

        assert store.set_enabled("aeloon.test", False) is True
        assert store.get("aeloon.test").enabled is False

        # Untracked plugin
        assert store.set_enabled("aeloon.missing", True) is False

    def test_remove(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store = PluginStateStore(path)
        store.set(_make_state())
        store.remove("aeloon.test")
        assert store.get("aeloon.test") is None

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store = PluginStateStore(path)
        store.remove("aeloon.nothing")  # should not raise

    def test_list_all(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store = PluginStateStore(path)
        store.set(_make_state(plugin_id="aeloon.alpha"))
        store.set(_make_state(plugin_id="aeloon.beta", version="2.0.0"))

        all_states = store.list_all()
        assert len(all_states) == 2
        assert "aeloon.alpha" in all_states
        assert "aeloon.beta" in all_states

    def test_recover_from_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("not valid json {{{{", encoding="utf-8")

        store = PluginStateStore(path)
        assert store.list_all() == {}

    def test_recover_from_non_dict_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        store = PluginStateStore(path)
        assert store.list_all() == {}

    def test_skip_malformed_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        data = {
            "aeloon.good": {
                "plugin_id": "aeloon.good",
                "installed_at": "2026-01-01",
                "source": "workspace",
                "enabled": True,
                "version": "1.0",
            },
            "aeloon.bad": "not a dict",
        }
        path.write_text(json.dumps(data), encoding="utf-8")

        store = PluginStateStore(path)
        assert store.get("aeloon.good") is not None
        assert store.get("aeloon.bad") is None

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "state.json"
        store = PluginStateStore(path)
        store.set(_make_state())
        assert path.exists()

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store = PluginStateStore(path)
        store.set(_make_state(version="1.0.0"))
        store.set(_make_state(version="2.0.0"))

        store2 = PluginStateStore(path)
        got = store2.get("aeloon.test")
        assert got.version == "2.0.0"

    def test_multiple_plugins_persist(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        store = PluginStateStore(path)
        store.set(_make_state(plugin_id="aeloon.a"))
        store.set(_make_state(plugin_id="aeloon.b"))
        store.set(_make_state(plugin_id="aeloon.c"))

        store2 = PluginStateStore(path)
        assert len(store2.list_all()) == 3
