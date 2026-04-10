"""Load and save config files."""

import json
import os
from pathlib import Path

from aeloon.core.config.schema import Config

# Track the active config path for multi-instance runs.
_current_config_path: Path | None = None


def set_config_path(path: Path) -> None:
    """Set the active config path."""
    global _current_config_path
    _current_config_path = path


def get_aeloon_home() -> Path:
    """Return the base Aeloon home directory."""
    env_home = os.environ.get("AELOON_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".aeloon"


def get_config_path() -> Path:
    """Return the config file path."""
    if _current_config_path:
        return _current_config_path
    return get_aeloon_home() / "config.json"


def load_config(config_path: Path | None = None) -> Config:
    """Load config from disk or return defaults."""
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Write config to disk."""
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict) -> dict:
    """Apply small config migrations in place."""
    # Move the legacy nested workspace flag to its current location.
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data
