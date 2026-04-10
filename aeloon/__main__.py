"""
Entry point for running aeloon as a module: python -m aeloon
"""

import json
import os
from pathlib import Path

try:
    _default_config_path = Path.home() / ".aeloon" / "config.json"
    if _default_config_path.exists():
        with open(_default_config_path, encoding="utf-8") as _f:
            _boot_cfg = json.load(_f)
        if _boot_cfg.get("agents", {}).get("defaults", {}).get("fast", False) is True:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
except Exception:
    pass

from aeloon.cli.commands import app

if __name__ == "__main__":
    app()
