from __future__ import annotations

import json
import os
import runpy
from unittest.mock import patch


def test_entrypoint_sets_litellm_local_cost_map_when_fast_enabled(tmp_path) -> None:
    config_dir = tmp_path / ".aeloon"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps({"agents": {"defaults": {"fast": True}}}),
        encoding="utf-8",
    )

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("aeloon.cli.commands.app"),
    ):
        runpy.run_module("aeloon.__main__", run_name="__not_main__")
        assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "true"
