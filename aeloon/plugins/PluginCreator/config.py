"""Configuration schema for the PluginCreator plugin."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    """Base model matching aeloon.config.schema.Base (camelCase alias)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class PluginCreatorConfig(_Base):
    """Configuration for the PluginCreator plugin."""

    enabled: bool = False
    workspace_dir: str = "~/.aeloon/plugincreator/workspaces"
    default_maturity: Literal["prototype", "mvp", "production_ready"] = "mvp"
    plan_first: bool = True
