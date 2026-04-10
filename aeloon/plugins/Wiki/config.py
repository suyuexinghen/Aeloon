"""Wiki plugin configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    """Local base to avoid circular imports with core config models."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class WikiConfig(_Base):
    """Root configuration for the wiki plugin."""

    enabled: bool = False
    repo_root: str = ""
    auto_query_enabled: bool = True
    supported_formats: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "md", "txt", "csv"]
    )
