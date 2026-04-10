"""Configuration schema for the AI4S science agent mode.

Extracted from ``aeloon/config/schema.py`` to support standalone plugin-owned
config validation via the Plugin SDK.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    """Base model matching ``aeloon.config.schema.Base`` (camelCase alias)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ArxivConfig(_Base):
    """Arxiv paper tool configuration."""

    enabled: bool = True
    jina_fallback: bool = False  # only use Jina for arxiv when explicitly enabled
    cache_dir: str = ""  # defaults to ~/.aeloon/arxiv_cache/
    fetch_timeout_s: float = 30.0  # arxiv API timeout
    pdf_timeout_s: float = 60.0  # PDF download timeout
    max_results: int = 10  # max search results
    rate_limit_interval_s: float = 3.0  # respect arxiv 1 req/3s


class GovernanceConfig(_Base):
    """Governance controls for the AI4S science agent."""

    enable_audit: bool = True
    enable_budget: bool = True
    risk_level: str = "green"  # green | yellow | red


class ScienceConfig(_Base):
    """Configuration for the AI4S science agent mode."""

    enabled: bool = False
    storage_dir: str = "~/.aeloon/science"
    default_budget_tokens: int = 50_000
    default_budget_seconds: int = 600
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
