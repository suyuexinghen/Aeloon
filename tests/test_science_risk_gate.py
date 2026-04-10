"""Tests for RiskGateMiddleware and GovernanceConfig (SP6 P6.3/P6.4)."""

from __future__ import annotations

import pytest

from aeloon.plugins.ScienceResearch.config import GovernanceConfig, ScienceConfig
from aeloon.plugins.ScienceResearch.middleware.risk_gate import (
    RiskClassification,
    RiskGateMiddleware,
    RiskLevel,
)

# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------


def test_risk_level_values() -> None:
    assert RiskLevel.GREEN == "green"
    assert RiskLevel.YELLOW == "yellow"
    assert RiskLevel.RED == "red"


# ---------------------------------------------------------------------------
# RiskClassification stub
# ---------------------------------------------------------------------------


def test_classify_always_returns_green() -> None:
    clf = RiskClassification()
    assert clf.classify("find papers on perovskite solar cells", ["energy"]) == RiskLevel.GREEN
    assert clf.classify("synthesize quantum computing survey", []) == RiskLevel.GREEN


# ---------------------------------------------------------------------------
# RiskGateMiddleware — around_llm / around_tool
# ---------------------------------------------------------------------------


async def test_green_passthrough_llm() -> None:
    gate = RiskGateMiddleware(RiskLevel.GREEN)

    async def fake_llm(messages: list, tools: list) -> str:
        return "llm_response"

    result = await gate.around_llm([], [], fake_llm)
    assert result == "llm_response"


async def test_green_passthrough_tool() -> None:
    gate = RiskGateMiddleware(RiskLevel.GREEN)

    async def fake_tool() -> str:
        return "tool_result"

    result = await gate.around_tool("my_tool", {}, fake_tool)
    assert result == "tool_result"


async def test_yellow_logs_warning_and_passes(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    gate = RiskGateMiddleware(RiskLevel.YELLOW)

    async def fake_llm(messages: list, tools: list) -> str:
        return "ok"

    with caplog.at_level(logging.WARNING):
        result = await gate.around_llm([], [], fake_llm)
    assert result == "ok"


async def test_red_raises_not_implemented() -> None:
    gate = RiskGateMiddleware(RiskLevel.RED)

    async def fake_llm(messages: list, tools: list) -> str:
        return "should not reach"

    with pytest.raises(NotImplementedError, match="human approval"):
        await gate.around_llm([], [], fake_llm)


async def test_red_tool_raises_not_implemented() -> None:
    gate = RiskGateMiddleware(RiskLevel.RED)

    async def fake_tool() -> str:
        return "result"

    with pytest.raises(NotImplementedError):
        await gate.around_tool("tool", {}, fake_tool)


# ---------------------------------------------------------------------------
# GovernanceConfig defaults
# ---------------------------------------------------------------------------


def test_governance_config_defaults() -> None:
    cfg = GovernanceConfig()
    assert cfg.enable_audit is True
    assert cfg.enable_budget is True
    assert cfg.risk_level == "green"


def test_science_config_has_governance() -> None:
    cfg = ScienceConfig()
    assert isinstance(cfg.governance, GovernanceConfig)
    assert cfg.governance.enable_audit is True


def test_science_config_governance_override() -> None:
    cfg = ScienceConfig(governance=GovernanceConfig(enable_audit=False, risk_level="yellow"))
    assert cfg.governance.enable_audit is False
    assert cfg.governance.risk_level == "yellow"
