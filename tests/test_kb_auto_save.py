"""Tests for KB auto-save hook."""

from unittest.mock import AsyncMock

from aeloon.plugins.KnowledgeBase.config import KBConfig
from aeloon.plugins.KnowledgeBase.services.auto_save import AutoSaveService


def _mock_kb_service() -> AsyncMock:
    kb = AsyncMock()
    kb.add_text.return_value = "doc_auto_1"
    return kb


async def test_auto_save_triggers() -> None:
    kb = _mock_kb_service()
    config = KBConfig(
        auto_save_enabled=True,
        auto_save_tools=["web_search"],
        auto_save_min_chars=10,
    )
    service = AutoSaveService(kb, config)

    await service.on_after_tool_call(
        tool_name="web_search",
        result="This is a long search result with enough content to be saved automatically.",
    )

    kb.add_text.assert_called_once()
    call_kwargs = kb.add_text.call_args
    assert call_kwargs.kwargs["topic"] == config.auto_save_topic


async def test_auto_save_wrong_tool() -> None:
    kb = _mock_kb_service()
    config = KBConfig(auto_save_enabled=True, auto_save_tools=["web_search"])
    service = AutoSaveService(kb, config)

    await service.on_after_tool_call(tool_name="other_tool", result="content" * 20)
    kb.add_text.assert_not_called()


async def test_auto_save_too_short() -> None:
    kb = _mock_kb_service()
    config = KBConfig(
        auto_save_enabled=True, auto_save_tools=["web_search"], auto_save_min_chars=100
    )
    service = AutoSaveService(kb, config)

    await service.on_after_tool_call(tool_name="web_search", result="short")
    kb.add_text.assert_not_called()


async def test_auto_save_disabled() -> None:
    kb = _mock_kb_service()
    config = KBConfig(auto_save_enabled=False)
    service = AutoSaveService(kb, config)

    await service.on_after_tool_call(tool_name="web_search", result="content" * 50)
    kb.add_text.assert_not_called()
