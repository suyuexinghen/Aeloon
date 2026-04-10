"""Tests for KB agent tools (kb_search, kb_add)."""

from unittest.mock import AsyncMock

from aeloon.plugins.KnowledgeBase.tools.add_tool import KBAddTool
from aeloon.plugins.KnowledgeBase.tools.search_tool import KBSearchTool


def _mock_kb_service() -> AsyncMock:
    kb = AsyncMock()
    kb.query_formatted.return_value = "## Knowledge Base Results\n\nFound 1 result."
    kb.add_document.return_value = "doc_123"
    kb.add_text.return_value = "doc_456"
    return kb


async def test_search_tool_schema() -> None:
    kb = _mock_kb_service()
    tool = KBSearchTool(kb)
    assert tool.name == "kb_search"
    schema = tool.to_schema()
    assert schema["type"] == "function"
    assert "query" in schema["function"]["parameters"]["properties"]


async def test_search_tool_execute() -> None:
    kb = _mock_kb_service()
    tool = KBSearchTool(kb)
    result = await tool.execute(query="transformer attention")
    assert "Knowledge Base Results" in result
    kb.query_formatted.assert_called_once()


async def test_search_tool_no_query() -> None:
    kb = _mock_kb_service()
    tool = KBSearchTool(kb)
    result = await tool.execute()
    assert "Error" in result


async def test_add_tool_schema() -> None:
    kb = _mock_kb_service()
    tool = KBAddTool(kb)
    assert tool.name == "kb_add"
    assert "topic" in tool.parameters["properties"]


async def test_add_tool_file() -> None:
    kb = _mock_kb_service()
    tool = KBAddTool(kb)
    result = await tool.execute(topic="research", source_path="/path/to/paper.pdf")
    assert "doc_123" in result


async def test_add_tool_text() -> None:
    kb = _mock_kb_service()
    tool = KBAddTool(kb)
    result = await tool.execute(topic="notes", content="# Notes\n\nContent", title="My Notes")
    assert "doc_456" in result


async def test_add_tool_no_content() -> None:
    kb = _mock_kb_service()
    tool = KBAddTool(kb)
    result = await tool.execute(topic="notes")
    assert "Error" in result
