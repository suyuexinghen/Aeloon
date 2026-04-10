"""Tests for KB @topic reference middleware."""

from unittest.mock import AsyncMock

from aeloon.plugins.KnowledgeBase.middleware.reference import KBReferenceMiddleware


def _mock_kb_service() -> AsyncMock:
    kb = AsyncMock()
    kb.query_formatted.return_value = "## Knowledge Base Results\n\nFound results."
    return kb


async def test_parse_single_topic() -> None:
    kb = _mock_kb_service()
    mw = KBReferenceMiddleware(kb)
    topics, clean = mw._parse_references("@research transformer attention")
    assert topics == ["research"]
    assert "transformer attention" in clean


async def test_parse_multiple_topics() -> None:
    kb = _mock_kb_service()
    mw = KBReferenceMiddleware(kb)
    topics, clean = mw._parse_references("@research @notes compare these")
    assert "research" in topics
    assert "notes" in topics
    assert "compare these" in clean


async def test_parse_at_kb() -> None:
    kb = _mock_kb_service()
    mw = KBReferenceMiddleware(kb)
    topics, clean = mw._parse_references("@KB search everything")
    assert topics == ["all"]
    assert "search everything" in clean


async def test_parse_no_reference() -> None:
    kb = _mock_kb_service()
    mw = KBReferenceMiddleware(kb)
    topics, clean = mw._parse_references("just a normal message")
    assert topics == []
    assert clean == "just a normal message"


async def test_process_with_reference() -> None:
    kb = _mock_kb_service()
    mw = KBReferenceMiddleware(kb)
    context: dict = {}
    result = await mw.process("@research transformer", context)
    assert "transformer" in result
    assert "kb_results" in context
    kb.query_formatted.assert_called_once()


async def test_process_without_reference() -> None:
    kb = _mock_kb_service()
    mw = KBReferenceMiddleware(kb)
    context: dict = {}
    await mw.process("no topic reference here", context)
    assert "kb_results" not in context
    kb.query_formatted.assert_not_called()
