"""Tests for wiki implicit query grounding."""

from __future__ import annotations

from aeloon.plugins.Wiki.middleware.query_context import WikiQueryMiddleware
from aeloon.plugins.Wiki.models import EvidenceItem, QueryResult, RelatedEntryOption
from aeloon.plugins.Wiki.services.usage_mode import UsageModeStore


class _FakeQueryService:
    def __init__(self, result: QueryResult) -> None:
        self._result = result

    async def search(
        self,
        query: str,
        *,
        max_results: int = 3,
        max_related: int = 3,
    ) -> QueryResult:
        return self._result

    def format_evidence_block(
        self,
        query: str,
        evidence: list[EvidenceItem],
        related_entries: list[RelatedEntryOption],
    ) -> str:
        suffix = "\nRelated wiki entries:" if related_entries else ""
        return f"## Wiki Evidence\nInjected evidence.{suffix}"

    def format_gap_block(self, query: str) -> str:
        return "## Wiki Coverage Gap\nNo evidence."


async def test_injects_evidence_and_related_guidance_for_matching_query() -> None:
    captured: list[dict] = []

    async def _call_llm(messages: list[dict], tool_defs: list[dict]) -> str:
        captured.extend(messages)
        return "ok"

    middleware = WikiQueryMiddleware()
    await middleware.capture_message_context(
        session_key="cli:chat",
        channel="cli",
        chat_id="chat",
    )
    middleware.set_query_service(
        _FakeQueryService(
            QueryResult(
                primary_evidence=[
                    EvidenceItem(
                        entry_id="summary-transformer",
                        title="Transformer Note",
                        rel_path="wiki/summaries/summary-transformer.md",
                        summary="Attention summary",
                        score=10,
                        snippets=["attention"],
                    )
                ],
                related_entries=[
                    RelatedEntryOption(
                        entry_id="concept-attention",
                        title="Attention",
                        rel_path="wiki/concepts/concept-attention.md",
                        summary="More depth on attention.",
                        score=8,
                    )
                ],
            )
        )
    )
    middleware.set_usage_mode_store(UsageModeStore())
    await middleware.around_llm(
        [
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "Explain transformer attention?"},
        ],
        [],
        _call_llm,
    )

    assert "## Wiki Evidence" in captured[0]["content"]
    assert "Related wiki entries:" in captured[0]["content"]


async def test_gap_block_for_knowledge_query_without_evidence() -> None:
    captured: list[dict] = []

    async def _call_llm(messages: list[dict], tool_defs: list[dict]) -> str:
        captured.extend(messages)
        return "ok"

    middleware = WikiQueryMiddleware()
    await middleware.capture_message_context(
        session_key="cli:chat",
        channel="cli",
        chat_id="chat",
    )
    middleware.set_query_service(
        _FakeQueryService(QueryResult(primary_evidence=[], related_entries=[]))
    )
    store = UsageModeStore()
    store.set_mode("cli:chat", "local-only")
    middleware.set_usage_mode_store(store)
    await middleware.around_llm(
        [
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "What is flash attention?"},
        ],
        [],
        _call_llm,
    )

    assert "## Wiki Coverage Gap" in captured[0]["content"]


async def test_casual_chat_without_evidence_is_unchanged() -> None:
    captured: list[dict] = []

    async def _call_llm(messages: list[dict], tool_defs: list[dict]) -> str:
        captured.extend(messages)
        return "ok"

    middleware = WikiQueryMiddleware()
    await middleware.capture_message_context(
        session_key="cli:chat",
        channel="cli",
        chat_id="chat",
    )
    middleware.set_query_service(
        _FakeQueryService(QueryResult(primary_evidence=[], related_entries=[]))
    )
    middleware.set_usage_mode_store(UsageModeStore())
    await middleware.around_llm(
        [
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "hello there"},
        ],
        [],
        _call_llm,
    )

    assert captured[0]["content"] == "base system"


async def test_off_mode_skips_wiki_injection() -> None:
    captured: list[dict] = []

    async def _call_llm(messages: list[dict], tool_defs: list[dict]) -> str:
        captured.extend(messages)
        return "ok"

    middleware = WikiQueryMiddleware()
    await middleware.capture_message_context(
        session_key="cli:chat",
        channel="cli",
        chat_id="chat",
    )
    middleware.set_query_service(
        _FakeQueryService(
            QueryResult(
                primary_evidence=[
                    EvidenceItem(
                        entry_id="summary-transformer",
                        title="Transformer Note",
                        rel_path="wiki/summaries/summary-transformer.md",
                        summary="Attention summary",
                        score=10,
                        snippets=["attention"],
                    )
                ],
                related_entries=[],
            )
        )
    )
    store = UsageModeStore()
    store.set_mode("cli:chat", "off")
    middleware.set_usage_mode_store(store)
    await middleware.around_llm(
        [
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "Explain transformer attention?"},
        ],
        [],
        _call_llm,
    )

    assert captured[0]["content"] == "base system"
