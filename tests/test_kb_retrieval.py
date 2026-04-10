"""Tests for KB retrieval engine and keyword search."""

from aeloon.plugins.KnowledgeBase.models import (
    Document,
    DocumentChunk,
    QueryResponse,
    RetrievalResult,
)
from aeloon.plugins.KnowledgeBase.retrieval.formatter import format_for_agent
from aeloon.plugins.KnowledgeBase.retrieval.keyword import KeywordSearch


class TestKeywordSearch:
    def test_extract_query_keywords(self) -> None:
        ks = KeywordSearch()
        kw = ks.extract_query_keywords("What is the transformer attention mechanism?")
        assert "transformer" in kw
        assert "attention" in kw
        assert "mechanism" in kw

    def test_extract_query_keywords_chinese(self) -> None:
        ks = KeywordSearch()
        kw = ks.extract_query_keywords("深度学习的应用")
        assert any("深度" in k or "学习" in k for k in kw)

    def test_search_chunks(self) -> None:
        ks = KeywordSearch()
        chunks = [
            DocumentChunk(
                content="The transformer uses attention mechanisms for sequence processing."
            ),
            DocumentChunk(content="BERT is a bidirectional encoder representation model."),
        ]
        results = ks.search_chunks(chunks, ["transformer", "attention"])
        assert len(results) == 1
        assert results[0].score > 0

    def test_search_chunks_min_match(self) -> None:
        ks = KeywordSearch()
        chunks = [
            DocumentChunk(content="The transformer uses attention mechanisms."),
        ]
        # min_match=3 means we need 3 keyword hits
        results = ks.search_chunks(chunks, ["transformer", "attention", "nonexistent"], min_match=3)
        assert len(results) == 0

    def test_match_document(self) -> None:
        ks = KeywordSearch()
        doc = Document(
            topic_id="t1",
            source_name="paper.pdf",
            keywords=["transformer", "attention"],
            summary="A paper about transformer architectures.",
        )
        score = ks.match_document(doc, ["transformer"])
        assert score > 0

    def test_match_document_no_match(self) -> None:
        ks = KeywordSearch()
        doc = Document(
            topic_id="t1",
            source_name="paper.pdf",
            keywords=["bert"],
            summary="About BERT.",
        )
        score = ks.match_document(doc, ["transformer"])
        assert score == 0.0

    def test_empty_keywords(self) -> None:
        ks = KeywordSearch()
        assert ks.search_chunks([], []) == []
        assert ks.extract_query_keywords("") == []


class TestFormatter:
    def test_format_with_results(self) -> None:
        response = QueryResponse(
            query="test",
            results=[
                RetrievalResult(
                    doc_id="d1",
                    relevance_score=0.85,
                    snippet="Test content here",
                    metadata={"source_name": "doc.pdf"},
                )
            ],
            total_matches=1,
            topics_searched=["research"],
        )
        text = format_for_agent(response)
        assert "Knowledge Base Results" in text
        assert "doc.pdf" in text
        assert "Test content" in text

    def test_format_empty(self) -> None:
        response = QueryResponse(query="test")
        text = format_for_agent(response)
        assert "No knowledge base results" in text
