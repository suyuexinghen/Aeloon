"""Tests for KB plugin models."""

from aeloon.plugins.KnowledgeBase.models import (
    AddResult,
    Document,
    DocumentChunk,
    DocumentFormat,
    DocumentStatus,
    ProcessedDocument,
    QueryResponse,
    Topic,
    compute_content_hash,
)


class TestDocumentStatus:
    def test_three_states(self) -> None:
        assert set(DocumentStatus) == {
            DocumentStatus.ACTIVE,
            DocumentStatus.OFF,
            DocumentStatus.TRASH,
        }

    def test_values(self) -> None:
        assert DocumentStatus.ACTIVE.value == "active"
        assert DocumentStatus.OFF.value == "off"
        assert DocumentStatus.TRASH.value == "trash"


class TestDocumentFormat:
    def test_five_formats(self) -> None:
        assert set(DocumentFormat) == {
            DocumentFormat.PDF,
            DocumentFormat.DOCX,
            DocumentFormat.MD,
            DocumentFormat.TXT,
            DocumentFormat.CSV,
        }


class TestTopic:
    def test_default_fields(self) -> None:
        t = Topic(name="test")
        assert t.name == "test"
        assert t.topic_id
        assert t.document_count == 0
        assert not t.is_special

    def test_serialization_roundtrip(self) -> None:
        t = Topic(name="research", description="Papers", is_special=False)
        data = t.model_dump(mode="json")
        t2 = Topic.model_validate(data)
        assert t2.name == "research"
        assert t2.topic_id == t.topic_id


class TestDocument:
    def test_default_status(self) -> None:
        d = Document(topic_id="t1", source_name="test.pdf")
        assert d.status == DocumentStatus.ACTIVE

    def test_roundtrip(self) -> None:
        d = Document(
            topic_id="t1",
            source_name="paper.pdf",
            source_format=DocumentFormat.PDF,
            keywords=["attention", "transformer"],
        )
        data = d.model_dump(mode="json")
        d2 = Document.model_validate(data)
        assert d2.source_name == "paper.pdf"
        assert d2.keywords == ["attention", "transformer"]


class TestDocumentChunk:
    def test_defaults(self) -> None:
        c = DocumentChunk(content="hello world")
        assert c.chunk_id
        assert c.index == 0
        assert c.start_char == 0


class TestProcessedDocument:
    def test_from_fields(self) -> None:
        pd = ProcessedDocument(markdown="# Title\n\nBody text.", keywords=["title"])
        assert pd.keywords == ["title"]
        assert len(pd.chunks) == 0


class TestQueryResponse:
    def test_empty(self) -> None:
        qr = QueryResponse(query="test")
        assert qr.total_matches == 0
        assert qr.results == []


class TestAddResult:
    def test_success(self) -> None:
        r = AddResult(doc_id="abc", source_name="f.pdf", success=True)
        assert r.success
        assert not r.error

    def test_failure(self) -> None:
        r = AddResult(source_name="f.xyz", error="unsupported")
        assert not r.success


class TestContentHash:
    def test_deterministic(self) -> None:
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("hello")
        assert h1 == h2

    def test_different_content(self) -> None:
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("world")
        assert h1 != h2
