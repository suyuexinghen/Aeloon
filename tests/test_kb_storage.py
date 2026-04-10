"""Tests for KB storage backend."""

from pathlib import Path

import pytest
from aeloon.plugins.KnowledgeBase.models import Document, DocumentChunk, DocumentStatus, Topic
from aeloon.plugins.KnowledgeBase.storage.jsonl_store import KBStorage


@pytest.fixture
def storage(tmp_path: Path) -> KBStorage:
    return KBStorage(tmp_path / "kb")


async def test_save_and_load_topics(storage: KBStorage) -> None:
    t1 = Topic(name="research")
    t2 = Topic(name="notes")
    await storage.save_topic(t1)
    await storage.save_topic(t2)

    topics = await storage.load_topics()
    assert len(topics) == 2
    names = {t.name for t in topics}
    assert names == {"research", "notes"}


async def test_topic_last_write_wins(storage: KBStorage) -> None:
    t = Topic(name="research", description="v1")
    await storage.save_topic(t)

    # Update by saving again with same topic_id
    t2 = Topic(topic_id=t.topic_id, name="research", description="v2")
    await storage.save_topic(t2)

    topics = await storage.load_topics()
    assert len(topics) == 1
    assert topics[0].description == "v2"


async def test_remove_topic(storage: KBStorage) -> None:
    t = Topic(name="old")
    await storage.save_topic(t)
    await storage.remove_topic(t.topic_id)

    # Removed topics should not appear
    topics = await storage.load_topics()
    assert len(topics) == 0


async def test_save_and_load_documents(storage: KBStorage) -> None:
    t = Topic(name="test")
    await storage.save_topic(t)

    d = Document(topic_id=t.topic_id, source_name="doc1.pdf")
    await storage.save_document(d)

    docs = await storage.load_documents(t.topic_id)
    assert len(docs) == 1
    assert docs[0].source_name == "doc1.pdf"


async def test_document_last_write_wins(storage: KBStorage) -> None:
    t = Topic(name="test")
    await storage.save_topic(t)

    d = Document(doc_id="d1", topic_id=t.topic_id, source_name="v1")
    await storage.save_document(d)

    d2 = Document(doc_id="d1", topic_id=t.topic_id, source_name="v2", status=DocumentStatus.OFF)
    await storage.save_document(d2)

    docs = await storage.load_documents(t.topic_id)
    assert len(docs) == 1
    assert docs[0].status == DocumentStatus.OFF


async def test_remove_document(storage: KBStorage) -> None:
    t = Topic(name="test")
    await storage.save_topic(t)

    d = Document(doc_id="d1", topic_id=t.topic_id, source_name="doc.pdf")
    await storage.save_document(d)
    await storage.remove_document(t.topic_id, "d1")

    docs = await storage.load_documents(t.topic_id)
    assert len(docs) == 0


async def test_save_and_load_content(storage: KBStorage) -> None:
    t = Topic(name="test")
    await storage.save_topic(t)

    await storage.save_content(t.topic_id, "d1", "# Hello\n\nWorld")
    content = await storage.load_content(t.topic_id, "d1")
    assert content == "# Hello\n\nWorld"


async def test_load_content_missing(storage: KBStorage) -> None:
    result = await storage.load_content("nonexistent", "d1")
    assert result is None


async def test_save_and_load_chunks(storage: KBStorage) -> None:
    t = Topic(name="test")
    await storage.save_topic(t)

    chunks = [
        DocumentChunk(chunk_id="c1", content="first chunk", index=0),
        DocumentChunk(chunk_id="c2", content="second chunk", index=1),
    ]
    await storage.save_chunks(t.topic_id, "d1", chunks)

    loaded = await storage.load_chunks(t.topic_id, "d1")
    assert len(loaded) == 2
    assert loaded[0].content == "first chunk"


async def test_chunks_overwrite(storage: KBStorage) -> None:
    t = Topic(name="test")
    await storage.save_topic(t)

    # First write
    await storage.save_chunks(t.topic_id, "d1", [DocumentChunk(content="v1")])
    # Overwrite
    await storage.save_chunks(
        t.topic_id, "d1", [DocumentChunk(content="v2"), DocumentChunk(content="v3")]
    )

    loaded = await storage.load_chunks(t.topic_id, "d1")
    assert len(loaded) == 2
    assert loaded[0].content == "v2"


async def test_base_path(storage: KBStorage, tmp_path: Path) -> None:
    assert storage.base_path == tmp_path / "kb"
