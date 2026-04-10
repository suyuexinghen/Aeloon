"""Tests for KB topic service."""

from pathlib import Path

import pytest
from aeloon.plugins.KnowledgeBase.config import KBConfig
from aeloon.plugins.KnowledgeBase.services.topic_service import TopicService
from aeloon.plugins.KnowledgeBase.storage.jsonl_store import KBStorage


@pytest.fixture
def service(tmp_path: Path) -> TopicService:
    storage = KBStorage(tmp_path / "kb")
    config = KBConfig()
    return TopicService(storage, config)


async def test_create_topic(service: TopicService) -> None:
    topic = await service.create("research", "Papers")
    assert topic.name == "research"
    assert topic.description == "Papers"
    assert not topic.is_special


async def test_create_topic_invalid_name(service: TopicService) -> None:
    with pytest.raises(ValueError, match="Invalid topic name"):
        await service.create("has spaces")


async def test_create_topic_duplicate(service: TopicService) -> None:
    await service.create("research")
    with pytest.raises(ValueError, match="already exists"):
        await service.create("research")


async def test_list_topics(service: TopicService) -> None:
    await service.create("a")
    await service.create("b")
    topics = await service.list()
    assert len(topics) == 2


async def test_get_topic(service: TopicService) -> None:
    created = await service.create("research")
    found = await service.get("research")
    assert found is not None
    assert found.topic_id == created.topic_id


async def test_get_topic_not_found(service: TopicService) -> None:
    assert await service.get("nonexistent") is None


async def test_remove_topic(service: TopicService) -> None:
    await service.create("old")
    ok, msg = await service.remove("old")
    assert ok
    assert await service.get("old") is None


async def test_remove_special_topic(service: TopicService) -> None:
    await service.ensure_special_topics()
    ok, msg = await service.remove("_uncategorized")
    assert not ok
    assert "special" in msg.lower()


async def test_remove_nonexistent(service: TopicService) -> None:
    ok, msg = await service.remove("nope")
    assert not ok


async def test_ensure_special_topics(service: TopicService) -> None:
    await service.ensure_special_topics()
    unc = await service.get("_uncategorized")
    trash = await service.get("_trash")
    assert unc is not None
    assert trash is not None
    assert unc.is_special
    assert trash.is_special


async def test_ensure_special_topics_idempotent(service: TopicService) -> None:
    await service.ensure_special_topics()
    await service.ensure_special_topics()
    topics = await service.list()
    special = [t for t in topics if t.is_special]
    assert len(special) == 2
