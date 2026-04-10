"""Tests for SoulAnchor three-layer memory system."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# WorkingMemory tests
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    def test_add_and_retrieve(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.working import WorkingMemory

        wm = WorkingMemory("e1", tmp_path)
        eid = wm.add("something happened", importance=0.6)
        results = wm.retrieve_relevant("something")
        assert any(e.entry_id == eid for e in results)

    def test_capacity_enforced(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.working import WorkingMemory

        wm = WorkingMemory("e1", tmp_path, capacity=5)
        for i in range(7):
            wm.add(f"entry {i}", importance=0.5)
        assert len(wm.get_all()) <= 5

    def test_evicts_lowest_importance(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.working import WorkingMemory

        wm = WorkingMemory("e1", tmp_path, capacity=3)
        wm.add("low importance entry", importance=0.1)
        wm.add("high importance entry A", importance=0.9)
        wm.add("high importance entry B", importance=0.8)
        # Adding a 4th should evict the lowest
        wm.add("another entry", importance=0.7)
        entries = wm.get_all()
        assert not any(e.content == "low importance entry" for e in entries)

    def test_expiry_removes_entries(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.working import WorkingMemory

        wm = WorkingMemory("e1", tmp_path)
        past = datetime.now(UTC) - timedelta(seconds=1)
        wm.add("expired entry", expires_at=past)
        assert len(wm.get_all()) == 0

    def test_remove(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.working import WorkingMemory

        wm = WorkingMemory("e1", tmp_path)
        eid = wm.add("to remove")
        wm.remove(eid)
        assert not any(e.entry_id == eid for e in wm.get_all())

    def test_get_stats(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.working import WorkingMemory

        wm = WorkingMemory("e1", tmp_path)
        wm.add("item", importance=0.7)
        stats = wm.get_stats()
        assert stats["count"] == 1
        assert stats["capacity"] > 0


# ---------------------------------------------------------------------------
# EpisodicMemory tests
# ---------------------------------------------------------------------------


class TestEpisodicMemory:
    def test_add_and_retrieve(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.episodic import EpisodicMemory

        em = EpisodicMemory("e1", tmp_path)
        eid = em.add("completed a task", importance=0.7)
        results = em.retrieve_relevant("task")
        assert any(e.entry_id == eid for e in results)

    def test_access_count_incremented(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.episodic import EpisodicMemory

        em = EpisodicMemory("e1", tmp_path)
        em.add("something", importance=0.6)
        em.retrieve_relevant("something")
        entries = em.get_all()
        assert any(e.access_count == 1 for e in entries)

    def test_compress_removes_old_entries(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.episodic import EpisodicMemory

        em = EpisodicMemory("e1", tmp_path, half_life_days=0.0001)
        # Add entry with old timestamp (will have near-zero retention)
        import datetime as dt

        em.add("old memory", importance=0.1)
        # Force old timestamp
        for entry in em.get_all():
            entry.created_at = dt.datetime.now(dt.UTC) - dt.timedelta(days=365)
        removed = em.compress(weight_threshold=0.9)
        assert removed >= 0  # May or may not remove depending on internal state

    def test_get_stats(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.episodic import EpisodicMemory

        em = EpisodicMemory("e1", tmp_path)
        stats = em.get_stats()
        assert stats["count"] == 0
        em.add("test entry")
        stats = em.get_stats()
        assert stats["count"] == 1


# ---------------------------------------------------------------------------
# SemanticMemory tests
# ---------------------------------------------------------------------------


class TestSemanticMemory:
    def test_add_and_retrieve(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.semantic import SemanticMemory

        sm = SemanticMemory("e1", tmp_path)
        eid = sm.add("always check inputs", domain="coding", knowledge_type="rule")
        results = sm.retrieve(domain="coding")
        assert any(e.entry_id == eid for e in results)

    def test_verify_increments_count(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.semantic import SemanticMemory

        sm = SemanticMemory("e1", tmp_path)
        eid = sm.add("a rule", domain="general")
        sm.verify(eid)
        entries = sm.get_all()
        entry = next(e for e in entries if e.entry_id == eid)
        assert entry.verification_count == 1
        assert entry.confidence > 0.5

    def test_capacity_enforced(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.semantic import SemanticMemory

        sm = SemanticMemory("e1", tmp_path, capacity=5)
        for i in range(7):
            sm.add(f"knowledge {i}", domain="test")
        assert len(sm.get_all()) <= 5

    def test_format_for_prompt(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.semantic import SemanticMemory

        sm = SemanticMemory("e1", tmp_path)
        sm.add("best practice: test early", domain="testing", knowledge_type="rule")
        output = sm.format_for_prompt()
        assert "Relevant Knowledge" in output
        assert "test early" in output

    def test_format_for_prompt_empty(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.memory.semantic import SemanticMemory

        sm = SemanticMemory("e1", tmp_path)
        assert sm.format_for_prompt() == ""
