"""Tests for SoulAnchor Entity model and EntityManager.

Critical tests:
  - bones field excluded from serialization (anti-tampering)
  - bones regenerated identically on reload
  - entity round-trip (create → save → load → verify)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aeloon.plugins.SoulAnchor.entity import Entity, EntityStatus
from aeloon.plugins.SoulAnchor.personality import EntitySoul, generate_bones

# ---------------------------------------------------------------------------
# Entity model tests
# ---------------------------------------------------------------------------


class TestEntity:
    def test_create_minimal(self) -> None:
        e = Entity(entity_id="e1", name="Alice", role="dev")
        assert e.entity_id == "e1"
        assert e.status == EntityStatus.ACTIVE
        assert e.level == 1
        assert e.bones is None

    def test_bones_excluded_from_serialization(self) -> None:
        """Bones must never appear in model_dump output."""
        e = Entity(entity_id="e1", name="Alice", role="dev")
        e.bones = generate_bones("e1", "dev", "test_salt")
        dumped = e.model_dump()
        assert "bones" not in dumped

    def test_bones_excluded_from_json(self) -> None:
        """Bones must never appear in model_dump(mode='json') output."""
        e = Entity(entity_id="e1", name="Alice", role="dev")
        e.bones = generate_bones("e1", "dev", "test_salt")
        dumped = e.model_dump(mode="json")
        assert "bones" not in dumped

    def test_soul_persisted_in_serialization(self) -> None:
        e = Entity(entity_id="e1", name="Alice", role="dev")
        e.soul = EntitySoul(name="Alice", voice="Precise.", intro_template="a dev")
        dumped = e.model_dump(mode="json")
        assert "soul" in dumped
        assert dumped["soul"]["voice"] == "Precise."

    def test_round_trip(self) -> None:
        """Entity must survive model_dump → model_validate round-trip."""
        e = Entity(entity_id="rt", name="RT", role="tester")
        e.soul = EntitySoul(name="RT", voice="Tests edge cases.")
        e.level = 5
        e.experience_points = 150.0

        data = e.model_dump(mode="json")
        restored = Entity.model_validate(data)
        assert restored.entity_id == e.entity_id
        assert restored.soul.voice == e.soul.voice
        assert restored.level == e.level
        assert restored.bones is None  # bones not in serialized data


class TestEntityStatus:
    def test_valid_statuses(self) -> None:
        for s in ("active", "learning", "resting", "archived"):
            e = Entity(entity_id="x", name="X", role="r", status=EntityStatus(s))
            assert e.status.value == s


# ---------------------------------------------------------------------------
# EntityManager tests
# ---------------------------------------------------------------------------


class TestEntityManager:
    def test_create_entity(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        mgr = EntityManager(tmp_path, {"bones_salt": "test"})
        entity = mgr.create_entity("alice", "Alice", "dev", voice="Precise coder.")
        assert entity.entity_id == "alice"
        assert entity.bones is not None
        assert entity.soul.voice == "Precise coder."

    def test_create_duplicate_raises(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        mgr = EntityManager(tmp_path, {"bones_salt": "test"})
        mgr.create_entity("dup", "Dup", "dev")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_entity("dup", "Dup2", "dev")

    def test_bones_not_in_storage(self, tmp_path: Path) -> None:
        """Bones must not appear in the JSONL file after save."""
        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        mgr = EntityManager(tmp_path, {"bones_salt": "test"})
        mgr.create_entity("ent_x", "X", "dev")

        jsonl_file = tmp_path / "entities" / "ent_x.jsonl"
        assert jsonl_file.exists()
        raw = jsonl_file.read_text()
        assert "bones" not in raw

    def test_bones_regenerated_on_reload(self, tmp_path: Path) -> None:
        """Bones must be regenerated identically after save → get cycle."""
        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        salt = "test_salt_42"
        mgr1 = EntityManager(tmp_path, {"bones_salt": salt})
        entity = mgr1.create_entity("regen", "R", "dev")
        original_analytical = entity.bones.traits.analytical
        original_peak = entity.bones.peak_trait

        # New manager, load from storage
        mgr2 = EntityManager(tmp_path, {"bones_salt": salt})
        reloaded = mgr2.get("regen")
        assert reloaded is not None
        assert reloaded.bones is not None
        assert reloaded.bones.traits.analytical == original_analytical
        assert reloaded.bones.peak_trait == original_peak

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        mgr = EntityManager(tmp_path, {})
        assert mgr.get("nonexistent") is None

    def test_list_entities(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        mgr = EntityManager(tmp_path, {"bones_salt": "s"})
        mgr.create_entity("a", "A", "dev")
        mgr.create_entity("b", "B", "tester")
        entities = mgr.list_entities()
        ids = {e.entity_id for e in entities}
        assert "a" in ids
        assert "b" in ids

    def test_archive_entity(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        mgr = EntityManager(tmp_path, {"bones_salt": "s"})
        mgr.create_entity("arch", "Arch", "dev")
        mgr.archive_entity("arch")

        entity = mgr.get("arch")
        assert entity is not None
        assert entity.status == EntityStatus.ARCHIVED

    def test_apply_xp_level_up(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        mgr = EntityManager(tmp_path, {"bones_salt": "s"})
        mgr.create_entity("leveler", "L", "dev")
        # Level 1 needs 100 XP to level up
        entity, leveled = mgr.apply_xp("leveler", 150.0)
        assert leveled is True
        assert entity.level == 2

    def test_bones_anti_tampering(self, tmp_path: Path) -> None:
        """Even if someone edits the JSONL to add bones fields, they are ignored."""
        import json

        from aeloon.plugins.SoulAnchor.services.entity_manager import EntityManager

        salt = "anti_tamper_salt"
        mgr = EntityManager(tmp_path, {"bones_salt": salt})
        entity = mgr.create_entity("tamper", "T", "dev")
        original_analytical = entity.bones.traits.analytical

        # Tamper: append a record with fake bones
        jsonl_file = tmp_path / "entities" / "tamper.jsonl"
        with jsonl_file.open("a") as f:
            tampered = entity.model_dump(mode="json")
            tampered["_type"] = "entity"
            tampered["bones"] = {"traits": {"analytical": 0.999}}  # fake high stat
            f.write(json.dumps(tampered) + "\n")

        # Reload — bones must be regenerated from hash, not from tampered data
        mgr2 = EntityManager(tmp_path, {"bones_salt": salt})
        reloaded = mgr2.get("tamper")
        assert reloaded is not None
        # Bones regenerated from hash — same as original
        assert abs(reloaded.bones.traits.analytical - original_analytical) < 0.001
