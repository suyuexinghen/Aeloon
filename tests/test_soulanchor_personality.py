"""Tests for SoulAnchor personality system — Bones + Soul, PRNG, hash, tier."""

from __future__ import annotations

from aeloon.plugins.SoulAnchor.personality import (
    _TRAIT_NAMES,
    TIER_FLOORS,
    EntitySoul,
    PersonalityTraits,
    StatName,
    Tier,
    build_entity_intro,
    build_personality_prompt,
    fnv1a_hash,
    generate_bones,
    mulberry32,
    tier_from_level,
)

# ---------------------------------------------------------------------------
# PRNG tests
# ---------------------------------------------------------------------------


class TestMulberry32:
    def test_same_seed_same_sequence(self) -> None:
        """Same seed must produce the same sequence of floats."""
        rand1 = mulberry32(12345)
        rand2 = mulberry32(12345)
        for _ in range(20):
            assert rand1() == rand2()

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different first values."""
        r1 = mulberry32(1)()
        r2 = mulberry32(2)()
        assert r1 != r2

    def test_output_in_range(self) -> None:
        """All outputs must be in [0.0, 1.0)."""
        rand = mulberry32(999)
        for _ in range(100):
            v = rand()
            assert 0.0 <= v < 1.0


class TestFnv1aHash:
    def test_deterministic(self) -> None:
        assert fnv1a_hash("hello") == fnv1a_hash("hello")

    def test_different_strings_differ(self) -> None:
        assert fnv1a_hash("hello") != fnv1a_hash("world")

    def test_returns_32bit(self) -> None:
        h = fnv1a_hash("test")
        assert 0 <= h < 2**32


# ---------------------------------------------------------------------------
# PersonalityTraits tests
# ---------------------------------------------------------------------------


class TestPersonalityTraits:
    def test_default_values_are_half(self) -> None:
        traits = PersonalityTraits()
        for name in _TRAIT_NAMES:
            assert getattr(traits, name) == 0.5

    def test_clamp_above_one(self) -> None:
        traits = PersonalityTraits(analytical=1.5)
        assert traits.analytical == 1.0

    def test_clamp_below_zero(self) -> None:
        traits = PersonalityTraits(curiosity=-0.3)
        assert traits.curiosity == 0.0

    def test_to_vector_length(self) -> None:
        traits = PersonalityTraits()
        vec = traits.to_vector()
        assert len(vec) == len(_TRAIT_NAMES)

    def test_behavior_hints_nonempty(self) -> None:
        traits = PersonalityTraits(analytical=0.9, curiosity=0.9, thoroughness=0.1)
        hints = traits.get_behavior_hints()
        assert isinstance(hints, str)
        assert len(hints) > 0

    def test_get_trait_for_category_cognitive(self) -> None:
        traits = PersonalityTraits(analytical=1.0, systematic=1.0, creativity=1.0)
        val = traits.get_trait_for_category("cognitive")
        assert abs(val - 1.0) < 0.01

    def test_get_trait_for_unknown_category(self) -> None:
        traits = PersonalityTraits()
        val = traits.get_trait_for_category("nonexistent")
        assert val == 0.5


# ---------------------------------------------------------------------------
# Tier tests
# ---------------------------------------------------------------------------


class TestTier:
    def test_tier_from_level(self) -> None:
        assert tier_from_level(1) == Tier.COMMON
        assert tier_from_level(6) == Tier.UNCOMMON
        assert tier_from_level(11) == Tier.RARE
        assert tier_from_level(16) == Tier.EPIC
        assert tier_from_level(21) == Tier.LEGENDARY

    def test_tier_floors_defined(self) -> None:
        for tier in Tier:
            assert tier in TIER_FLOORS
            assert TIER_FLOORS[tier] >= 0


# ---------------------------------------------------------------------------
# generate_bones tests
# ---------------------------------------------------------------------------


class TestGenerateBones:
    def test_deterministic(self) -> None:
        """Same inputs always produce identical bones."""
        b1 = generate_bones("test", "dev", "salt")
        b2 = generate_bones("test", "dev", "salt")
        assert b1.traits.analytical == b2.traits.analytical
        assert b1.peak_trait == b2.peak_trait
        assert b1.dump_trait == b2.dump_trait
        assert b1.stats == b2.stats

    def test_different_entities_differ(self) -> None:
        """Different entity_ids produce different bones."""
        b1 = generate_bones("entity_1", "dev", "salt")
        b2 = generate_bones("entity_2", "dev", "salt")
        # Traits should differ (may match by chance but very unlikely)
        assert b1.traits.to_vector() != b2.traits.to_vector()

    def test_peak_dump_different(self) -> None:
        """Peak and dump traits must always be different fields."""
        for i in range(10):
            b = generate_bones(f"ent_{i}", "role", "salt")
            assert b.peak_trait != b.dump_trait

    def test_peak_dump_valid_trait_names(self) -> None:
        b = generate_bones("x", "y", "z")
        assert b.peak_trait in _TRAIT_NAMES
        assert b.dump_trait in _TRAIT_NAMES

    def test_trait_bounds(self) -> None:
        """All generated trait values must be in [0.0, 1.0]."""
        for i in range(20):
            b = generate_bones(f"e{i}", "role", "salt")
            for name in _TRAIT_NAMES:
                v = getattr(b.traits, name)
                assert 0.0 <= v <= 1.0, f"{name}={v} out of bounds"

    def test_stat_names_present(self) -> None:
        b = generate_bones("ent", "role", "salt")
        for stat in StatName:
            assert stat.value in b.stats

    def test_tier_floor_respected(self) -> None:
        """Stats for a given tier must be >= tier floor."""
        for level, tier in [(1, Tier.COMMON), (6, Tier.UNCOMMON), (11, Tier.RARE)]:
            b = generate_bones("ent", "role", "salt", level=level)
            floor = TIER_FLOORS[tier]
            for val in b.stats.values():
                assert val >= floor, f"Stat {val} below tier floor {floor} for {tier}"

    def test_tier_advances_with_level(self) -> None:
        b1 = generate_bones("ent", "role", "salt", level=1)
        b2 = generate_bones("ent", "role", "salt", level=21)
        assert b1.tier == Tier.COMMON
        assert b2.tier == Tier.LEGENDARY

    def test_bones_not_serializable(self) -> None:
        """EntityBones is not serialized by Entity (exclude=True on the field)."""
        from aeloon.plugins.SoulAnchor.entity import Entity

        bones = generate_bones("e", "r", "s")
        entity = Entity(entity_id="e", name="E", role="r", bones=bones)
        dumped = entity.model_dump()
        assert "bones" not in dumped


# ---------------------------------------------------------------------------
# EntitySoul tests
# ---------------------------------------------------------------------------


class TestEntitySoul:
    def test_defaults(self) -> None:
        soul = EntitySoul()
        assert soul.name == ""
        assert soul.voice == ""
        assert soul.intro_template == ""

    def test_custom_values(self) -> None:
        soul = EntitySoul(name="Aria", voice="Precise. Methodical.", intro_template="a dev")
        assert soul.name == "Aria"


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def _make_entity(self) -> object:
        from aeloon.plugins.SoulAnchor.entity import Entity

        entity = Entity(entity_id="test", name="Aria", role="developer")
        entity.soul = EntitySoul(
            name="Aria",
            voice="Writes clean code.",
            intro_template="a developer",
        )
        entity.bones = generate_bones("test", "developer", "salt")
        return entity

    def test_build_entity_intro(self) -> None:
        entity = self._make_entity()
        intro = build_entity_intro(entity)
        assert "Aria" in intro
        assert "developer" in intro

    def test_build_personality_prompt_contains_voice(self) -> None:
        entity = self._make_entity()
        prompt = build_personality_prompt(entity)
        assert "Writes clean code." in prompt

    def test_build_personality_prompt_contains_tier(self) -> None:
        entity = self._make_entity()
        prompt = build_personality_prompt(entity)
        assert "Tier" in prompt

    def test_build_personality_prompt_no_bones(self) -> None:
        from aeloon.plugins.SoulAnchor.entity import Entity

        entity = Entity(entity_id="nb", name="NoBones", role="r")
        result = build_personality_prompt(entity)
        assert "NoBones" in result
