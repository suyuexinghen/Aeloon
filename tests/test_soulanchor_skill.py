"""Tests for SoulAnchor skill system — registry, learning, growth."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# SkillRegistry tests
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def test_register_and_get(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.skill.registry import SkillLevel, SkillRegistry

        reg = SkillRegistry("e1", tmp_path)
        reg.register("python", "Python", "coding")
        assert reg.get_proficiency("python") == 0.0
        assert reg.get_level("python") == SkillLevel.L1

    def test_register_idempotent(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.skill.registry import SkillRegistry

        reg = SkillRegistry("e1", tmp_path)
        reg.register("python", "Python", "coding")
        reg.register("python", "Python 2", "coding")  # Should be no-op
        skills = reg.list_all()
        python_skills = [s for s in skills if s.skill_id == "python"]
        assert len(python_skills) == 1
        assert python_skills[0].name == "Python"  # First registration wins

    def test_add_proficiency(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.skill.registry import SkillLevel, SkillRegistry

        reg = SkillRegistry("e1", tmp_path)
        reg.register("skill_x", "Skill X", "general")
        new_val = reg.add_proficiency("skill_x", 0.35)
        assert new_val == 0.35
        assert reg.get_level("skill_x") == SkillLevel.L2

    def test_proficiency_clamped(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.skill.registry import SkillRegistry

        reg = SkillRegistry("e1", tmp_path)
        reg.register("s", "S", "g")
        reg.add_proficiency("s", 999.0)
        assert reg.get_proficiency("s") == 1.0

    def test_list_by_category(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.skill.registry import SkillRegistry

        reg = SkillRegistry("e1", tmp_path)
        reg.register("py", "Python", "coding")
        reg.register("trust", "Trust", "social")
        coding_skills = reg.list_by_category("coding")
        assert len(coding_skills) == 1
        assert coding_skills[0].skill_id == "py"

    def test_unknown_skill_returns_zero(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.skill.registry import SkillRegistry

        reg = SkillRegistry("e1", tmp_path)
        assert reg.get_proficiency("nonexistent") == 0.0

    def test_level_boundaries(self, tmp_path: Path) -> None:
        from aeloon.plugins.SoulAnchor.skill.registry import SkillLevel, SkillRegistry

        reg = SkillRegistry("e1", tmp_path)
        for level, proficiency in [
            (SkillLevel.L1, 0.1),
            (SkillLevel.L2, 0.4),
            (SkillLevel.L3, 0.7),
            (SkillLevel.L4, 0.9),
        ]:
            reg.register(f"s_{level.value}", "S", "g", initial_proficiency=proficiency)
            assert reg.get_level(f"s_{level.value}") == level


# ---------------------------------------------------------------------------
# Practice learning tests
# ---------------------------------------------------------------------------


class TestLearningFunctions:
    def test_learn_by_practice_success_less_than_failure(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.learning import learn_by_practice

        success_growth = learn_by_practice(0.0, True, 0.5)
        failure_growth = learn_by_practice(0.0, False, 0.5)
        assert failure_growth > success_growth

    def test_diminishing_returns(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.learning import learn_by_practice

        growth_low = learn_by_practice(0.1, True, 0.5)
        growth_high = learn_by_practice(0.9, True, 0.5)
        assert growth_low > growth_high

    def test_curiosity_boosts_growth(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.learning import learn_by_practice

        growth_curious = learn_by_practice(0.3, True, 0.5, curiosity=1.0)
        growth_bored = learn_by_practice(0.3, True, 0.5, curiosity=0.0)
        assert growth_curious > growth_bored

    def test_learn_by_observation_reproducible(self) -> None:
        """Same inputs → same observation outcome."""
        from aeloon.plugins.SoulAnchor.skill.learning import learn_by_observation

        r1 = learn_by_observation(0.3, 0.8, "entity_abc")
        r2 = learn_by_observation(0.3, 0.8, "entity_abc")
        assert r1 == r2

    def test_learn_by_observation_no_upskill_from_weaker(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.learning import learn_by_observation

        learned, amount = learn_by_observation(0.8, 0.3, "entity_x")
        assert not learned
        assert amount == 0.0

    def test_learn_by_collaboration_both_learn(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.learning import learn_by_collaboration

        growth_a, growth_b = learn_by_collaboration(0.3, 0.3, success=True)
        assert growth_a > 0
        assert growth_b > 0


# ---------------------------------------------------------------------------
# GrowthTracker tests
# ---------------------------------------------------------------------------


class TestGrowthTracker:
    def test_xp_calculation(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.growth import GrowthTracker

        gt = GrowthTracker()
        xp = gt.calculate_xp(complexity=1.0, success=True)
        assert xp > 0

    def test_level_up_threshold(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.growth import GrowthTracker

        gt = GrowthTracker(xp_multiplier=100)
        # Level 1 needs 100 XP
        new_level = gt.check_level_up(1, 0.0, 100.0)
        assert new_level == 2

    def test_no_level_up_below_threshold(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.growth import GrowthTracker

        gt = GrowthTracker(xp_multiplier=100)
        new_level = gt.check_level_up(1, 0.0, 50.0)
        assert new_level == 1

    def test_multiple_level_ups(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.growth import GrowthTracker

        gt = GrowthTracker(xp_multiplier=100)
        # Level 1 (100) + Level 2 (200) + some = 3 levels in one go
        new_level = gt.check_level_up(1, 0.0, 350.0)
        assert new_level >= 3

    def test_tier_change_detected(self) -> None:
        from aeloon.plugins.SoulAnchor.personality import Tier
        from aeloon.plugins.SoulAnchor.skill.growth import GrowthTracker

        gt = GrowthTracker()
        old_tier, new_tier = gt.get_tier_change(5, 6)
        assert old_tier == Tier.COMMON
        assert new_tier == Tier.UNCOMMON

    def test_no_tier_change_same_tier(self) -> None:
        from aeloon.plugins.SoulAnchor.skill.growth import GrowthTracker

        gt = GrowthTracker()
        old_tier, new_tier = gt.get_tier_change(1, 2)
        assert old_tier is None  # No change
