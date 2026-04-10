"""Tests for Pet Manor model layer: Pet, Personality, Manor, Grid, Building, Blueprint.

Covers: Pet.create(), Personality generation, mood computation, personality evolution,
Manor filtering, Grid operations, Building placement, Blueprint loading, footprint rotation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aeloon.plugins.PetManor.models.building import (
    Blueprint,
    Building,
    get_absolute_cells,
    get_rotated_footprint,
    load_blueprints,
)
from aeloon.plugins.PetManor.models.manor import Grid, Manor
from aeloon.plugins.PetManor.models.pet import (
    VALID_SPECIES,
    Personality,
    Pet,
    compute_mood,
    describe_personality,
    evolve_personality,
    generate_personality,
    mood_text,
    practice_skill,
    skill_level_name,
)

# ---------------------------------------------------------------------------
# Test Personality
# ---------------------------------------------------------------------------


class TestPersonality:
    """Test Personality dataclass and generation."""

    def test_personality_to_dict_roundtrip(self) -> None:
        """Personality.to_dict() / from_dict() round-trips correctly."""
        p = Personality(sociability=0.7, energy=0.5, curiosity=0.8, boldness=0.3)
        d = p.to_dict()
        assert d == {
            "sociability": 0.7,
            "energy": 0.5,
            "curiosity": 0.8,
            "boldness": 0.3,
        }
        p2 = Personality.from_dict(d)
        assert p2.sociability == 0.7
        assert p2.energy == 0.5
        assert p2.curiosity == 0.8
        assert p2.boldness == 0.3

    def test_from_dict_defaults(self) -> None:
        """Personality.from_dict() uses defaults for missing keys."""
        p = Personality.from_dict({})
        assert p.sociability == 0.5
        assert p.energy == 0.5
        assert p.curiosity == 0.5
        assert p.boldness == 0.5

    def test_generate_personality_valid_species(self) -> None:
        """generate_personality() works for all valid species."""
        for species in VALID_SPECIES:
            p = generate_personality(species)
            assert 0.05 <= p.sociability <= 0.95
            assert 0.05 <= p.energy <= 0.95
            assert 0.05 <= p.curiosity <= 0.95
            assert 0.05 <= p.boldness <= 0.95

    def test_generate_personality_clamps_range(self) -> None:
        """generate_personality() clamps values to [0.05, 0.95]."""
        # Generate many personalities and verify none exceed bounds
        for _ in range(100):
            p = generate_personality("mammal")
            assert 0.05 <= p.sociability <= 0.95
            assert 0.05 <= p.energy <= 0.95
            assert 0.05 <= p.curiosity <= 0.95
            assert 0.05 <= p.boldness <= 0.95

    def test_describe_personality_non_empty(self) -> None:
        """describe_personality() returns non-empty string."""
        p = Personality(sociability=0.8, energy=0.7, curiosity=0.6, boldness=0.4)
        desc = describe_personality(p)
        assert desc
        assert "This pet" in desc

    def test_describe_personality_all_high(self) -> None:
        """describe_personality() captures high traits."""
        p = Personality(sociability=0.9, energy=0.9, curiosity=0.9, boldness=0.9)
        desc = describe_personality(p)
        assert "loves attention" in desc
        assert "always bouncing around" in desc
        assert "explores everything" in desc
        assert "fearless" in desc

    def test_describe_personality_all_low(self) -> None:
        """describe_personality() captures low traits."""
        p = Personality(sociability=0.1, energy=0.1, curiosity=0.1, boldness=0.1)
        desc = describe_personality(p)
        assert "prefers solitude" in desc
        assert "calm and relaxed" in desc
        assert "a homebody" in desc
        assert "a bit shy" in desc

    def test_describe_personality_balanced(self) -> None:
        """describe_personality() shows balanced for middle values."""
        p = Personality(sociability=0.5, energy=0.5, curiosity=0.5, boldness=0.5)
        desc = describe_personality(p)
        assert "well-balanced" in desc

    def test_evolve_personality_nudges_up(self) -> None:
        """evolve_personality() nudges traits up for positive events."""
        p = Personality(sociability=0.5, energy=0.5, curiosity=0.5, boldness=0.5)
        p2 = evolve_personality(p, "player_petted")
        assert p2.sociability > p.sociability
        assert 0.05 <= p2.sociability <= 0.95

    def test_evolve_personality_nudges_down(self) -> None:
        """evolve_personality() nudges traits down for negative events."""
        p = Personality(sociability=0.5, energy=0.5, curiosity=0.5, boldness=0.5)
        p2 = evolve_personality(p, "scared")
        assert p2.boldness < p.boldness
        assert 0.05 <= p2.boldness <= 0.95

    def test_evolve_personality_unknown_event_no_change(self) -> None:
        """evolve_personality() returns same personality for unknown events."""
        p = Personality(sociability=0.5, energy=0.5, curiosity=0.5, boldness=0.5)
        p2 = evolve_personality(p, "unknown_event")
        assert p2.sociability == p.sociability
        assert p2.energy == p.energy
        assert p2.curiosity == p.curiosity
        assert p2.boldness == p.boldness

    def test_evolve_personality_clamps_bounds(self) -> None:
        """evolve_personality() clamps to [0.05, 0.95]."""
        p = Personality(sociability=0.94, energy=0.5, curiosity=0.5, boldness=0.5)
        p2 = evolve_personality(p, "player_petted")
        assert p2.sociability <= 0.95

        p3 = Personality(sociability=0.06, energy=0.5, curiosity=0.5, boldness=0.5)
        p4 = evolve_personality(p3, "scared")
        assert p4.sociability >= 0.05


# ---------------------------------------------------------------------------
# Test Pet
# ---------------------------------------------------------------------------


class TestPet:
    """Test Pet dataclass and creation."""

    def test_pet_create_valid(self) -> None:
        """Pet.create() creates a valid pet."""
        pet = Pet.create("Fluffy", "mammal")
        assert pet.name == "Fluffy"
        assert pet.species == "mammal"
        assert pet.status == "active"
        assert pet.pet_id
        assert 0.05 <= pet.personality.sociability <= 0.95
        assert 0.05 <= pet.personality.energy <= 0.95
        assert 0.05 <= pet.personality.curiosity <= 0.95
        assert 0.05 <= pet.personality.boldness <= 0.95
        assert 2 <= pet.pos_x <= 13
        assert 2 <= pet.pos_y <= 13
        assert pet.hunger == 1.0
        assert pet.energy == 1.0
        assert pet.happiness == 1.0
        assert pet.exercise == 1.0
        assert pet.created_at
        assert pet.updated_at

    def test_pet_to_dict_roundtrip(self) -> None:
        """Pet.to_dict() / from_dict() round-trips correctly (with rounding)."""
        pet = Pet.create("Whiskers", "bird")
        d = pet.to_dict()
        assert d["_type"] == "pet_state"
        assert d["name"] == "Whiskers"
        assert d["species"] == "bird"

        pet2 = Pet.from_dict(d)
        assert pet2.name == pet.name
        assert pet2.species == pet.species
        assert pet2.pet_id == pet.pet_id
        assert pet2.status == pet.status
        assert pet2.hunger == pet.hunger
        assert pet2.energy == pet.energy
        assert pet2.happiness == pet.happiness
        assert pet2.exercise == pet.exercise
        # Personality values are rounded to 4 decimals in to_dict()
        assert pet2.personality.sociability == round(pet.personality.sociability, 4)
        assert pet2.personality.energy == round(pet.personality.energy, 4)
        assert pet2.personality.curiosity == round(pet.personality.curiosity, 4)
        assert pet2.personality.boldness == round(pet.personality.boldness, 4)

    def test_pet_from_dict_defaults(self) -> None:
        """Pet.from_dict() uses defaults for missing keys."""
        d = {
            "pet_id": "abc123",
            "name": "Test",
            "species": "reptile",
        }
        pet = Pet.from_dict(d)
        assert pet.status == "active"
        assert pet.hunger == 1.0
        assert pet.energy == 1.0
        assert pet.happiness == 1.0
        assert pet.exercise == 1.0
        assert pet.personality.sociability == 0.5
        assert pet.personality.energy == 0.5
        assert pet.personality.curiosity == 0.5
        assert pet.personality.boldness == 0.5
        assert pet.pos_x == 0
        assert pet.pos_y == 0
        assert pet.skills == {}

    def test_pet_stats_dict(self) -> None:
        """Pet.stats_dict() returns correct stats."""
        pet = Pet.create("Scales", "reptile")
        stats = pet.stats_dict()
        assert stats == {
            "hunger": 1.0,
            "energy": 1.0,
            "happiness": 1.0,
            "exercise": 1.0,
        }

    def test_compute_mood_range(self) -> None:
        """compute_mood() returns values in [-1, +1]."""
        pet = Pet.create("Test", "mammal")
        for _ in range(100):
            # Vary stats
            pet.hunger = 0.0
            mood = compute_mood(pet)
            assert -1.0 <= mood <= 1.0

            pet.hunger = 1.0
            mood = compute_mood(pet)
            assert -1.0 <= mood <= 1.0

    def test_compute_mood_all_full(self) -> None:
        """compute_mood() returns +1 when all stats are full."""
        pet = Pet.create("Test", "mammal")
        pet.hunger = 1.0
        pet.energy = 1.0
        pet.happiness = 1.0
        pet.exercise = 1.0
        mood = compute_mood(pet)
        assert mood == pytest.approx(1.0)

    def test_compute_mood_all_depleted(self) -> None:
        """compute_mood() returns -1 when all stats are depleted."""
        pet = Pet.create("Test", "mammal")
        pet.hunger = 0.0
        pet.energy = 0.0
        pet.happiness = 0.0
        pet.exercise = 0.0
        mood = compute_mood(pet)
        assert mood == pytest.approx(-1.0)

    def test_mood_text_labels(self) -> None:
        """mood_text() returns correct labels."""
        assert mood_text(-0.8) == "miserable"
        assert mood_text(-0.5) == "sad"
        assert mood_text(0.0) == "neutral"
        assert mood_text(0.4) == "happy"
        assert mood_text(0.8) == "ecstatic"

    def test_skill_level_names(self) -> None:
        """skill_level_name() returns correct labels."""
        assert skill_level_name(0.2) == "L1-Novice"
        assert skill_level_name(0.4) == "L2-Practiced"
        assert skill_level_name(0.7) == "L3-Skilled"
        assert skill_level_name(0.9) == "L4-Expert"

    def test_practice_skill_increases(self) -> None:
        """practice_skill() increases proficiency."""
        pet = Pet.create("Test", "mammal")
        initial = pet.skills.get("foraging", 0.0)
        practice_skill(pet, "foraging", success=True)
        assert pet.skills["foraging"] > initial

    def test_practice_skill_diminishing_returns(self) -> None:
        """practice_skill() has diminishing returns."""
        pet = Pet.create("Test", "mammal")
        # Practice many times - growth should slow down
        gains = []
        for _ in range(10):
            before = pet.skills.get("test_skill", 0.0)
            practice_skill(pet, "test_skill", success=True)
            after = pet.skills["test_skill"]
            gains.append(after - before)
        # Later gains should be smaller (not always, but generally)
        assert sum(gains[:3]) >= sum(gains[-3:])

    def test_practice_skill_failure_half_gain(self) -> None:
        """practice_skill() with success=False gives half gain."""
        pet = Pet.create("Test", "mammal")
        practice_skill(pet, "test", success=True)
        gain_success = pet.skills["test"]
        pet.skills["test"] = 0.0
        practice_skill(pet, "test", success=False)
        gain_failure = pet.skills["test"]
        assert gain_failure < gain_success


# ---------------------------------------------------------------------------
# Test Grid
# ---------------------------------------------------------------------------


class TestGrid:
    """Test Grid operations."""

    def test_grid_in_bounds(self) -> None:
        """Grid.in_bounds() checks boundaries correctly."""
        grid = Grid(width=16, height=16)
        assert grid.in_bounds(0, 0) is True
        assert grid.in_bounds(15, 15) is True
        assert grid.in_bounds(0, 15) is True
        assert grid.in_bounds(15, 0) is True
        assert grid.in_bounds(-1, 0) is False
        assert grid.in_bounds(0, -1) is False
        assert grid.in_bounds(16, 0) is False
        assert grid.in_bounds(0, 16) is False

    def test_grid_is_free(self) -> None:
        """Grid.is_free() returns True for unoccupied in-bounds cells."""
        grid = Grid(width=16, height=16)
        assert grid.is_free(5, 5) is True
        grid.occupy(5, 5, "building1")
        assert grid.is_free(5, 5) is False
        assert grid.is_free(6, 6) is True

    def test_grid_is_free_out_of_bounds(self) -> None:
        """Grid.is_free() returns False for out-of-bounds cells."""
        grid = Grid(width=16, height=16)
        assert grid.is_free(-1, 0) is False
        assert grid.is_free(0, -1) is False
        assert grid.is_free(16, 0) is False
        assert grid.is_free(0, 16) is False

    def test_grid_occupy_and_free(self) -> None:
        """Grid.occupy() and free() manage cell occupation."""
        grid = Grid(width=16, height=16)
        grid.occupy(3, 4, "b1")
        assert grid.occupied[(3, 4)] == "b1"
        grid.free(3, 4)
        assert (3, 4) not in grid.occupied

    def test_grid_free_nonexistent_safe(self) -> None:
        """Grid.free() on non-occupied cell is safe."""
        grid = Grid(width=16, height=16)
        grid.free(99, 99)  # Should not raise

    def test_grid_is_walkable(self) -> None:
        """Grid.is_walkable() returns True for in-bounds cells."""
        grid = Grid(width=16, height=16)
        assert grid.is_walkable(0, 0) is True
        assert grid.is_walkable(15, 15) is True
        assert grid.is_walkable(-1, 0) is False
        assert grid.is_walkable(0, 16) is False


# ---------------------------------------------------------------------------
# Test Manor
# ---------------------------------------------------------------------------


class TestManor:
    """Test Manor container."""

    def test_manor_active_pets_filters(self) -> None:
        """Manor.active_pets() filters by status='active'."""
        manor = Manor()
        pet1 = Pet.create("Active1", "mammal")
        pet2 = Pet.create("Archived", "bird")
        pet2.status = "archived"
        manor.pets[pet1.pet_id] = pet1
        manor.pets[pet2.pet_id] = pet2
        active = manor.active_pets()
        assert len(active) == 1
        assert active[0].pet_id == pet1.pet_id

    def test_manor_find_pet_by_name_case_insensitive(self) -> None:
        """Manor.find_pet_by_name() is case-insensitive."""
        manor = Manor()
        pet = Pet.create("Fluffy", "mammal")
        manor.pets[pet.pet_id] = pet
        assert manor.find_pet_by_name("Fluffy") == pet
        assert manor.find_pet_by_name("fluffy") == pet
        assert manor.find_pet_by_name("FLUFFY") == pet

    def test_manor_find_pet_by_name_active_only(self) -> None:
        """Manor.find_pet_by_name() only returns active pets."""
        manor = Manor()
        pet = Pet.create("Hidden", "reptile")
        pet.status = "archived"
        manor.pets[pet.pet_id] = pet
        assert manor.find_pet_by_name("Hidden") is None

    def test_manor_find_pet_by_name_not_found(self) -> None:
        """Manor.find_pet_by_name() returns None for unknown names."""
        manor = Manor()
        assert manor.find_pet_by_name("Nonexistent") is None

    def test_manor_find_building_at(self) -> None:
        """Manor.find_building_at() returns building at position."""
        manor = Manor()
        bp = Blueprint(
            id="test_bp",
            name="Test",
            theme="test",
            size=(1, 1),
            cells=[(0, 0)],
            cost=10,
            description="Test",
            symbol="T",
            provides="test",
        )
        building = Building.create("test_bp", 5, 5)
        manor.blueprints["test_bp"] = bp
        manor.buildings[building.building_id] = building
        manor.grid.occupy(5, 5, building.building_id)
        assert manor.find_building_at(5, 5) == building
        assert manor.find_building_at(0, 0) is None

    def test_manor_find_buildings_providing(self) -> None:
        """Manor.find_buildings_providing() filters by resource."""
        manor = Manor()
        bp1 = Blueprint(
            id="bp1",
            name="Feeder",
            theme="tech",
            size=(1, 1),
            cells=[(0, 0)],
            cost=10,
            description="",
            symbol="F",
            provides="food",
        )
        bp2 = Blueprint(
            id="bp2",
            name="Shelter",
            theme="forest",
            size=(1, 1),
            cells=[(0, 0)],
            cost=10,
            description="",
            symbol="S",
            provides="shelter",
        )
        b1 = Building.create("bp1", 1, 1)
        b2 = Building.create("bp2", 2, 2)
        manor.blueprints["bp1"] = bp1
        manor.blueprints["bp2"] = bp2
        manor.buildings[b1.building_id] = b1
        manor.buildings[b2.building_id] = b2
        food_buildings = manor.find_buildings_providing("food")
        assert len(food_buildings) == 1
        assert food_buildings[0].building_id == b1.building_id


# ---------------------------------------------------------------------------
# Test Building and Blueprint
# ---------------------------------------------------------------------------


class TestBuildingAndBlueprint:
    """Test Building and Blueprint models."""

    def test_blueprint_from_dict(self) -> None:
        """Blueprint.from_dict() creates blueprint from dict."""
        d = {
            "id": "test_bp",
            "name": "Test Building",
            "theme": "test",
            "size": [2, 3],
            "cells": [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]],
            "cost": 100,
            "description": "A test building",
            "symbol": "X",
            "provides": "test_resource",
        }
        bp = Blueprint.from_dict(d)
        assert bp.id == "test_bp"
        assert bp.name == "Test Building"
        assert bp.theme == "test"
        assert bp.size == (2, 3)
        assert bp.cells == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
        assert bp.cost == 100
        assert bp.description == "A test building"
        assert bp.symbol == "X"
        assert bp.provides == "test_resource"

    def test_building_create(self) -> None:
        """Building.create() creates a building with generated ID."""
        building = Building.create("test_bp", 5, 7, rotation=2)
        assert building.blueprint_id == "test_bp"
        assert building.origin_x == 5
        assert building.origin_y == 7
        assert building.rotation == 2
        assert building.building_id
        assert building.placed_at

    def test_building_to_dict_roundtrip(self) -> None:
        """Building.to_dict() / from_dict() round-trips correctly."""
        building = Building.create("test_bp", 3, 4, rotation=1)
        d = building.to_dict()
        assert d["_type"] == "building_placed"
        assert d["blueprint_id"] == "test_bp"
        assert d["origin_x"] == 3
        assert d["origin_y"] == 4
        assert d["rotation"] == 1

        b2 = Building.from_dict(d)
        assert b2.building_id == building.building_id
        assert b2.blueprint_id == building.blueprint_id
        assert b2.origin_x == building.origin_x
        assert b2.origin_y == building.origin_y
        assert b2.rotation == building.rotation

    def test_building_from_dict_defaults(self) -> None:
        """Building.from_dict() uses defaults for missing keys."""
        d = {
            "building_id": "b123",
            "blueprint_id": "bp1",
            "origin_x": 1,
            "origin_y": 2,
        }
        b = Building.from_dict(d)
        assert b.rotation == 0
        assert b.placed_at == ""

    def test_load_blueprints(self) -> None:
        """load_blueprints() loads blueprints from JSON file."""
        data_dir = Path(__file__).parent.parent / "aeloon" / "plugins" / "PetManor" / "data"
        bps = load_blueprints(data_dir / "blueprints.json")
        assert len(bps) > 0
        assert "forest_hut" in bps
        assert "tech_feeder" in bps
        assert bps["forest_hut"].symbol == "H"
        assert bps["tech_feeder"].provides == "food"


# ---------------------------------------------------------------------------
# Test footprint rotation
# ---------------------------------------------------------------------------


class TestFootprintRotation:
    """Test building footprint rotation."""

    def test_get_rotated_footprint_0_degrees(self) -> None:
        """get_rotated_footprint() with rotation=0 is identity."""
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        result = get_rotated_footprint(cells, 0)
        assert result == [(0, 0), (1, 0), (0, 1), (1, 1)]

    def test_get_rotated_footprint_90_degrees(self) -> None:
        """get_rotated_footprint() with rotation=1 rotates 90deg clockwise."""
        cells = [(1, 0), (2, 0)]  # Horizontal 2-cell
        result = get_rotated_footprint(cells, 1)
        # (x, y) -> (-y, x)
        assert set(result) == {(0, 1), (0, 2)}

    def test_get_rotated_footprint_180_degrees(self) -> None:
        """get_rotated_footprint() with rotation=2 rotates 180deg."""
        cells = [(1, 2), (3, 4)]
        result = get_rotated_footprint(cells, 2)
        # (x, y) -> (-x, -y)
        assert set(result) == {(-1, -2), (-3, -4)}

    def test_get_rotated_footprint_270_degrees(self) -> None:
        """get_rotated_footprint() with rotation=3 rotates 270deg clockwise."""
        cells = [(1, 0), (2, 0)]
        result = get_rotated_footprint(cells, 3)
        # (x, y) -> (y, -x)
        assert set(result) == {(0, -1), (0, -2)}

    def test_get_rotated_footprint_wraps_rotation(self) -> None:
        """get_rotated_footprint() wraps rotation > 3."""
        cells = [(1, 0)]
        assert get_rotated_footprint(cells, 4) == get_rotated_footprint(cells, 0)
        assert get_rotated_footprint(cells, 5) == get_rotated_footprint(cells, 1)

    def test_get_absolute_cells_offset(self) -> None:
        """get_absolute_cells() applies origin offset."""
        bp = Blueprint(
            id="test",
            name="Test",
            theme="test",
            size=(2, 2),
            cells=[(0, 0), (1, 0), (0, 1), (1, 1)],
            cost=10,
            description="",
            symbol="T",
            provides="test",
        )
        building = Building.create("test", 5, 7, rotation=0)
        cells = get_absolute_cells(building, {"test": bp})
        assert set(cells) == {(5, 7), (6, 7), (5, 8), (6, 8)}

    def test_get_absolute_cells_with_rotation(self) -> None:
        """get_absolute_cells() combines offset and rotation."""
        bp = Blueprint(
            id="test",
            name="Test",
            theme="test",
            size=(2, 1),
            cells=[(0, 0), (1, 0)],
            cost=10,
            description="",
            symbol="T",
            provides="test",
        )
        building = Building.create("test", 5, 7, rotation=1)
        cells = get_absolute_cells(building, {"test": bp})
        # (0,0) -> (0,0), (1,0) -> (0,1) after rotation
        # Then add offset (5, 7)
        assert set(cells) == {(5, 7), (5, 8)}

    def test_get_absolute_cells_unknown_blueprint(self) -> None:
        """get_absolute_cells() returns empty list for unknown blueprint."""
        building = Building.create("unknown", 5, 7)
        cells = get_absolute_cells(building, {})
        assert cells == []
