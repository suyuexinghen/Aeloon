"""Tests for Pet Manor command handlers: PetCommands, BuildCommands, ManorCommands, SimCommands.

Covers: All subcommands return strings, error handling, state mutations, and storage interaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aeloon.plugins.PetManor.commands.build_commands import BuildCommands
from aeloon.plugins.PetManor.commands.manor_commands import ManorCommands
from aeloon.plugins.PetManor.commands.pet_commands import PetCommands
from aeloon.plugins.PetManor.commands.sim_commands import SimCommands
from aeloon.plugins.PetManor.models.building import load_blueprints
from aeloon.plugins.PetManor.models.manor import Manor

# ---------------------------------------------------------------------------
# Mock engine for SimCommands and ManorCommands
# ---------------------------------------------------------------------------


class MockEngine:
    """Mock SimulationEngine for testing."""

    def __init__(self) -> None:
        self.speed_multiplier = 1
        self._running = MockEvent()

    def start(self) -> None:
        self._running.set()

    def pause(self) -> None:
        self._running.clear()


class MockEvent:
    """Mock threading.Event for testing."""

    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True

    def clear(self) -> None:
        self._is_set = False


# ---------------------------------------------------------------------------
# Test PetCommands
# ---------------------------------------------------------------------------


class TestPetCommands:
    """Test /pet command handler."""

    @pytest.fixture
    def manor(self, tmp_path: Path) -> Manor:
        """Fresh manor for each test."""
        return Manor()

    @pytest.fixture
    def pet_commands(self, manor: Manor, tmp_path: Path) -> PetCommands:
        """PetCommands instance with test storage."""
        return PetCommands(manor, tmp_path)

    def test_create_pet_success(self, pet_commands: PetCommands, manor: Manor) -> None:
        """Pet create creates a new pet."""
        result = pet_commands.execute("create", ["Fluffy", "mammal"])
        assert "Adopted" in result
        assert "Fluffy" in result
        assert len(manor.pets) == 1
        assert manor.find_pet_by_name("Fluffy") is not None

    def test_create_pet_missing_args(self, pet_commands: PetCommands) -> None:
        """Pet create with missing args returns error."""
        result = pet_commands.execute("create", ["Fluffy"])
        assert "Error:" in result
        assert "Usage:" in result

    def test_create_pet_invalid_species(self, pet_commands: PetCommands) -> None:
        """Pet create with invalid species returns error."""
        result = pet_commands.execute("create", ["Fluffy", "dragon"])
        assert "Error:" in result
        assert "Unknown species" in result

    def test_create_pet_duplicate_name(self, pet_commands: PetCommands, manor: Manor) -> None:
        """Pet create with duplicate name returns error."""
        pet_commands.execute("create", ["Fluffy", "mammal"])
        result = pet_commands.execute("create", ["Fluffy", "bird"])
        assert "Error:" in result
        assert "already exists" in result

    def test_create_pet_max_limit(self, pet_commands: PetCommands, manor: Manor) -> None:
        """Pet create enforces MAX_PETS limit."""
        for i in range(manor.MAX_PETS):
            pet_commands.execute("create", [f"Pet{i}", "mammal"])
        result = pet_commands.execute("create", ["TooMany", "mammal"])
        assert "Error:" in result
        assert "Maximum" in result

    def test_list_empty(self, pet_commands: PetCommands) -> None:
        """Pet list with no pets returns message."""
        result = pet_commands.execute("list", [])
        assert "No pets yet" in result

    def test_list_shows_pets(self, pet_commands: PetCommands) -> None:
        """Pet list shows active pets."""
        pet_commands.execute("create", ["Fluffy", "mammal"])
        result = pet_commands.execute("list", [])
        assert "Fluffy" in result
        assert "|" in result  # Table format

    def test_info_missing_args(self, pet_commands: PetCommands) -> None:
        """Pet info with missing args returns error."""
        result = pet_commands.execute("info", [])
        assert "Error:" in result
        assert "Usage:" in result

    def test_info_pet_not_found(self, pet_commands: PetCommands) -> None:
        """Pet info for unknown pet returns error."""
        result = pet_commands.execute("info", ["Ghost"])
        assert "Error:" in result
        assert "not found" in result

    def test_info_shows_details(self, pet_commands: PetCommands) -> None:
        """Pet info shows pet details."""
        pet_commands.execute("create", ["Whiskers", "bird"])
        result = pet_commands.execute("info", ["Whiskers"])
        assert "Whiskers" in result
        assert "bird" in result
        assert "Stats" in result
        assert "Personality" in result

    def test_feed_missing_args(self, pet_commands: PetCommands) -> None:
        """Pet feed with missing args returns error."""
        result = pet_commands.execute("feed", [])
        assert "Error:" in result
        assert "Usage:" in result

    def test_feed_pet_not_found(self, pet_commands: PetCommands) -> None:
        """Pet feed for unknown pet returns error."""
        result = pet_commands.execute("feed", ["Ghost"])
        assert "Error:" in result
        assert "not found" in result

    def test_feed_increases_hunger(self, pet_commands: PetCommands, manor: Manor) -> None:
        """Pet feed increases hunger stat."""
        pet_commands.execute("create", ["Hungry", "reptile"])
        pet = manor.find_pet_by_name("Hungry")
        assert pet is not None
        pet.hunger = 0.5
        result = pet_commands.execute("feed", ["Hungry"])
        assert "Fed" in result
        assert pet.hunger == pytest.approx(0.9)

    def test_feed_clamps_at_max(self, pet_commands: PetCommands, manor: Manor) -> None:
        """Pet feed clamps hunger at 1.0."""
        pet_commands.execute("create", ["Full", "mammal"])
        pet = manor.find_pet_by_name("Full")
        assert pet is not None
        pet.hunger = 0.9
        pet_commands.execute("feed", ["Full"])
        assert pet.hunger == 1.0

    def test_play_missing_args(self, pet_commands: PetCommands) -> None:
        """Pet play with missing args returns error."""
        result = pet_commands.execute("play", [])
        assert "Error:" in result
        assert "Usage:" in result

    def test_play_pet_not_found(self, pet_commands: PetCommands) -> None:
        """Pet play for unknown pet returns error."""
        result = pet_commands.execute("play", ["Ghost"])
        assert "Error:" in result
        assert "not found" in result

    def test_play_updates_stats(self, pet_commands: PetCommands, manor: Manor) -> None:
        """Pet play updates happiness, exercise, energy."""
        pet_commands.execute("create", ["Playful", "bird"])
        pet = manor.find_pet_by_name("Playful")
        assert pet is not None
        result = pet_commands.execute("play", ["Playful"])
        assert "Played with" in result
        assert pet.happiness == pytest.approx(1.0)
        assert pet.exercise == pytest.approx(1.0)
        assert pet.energy == pytest.approx(0.9)

    def test_rename_missing_args(self, pet_commands: PetCommands) -> None:
        """Pet rename with missing args returns error."""
        result = pet_commands.execute("rename", ["Old"])
        assert "Error:" in result
        assert "Usage:" in result

    def test_rename_pet_not_found(self, pet_commands: PetCommands) -> None:
        """Pet rename for unknown pet returns error."""
        result = pet_commands.execute("rename", ["Ghost", "NewName"])
        assert "Error:" in result
        assert "not found" in result

    def test_rename_duplicate_name(self, pet_commands: PetCommands) -> None:
        """Pet rename to existing name returns error."""
        pet_commands.execute("create", ["Pet1", "mammal"])
        pet_commands.execute("create", ["Pet2", "bird"])
        result = pet_commands.execute("rename", ["Pet1", "Pet2"])
        assert "Error:" in result
        assert "already exists" in result

    def test_rename_success(self, pet_commands: PetCommands, manor: Manor) -> None:
        """Pet rename changes pet name."""
        pet_commands.execute("create", ["OldName", "reptile"])
        result = pet_commands.execute("rename", ["OldName", "NewName"])
        assert "Renamed" in result
        assert "NewName" in result
        assert manor.find_pet_by_name("NewName") is not None
        assert manor.find_pet_by_name("OldName") is None

    def test_archive_missing_args(self, pet_commands: PetCommands) -> None:
        """Pet archive with missing args returns error."""
        result = pet_commands.execute("archive", [])
        assert "Error:" in result
        assert "Usage:" in result

    def test_archive_pet_not_found(self, pet_commands: PetCommands) -> None:
        """Pet archive for unknown pet returns error."""
        result = pet_commands.execute("archive", ["Ghost"])
        assert "Error:" in result
        assert "not found" in result

    def test_archive_success(self, pet_commands: PetCommands, manor: Manor) -> None:
        """Pet archive sets status to archived."""
        pet_commands.execute("create", ["ToArchive", "mammal"])
        result = pet_commands.execute("archive", ["ToArchive"])
        assert "Archived" in result
        pet = manor.find_pet_by_name("ToArchive")
        assert pet is None  # Not found because find_pet_by_name filters active

    def test_help(self, pet_commands: PetCommands) -> None:
        """Pet help returns usage info."""
        result = pet_commands.execute("help", [])
        assert "/pet create" in result
        assert "/pet list" in result

    def test_unknown_subcommand(self, pet_commands: PetCommands) -> None:
        """Pet unknown subcommand returns error."""
        result = pet_commands.execute("bogus", [])
        assert "Error:" in result
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# Test BuildCommands
# ---------------------------------------------------------------------------


class TestBuildCommands:
    """Test /build command handler."""

    @pytest.fixture
    def manor(self, tmp_path: Path) -> Manor:
        """Manor with blueprints loaded."""
        data_dir = Path(__file__).parent.parent / "aeloon" / "plugins" / "PetManor" / "data"
        blueprints = load_blueprints(data_dir / "blueprints.json")
        return Manor(blueprints=blueprints)

    @pytest.fixture
    def build_commands(self, manor: Manor, tmp_path: Path) -> BuildCommands:
        """BuildCommands instance."""
        return BuildCommands(manor, tmp_path)

    def test_list_shows_blueprints(self, build_commands: BuildCommands) -> None:
        """Build list shows available blueprints."""
        result = build_commands.execute("list", [])
        assert "forest_hut" in result
        assert "tech_feeder" in result
        assert "|" in result  # Table format

    def test_list_empty_blueprints(self, tmp_path: Path) -> None:
        """Build list with no blueprints returns message."""
        manor = Manor(blueprints={})
        commands = BuildCommands(manor, tmp_path)
        result = commands.execute("list", [])
        assert "No blueprints" in result

    def test_place_missing_args(self, build_commands: BuildCommands) -> None:
        """Build place with missing args returns error."""
        result = build_commands.execute("place", ["forest_hut"])
        assert "Error:" in result
        assert "Usage:" in result

    def test_place_unknown_blueprint(self, build_commands: BuildCommands) -> None:
        """Build place with unknown blueprint returns error."""
        result = build_commands.execute("place", ["unknown_bp", "5", "5"])
        assert "Error:" in result
        assert "Unknown blueprint" in result

    def test_place_invalid_coords(self, build_commands: BuildCommands) -> None:
        """Build place with invalid coords returns error."""
        result = build_commands.execute("place", ["forest_hut", "abc", "5"])
        assert "Error:" in result
        assert "integers" in result

    def test_place_out_of_bounds(self, build_commands: BuildCommands) -> None:
        """Build place out of bounds returns error."""
        result = build_commands.execute("place", ["forest_hut", "15", "15"])
        assert "Error:" in result
        assert "out of bounds" in result

    def test_place_occupied_cell(self, build_commands: BuildCommands, manor: Manor) -> None:
        """Build place on occupied cell returns error."""
        # Place first building
        build_commands.execute("place", ["forest_hut", "2", "2"])
        # Try to place another at overlapping position
        result = build_commands.execute("place", ["forest_tree", "2", "2"])
        assert "Error:" in result
        assert "occupied" in result

    def test_place_success(self, build_commands: BuildCommands, manor: Manor) -> None:
        """Build place successfully places building."""
        result = build_commands.execute("place", ["forest_tree", "5", "5"])
        assert "Placed" in result
        assert "Great Tree" in result
        assert len(manor.buildings) == 1
        assert manor.grid.occupied.get((5, 5))

    def test_remove_missing_args(self, build_commands: BuildCommands) -> None:
        """Build remove with missing args returns error."""
        result = build_commands.execute("remove", ["5"])
        assert "Error:" in result
        assert "Usage:" in result

    def test_remove_invalid_coords(self, build_commands: BuildCommands) -> None:
        """Build remove with invalid coords returns error."""
        result = build_commands.execute("remove", ["abc", "5"])
        assert "Error:" in result
        assert "integers" in result

    def test_remove_no_building(self, build_commands: BuildCommands) -> None:
        """Build remove with no building returns error."""
        result = build_commands.execute("remove", ["5", "5"])
        assert "Error:" in result
        assert "No building" in result

    def test_remove_success(self, build_commands: BuildCommands, manor: Manor) -> None:
        """Build remove removes building."""
        build_commands.execute("place", ["forest_tree", "5", "5"])
        result = build_commands.execute("remove", ["5", "5"])
        assert "Removed" in result
        assert len(manor.buildings) == 0
        assert (5, 5) not in manor.grid.occupied

    def test_grid_empty(self, build_commands: BuildCommands) -> None:
        """Build grid shows empty grid."""
        result = build_commands.execute("grid", [])
        assert "   " in result  # Header
        assert "." in result  # Grid cells

    def test_grid_with_building(self, build_commands: BuildCommands) -> None:
        """Build grid shows buildings."""
        build_commands.execute("place", ["forest_tree", "5", "5"])
        result = build_commands.execute("grid", [])
        assert "T" in result  # Tree symbol

    def test_grid_with_pet(self, build_commands: BuildCommands, manor: Manor) -> None:
        """Build grid shows pets."""
        from aeloon.plugins.PetManor.models.pet import Pet

        pet = Pet.create("TestPet", "mammal")
        pet.pos_x = 3
        pet.pos_y = 3
        manor.pets[pet.pet_id] = pet
        result = build_commands.execute("grid", [])
        assert "T" in result  # First letter of pet name

    def test_help(self, build_commands: BuildCommands) -> None:
        """Build help returns usage info."""
        result = build_commands.execute("help", [])
        assert "/build list" in result
        assert "/build place" in result

    def test_unknown_subcommand(self, build_commands: BuildCommands) -> None:
        """Build unknown subcommand returns error."""
        result = build_commands.execute("bogus", [])
        assert "Error:" in result
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# Test ManorCommands
# ---------------------------------------------------------------------------


class TestManorCommands:
    """Test /manor command handler."""

    @pytest.fixture
    def engine(self) -> MockEngine:
        """Mock engine."""
        return MockEngine()

    @pytest.fixture
    def manor(self) -> Manor:
        """Fresh manor."""
        return Manor()

    @pytest.fixture
    def manor_commands(self, manor: Manor, engine: MockEngine, tmp_path: Path) -> ManorCommands:
        """ManorCommands instance."""
        return ManorCommands(manor, engine, tmp_path)

    def test_status(self, manor_commands: ManorCommands, manor: Manor) -> None:
        """Manor status shows overview."""
        result = manor_commands.execute("status", [])
        assert "Pets:" in result
        assert "Buildings:" in result
        assert "Game time:" in result

    def test_status_with_pets(self, manor_commands: ManorCommands, manor: Manor) -> None:
        """Manor status with pets shows count."""
        from aeloon.plugins.PetManor.models.pet import Pet

        pet = Pet.create("Test", "mammal")
        manor.pets[pet.pet_id] = pet
        result = manor_commands.execute("status", [])
        assert "Pets: 1/" in result

    def test_save(self, manor_commands: ManorCommands) -> None:
        """Manor save persists state."""
        result = manor_commands.execute("save", [])
        assert "Game saved" in result

    def test_time(self, manor_commands: ManorCommands, manor: Manor) -> None:
        """Manor time shows game time and sim status."""
        manor.game_time = 3600.0  # 1 hour
        result = manor_commands.execute("time", [])
        assert "Game time:" in result
        assert "1.0h" in result
        assert "Simulation:" in result

    def test_time_running(self, manor_commands: ManorCommands, engine: MockEngine) -> None:
        """Manor time shows running status when sim is active."""
        engine.start()
        result = manor_commands.execute("time", [])
        assert "running" in result

    def test_help(self, manor_commands: ManorCommands) -> None:
        """Manor help returns usage info."""
        result = manor_commands.execute("help", [])
        assert "/manor status" in result
        assert "/manor save" in result

    def test_unknown_subcommand(self, manor_commands: ManorCommands) -> None:
        """Manor unknown subcommand returns error."""
        result = manor_commands.execute("bogus", [])
        assert "Error:" in result
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# Test SimCommands
# ---------------------------------------------------------------------------


class TestSimCommands:
    """Test /sim command handler."""

    @pytest.fixture
    def engine(self) -> MockEngine:
        """Mock engine."""
        return MockEngine()

    @pytest.fixture
    def sim_commands(self, engine: MockEngine) -> SimCommands:
        """SimCommands instance."""
        return SimCommands(engine)

    def test_start(self, sim_commands: SimCommands, engine: MockEngine) -> None:
        """Sim start starts the engine."""
        result = sim_commands.execute("start", [])
        assert "started" in result
        assert engine._running.is_set() is True

    def test_pause(self, sim_commands: SimCommands, engine: MockEngine) -> None:
        """Sim pause pauses the engine."""
        engine.start()
        result = sim_commands.execute("pause", [])
        assert "paused" in result
        assert engine._running.is_set() is False

    def test_speed_missing_args(self, sim_commands: SimCommands) -> None:
        """Sim speed with missing args returns error."""
        result = sim_commands.execute("speed", [])
        assert "Error:" in result
        assert "Usage:" in result

    def test_speed_invalid_value(self, sim_commands: SimCommands) -> None:
        """Sim speed with non-integer returns error."""
        result = sim_commands.execute("speed", ["abc"])
        assert "Error:" in result
        assert "integer" in result

    def test_speed_invalid_option(self, sim_commands: SimCommands) -> None:
        """Sim speed with invalid option returns error."""
        result = sim_commands.execute("speed", ["3"])
        assert "Error:" in result
        assert "1, 2, 5, or 10" in result

    def test_speed_valid_options(self, sim_commands: SimCommands, engine: MockEngine) -> None:
        """Sim speed accepts valid options."""
        for speed in [1, 2, 5, 10]:
            result = sim_commands.execute("speed", [str(speed)])
            assert f"{speed}x" in result
            assert engine.speed_multiplier == speed

    def test_help(self, sim_commands: SimCommands) -> None:
        """Sim help returns usage info."""
        result = sim_commands.execute("help", [])
        assert "/sim start" in result
        assert "/sim pause" in result
        assert "/sim speed" in result

    def test_unknown_subcommand(self, sim_commands: SimCommands) -> None:
        """Sim unknown subcommand returns error."""
        result = sim_commands.execute("bogus", [])
        assert "Error:" in result
        assert "Unknown" in result
