"""Tests for Pet Manor plugin: config, registration, activation, tools, renderer, storage.

Covers: PetManorConfig defaults, PetManorPlugin lifecycle, tool execution,
TextRenderer methods, and storage round-trip.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aeloon.plugins.PetManor.config import PetManorConfig, SimulationConfig
from aeloon.plugins.PetManor.models.building import Building, load_blueprints
from aeloon.plugins.PetManor.models.manor import Manor
from aeloon.plugins.PetManor.models.pet import Pet
from aeloon.plugins.PetManor.plugin import PetManorPlugin
from aeloon.plugins.PetManor.render.text_renderer import TextRenderer
from aeloon.plugins.PetManor.storage.jsonl_store import (
    load_all,
    load_all_pets,
    load_buildings,
    load_pet,
    save_all,
    save_building_placed,
    save_pet,
)
from aeloon.plugins.PetManor.storage.migrate import migrate_if_needed
from aeloon.plugins.PetManor.tools.build_tools import PlaceBuildingTool
from aeloon.plugins.PetManor.tools.manor_tools import ManorStatusTool
from aeloon.plugins.PetManor.tools.pet_tools import GetPetInfoTool, ListPetsTool

from aeloon.plugins._sdk.api import PluginAPI
from aeloon.plugins._sdk.manifest import load_manifest
from aeloon.plugins._sdk.registry import PluginRegistry
from aeloon.plugins._sdk.runtime import PluginRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_loop() -> MagicMock:
    """Mock AgentLoop with provider."""
    loop = MagicMock()
    loop.provider = MagicMock()
    loop.provider.chat = AsyncMock(return_value=MagicMock(content="test response"))
    loop.model = "test-model"
    loop.profiler = MagicMock(enabled=False)
    return loop


@pytest.fixture
def plugin_api(mock_agent_loop: MagicMock, tmp_path: Path) -> PluginAPI:
    """PluginAPI instance for testing."""
    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=mock_agent_loop,
        plugin_id="aeloon.pet_manor",
        config={},
        storage_base=tmp_path,
    )
    return PluginAPI(
        plugin_id="aeloon.pet_manor",
        version="1.0.0",
        config={},
        runtime=runtime,
        registry=registry,
    )


@pytest.fixture
def mock_engine() -> MagicMock:
    """Mock SimulationEngine."""
    engine = MagicMock()
    event = MagicMock()
    event.is_set = MagicMock(return_value=False)
    engine._running = event
    engine.stop = MagicMock()
    engine.drain_notifications = MagicMock(return_value=[])
    return engine


@pytest.fixture
def populated_manor() -> Manor:
    """Manor with test data."""
    data_dir = Path(__file__).parent.parent / "aeloon" / "plugins" / "PetManor" / "data"
    blueprints = load_blueprints(data_dir / "blueprints.json")
    manor = Manor(blueprints=blueprints)

    # Add test pet
    pet = Pet.create("TestPet", "mammal")
    pet.pos_x = 5
    pet.pos_y = 5
    manor.pets[pet.pet_id] = pet

    return manor


# ---------------------------------------------------------------------------
# Test PetManorConfig
# ---------------------------------------------------------------------------


class TestPetManorConfig:
    """Test PetManorConfig schema."""

    def test_defaults(self) -> None:
        """PetManorConfig has sensible defaults."""
        cfg = PetManorConfig()
        assert cfg.enabled is True
        assert cfg.max_pets == 6

    def test_simulation_defaults(self) -> None:
        """SimulationConfig has correct defaults."""
        cfg = PetManorConfig()
        assert cfg.simulation.auto_start is False
        assert cfg.simulation.tick_interval == 0.5
        assert cfg.simulation.default_speed == 1
        assert cfg.simulation.auto_save_interval == 60

    def test_custom_values(self) -> None:
        """PetManorConfig accepts custom values."""
        cfg = PetManorConfig(max_pets=10, enabled=False)
        assert cfg.max_pets == 10
        assert cfg.enabled is False

    def test_simulation_custom_values(self) -> None:
        """SimulationConfig accepts custom values."""
        sim = SimulationConfig(
            auto_start=True,
            tick_interval=1.0,
            default_speed=5,
            auto_save_interval=120,
        )
        assert sim.auto_start is True
        assert sim.tick_interval == 1.0
        assert sim.default_speed == 5
        assert sim.auto_save_interval == 120


# ---------------------------------------------------------------------------
# Test PetManorPlugin Registration
# ---------------------------------------------------------------------------


class TestPetManorPluginRegistration:
    """Test PetManorPlugin.register()."""

    def test_register_creates_commands(self, plugin_api: PluginAPI) -> None:
        """register() creates command records."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)

        command_names = {r.name for r in plugin_api._pending_commands}
        assert "pet" in command_names
        assert "build" in command_names
        assert "manor" in command_names
        assert "sim" in command_names

    def test_register_creates_services(self, plugin_api: PluginAPI) -> None:
        """register() creates service records."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)

        service_names = {r.name for r in plugin_api._pending_services}
        assert "simulation" in service_names

    def test_register_creates_config_schema(self, plugin_api: PluginAPI) -> None:
        """register() creates config schema."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)

        assert len(plugin_api._pending_config_schemas) == 1

    def test_register_creates_cli(self, plugin_api: PluginAPI) -> None:
        """register() creates CLI records."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)

        cli_names = {r.name for r in plugin_api._pending_cli}
        assert "manor" in cli_names

    def test_register_creates_hooks(self, plugin_api: PluginAPI) -> None:
        """register() creates hook records."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)

        assert len(plugin_api._pending_hooks) > 0

    def test_commit_registers_commands(self, mock_agent_loop: MagicMock, tmp_path: Path) -> None:
        """After commit, commands are in registry."""
        registry = PluginRegistry()
        runtime = PluginRuntime(
            agent_loop=mock_agent_loop,
            plugin_id="aeloon.pet_manor",
            config={},
            storage_base=tmp_path,
        )
        api = PluginAPI(
            plugin_id="aeloon.pet_manor",
            version="1.0.0",
            config={},
            runtime=runtime,
            registry=registry,
        )

        plugin = PetManorPlugin()
        plugin.register(api)
        api._commit()

        assert "pet" in registry.commands
        assert "build" in registry.commands
        assert "manor" in registry.commands
        assert "sim" in registry.commands


# ---------------------------------------------------------------------------
# Test PetManorPlugin Activation
# ---------------------------------------------------------------------------


class TestPetManorPluginActivation:
    """Test PetManorPlugin lifecycle."""

    @pytest.mark.asyncio
    async def test_activate_loads_manor(self, plugin_api: PluginAPI) -> None:
        """activate() loads manor state."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)
        api = plugin_api
        await plugin.activate(api)

        assert plugin._manor is not None
        assert isinstance(plugin._manor, Manor)

    @pytest.mark.asyncio
    async def test_activate_loads_blueprints(self, plugin_api: PluginAPI) -> None:
        """activate() loads blueprints."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)
        await plugin.activate(plugin_api)

        assert plugin._manor is not None
        assert len(plugin._manor.blueprints) > 0
        assert "forest_hut" in plugin._manor.blueprints

    @pytest.mark.asyncio
    async def test_deactivate_saves_state(self, plugin_api: PluginAPI) -> None:
        """deactivate() saves manor state."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)
        await plugin.activate(plugin_api)

        # Add a pet
        pet = Pet.create("Test", "mammal")
        plugin._manor.pets[pet.pet_id] = pet

        await plugin.deactivate()

        # Verify file was created
        pet_file = plugin_api.runtime.storage_path / "pets" / f"{pet.pet_id}.jsonl"
        assert pet_file.exists()


# ---------------------------------------------------------------------------
# Test Health Check
# ---------------------------------------------------------------------------


class TestPetManorPluginHealthCheck:
    """Test health_check()."""

    def test_health_check_before_activation(self) -> None:
        """health_check() returns False before activation."""
        plugin = PetManorPlugin()
        health = plugin.health_check()
        assert health["manor_loaded"] is False
        assert health["engine_active"] is False
        assert health["pets"] == 0
        assert health["buildings"] == 0

    def test_health_check_after_activation(
        self, plugin_api: PluginAPI, mock_engine: MagicMock
    ) -> None:
        """health_check() returns correct status after activation."""
        plugin = PetManorPlugin()
        plugin.register(plugin_api)
        plugin._manor = Manor()
        plugin._engine = mock_engine

        health = plugin.health_check()
        assert health["manor_loaded"] is True
        assert health["engine_active"] is False  # Mock returns False
        assert health["pets"] == 0
        assert health["buildings"] == 0


# ---------------------------------------------------------------------------
# Test Tools
# ---------------------------------------------------------------------------


class TestPetManorTools:
    """Test agent tools."""

    def test_get_pet_info_tool_schema(self) -> None:
        """GetPetInfoTool has correct schema."""
        plugin = PetManorPlugin()
        tool = GetPetInfoTool(plugin=plugin)
        assert tool.name == "get_pet_info"
        assert "pet_name" in tool.parameters["properties"]
        assert "pet_name" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_get_pet_info_tool_execute(self, populated_manor: Manor) -> None:
        """GetPetInfoTool.execute() returns pet info."""
        plugin = PetManorPlugin()
        plugin._manor = populated_manor

        tool = GetPetInfoTool(plugin=plugin)
        pet = list(populated_manor.pets.values())[0]
        result = await tool.execute(pet_name=pet.name)

        assert pet.name in result
        assert pet.species in result
        assert "Status:" in result

    @pytest.mark.asyncio
    async def test_get_pet_info_tool_not_found(self) -> None:
        """GetPetInfoTool.execute() returns error for unknown pet."""
        plugin = PetManorPlugin()
        plugin._manor = Manor()

        tool = GetPetInfoTool(plugin=plugin)
        result = await tool.execute(pet_name="Ghost")

        assert "Error:" in result
        assert "not found" in result

    def test_list_pets_tool_schema(self) -> None:
        """ListPetsTool has correct schema."""
        plugin = PetManorPlugin()
        tool = ListPetsTool(plugin=plugin)
        assert tool.name == "list_pets"
        assert tool.parameters["properties"] == {}

    @pytest.mark.asyncio
    async def test_list_pets_tool_empty(self) -> None:
        """ListPetsTool.execute() returns empty list."""
        plugin = PetManorPlugin()
        plugin._manor = Manor()

        tool = ListPetsTool(plugin=plugin)
        result = await tool.execute()

        assert "No active pets" in result

    @pytest.mark.asyncio
    async def test_list_pets_tool_populated(self, populated_manor: Manor) -> None:
        """ListPetsTool.execute() returns pets."""
        plugin = PetManorPlugin()
        plugin._manor = populated_manor

        tool = ListPetsTool(plugin=plugin)
        result = await tool.execute()

        assert "TestPet" in result

    def test_manor_status_tool_schema(self) -> None:
        """ManorStatusTool has correct schema."""
        plugin = PetManorPlugin()
        tool = ManorStatusTool(plugin=plugin)
        assert tool.name == "manor_status"
        assert tool.parameters["properties"] == {}

    @pytest.mark.asyncio
    async def test_manor_status_tool_execute(self, populated_manor: Manor) -> None:
        """ManorStatusTool.execute() returns status."""
        plugin = PetManorPlugin()
        plugin._manor = populated_manor

        tool = ManorStatusTool(plugin=plugin)
        result = await tool.execute()

        assert "Pets:" in result
        assert "Buildings:" in result

    def test_place_building_tool_schema(self) -> None:
        """PlaceBuildingTool has correct schema."""
        plugin = PetManorPlugin()
        tool = PlaceBuildingTool(plugin=plugin)
        assert tool.name == "place_building"
        assert "blueprint_id" in tool.parameters["required"]
        assert "x" in tool.parameters["required"]
        assert "y" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_place_building_tool_execute(
        self, populated_manor: Manor, tmp_path: Path
    ) -> None:
        """PlaceBuildingTool.execute() places building."""
        plugin = PetManorPlugin()
        plugin._manor = populated_manor
        plugin._api = MagicMock()
        plugin._api.runtime.storage_path = tmp_path

        tool = PlaceBuildingTool(plugin=plugin)
        result = await tool.execute(blueprint_id="forest_hut", x=10, y=10)

        assert "Placed" in result
        assert "Forest Hut" in result


# ---------------------------------------------------------------------------
# Test TextRenderer
# ---------------------------------------------------------------------------


class TestTextRenderer:
    """Test TextRenderer methods."""

    @pytest.fixture
    def renderer(self) -> TextRenderer:
        """TextRenderer instance."""
        return TextRenderer()

    def test_success(self, renderer: TextRenderer) -> None:
        """success() returns message as-is."""
        assert renderer.success("Done") == "Done"

    def test_error(self, renderer: TextRenderer) -> None:
        """error() prefixes message."""
        assert renderer.error("Failed") == "Error: Failed"

    def test_info(self, renderer: TextRenderer) -> None:
        """info() returns message as-is."""
        assert renderer.info("Info") == "Info"

    def test_warn(self, renderer: TextRenderer) -> None:
        """warn() prefixes message."""
        assert renderer.warn("Warning") == "Warning: Warning"

    def test_notification(self, renderer: TextRenderer) -> None:
        """notification() prefixes message."""
        assert renderer.notification("Alert") == "[!] Alert"

    def test_render_pet_list_empty(self, renderer: TextRenderer) -> None:
        """render_pet_list() shows message for empty list."""
        result = renderer.render_pet_list([])
        assert "No pets yet" in result

    def test_render_pet_list_with_pets(self, renderer: TextRenderer) -> None:
        """render_pet_list() shows pets as table."""
        pet = Pet.create("Fluffy", "mammal")
        result = renderer.render_pet_list([pet])
        assert "Fluffy" in result
        assert "|" in result  # Table format

    def test_render_pet_info(self, renderer: TextRenderer) -> None:
        """render_pet_info() shows detailed pet info."""
        pet = Pet.create("Test", "bird")
        result = renderer.render_pet_info(pet, {})
        assert "Test" in result
        assert "bird" in result
        assert "Stats" in result
        assert "Personality" in result

    def test_render_grid(self, renderer: TextRenderer, populated_manor: Manor) -> None:
        """render_grid() shows ASCII grid."""
        result = renderer.render_grid(populated_manor)
        assert "## Manor Grid" in result
        assert "Legend:" in result

    def test_render_building_list(self, renderer: TextRenderer) -> None:
        """render_building_list() shows blueprints as table."""
        data_dir = Path(__file__).parent.parent / "aeloon" / "plugins" / "PetManor" / "data"
        blueprints = load_blueprints(data_dir / "blueprints.json")
        result = renderer.render_building_list(blueprints)
        assert "forest_hut" in result
        assert "|" in result  # Table format

    def test_render_manor_status(self, renderer: TextRenderer) -> None:
        """render_manor_status() shows overview."""
        manor = Manor()
        result = renderer.render_manor_status(manor)
        assert "Pets:" in result
        assert "Buildings:" in result
        assert "Game time:" in result

    def test_render_help(self, renderer: TextRenderer) -> None:
        """render_help() shows command help."""
        result = renderer.render_help()
        assert "Pet Manor Commands" in result
        assert "/pet create" in result
        assert "/build place" in result


# ---------------------------------------------------------------------------
# Test Storage
# ---------------------------------------------------------------------------


class TestStorage:
    """Test storage round-trip."""

    def test_save_and_load_pet(self, tmp_path: Path) -> None:
        """save_pet() and load_pet() round-trip correctly."""
        pet = Pet.create("SaveTest", "reptile")
        save_pet(tmp_path, pet)

        loaded = load_pet(tmp_path, pet.pet_id)
        assert loaded is not None
        assert loaded.name == pet.name
        assert loaded.species == pet.species

    def test_load_nonexistent_pet(self, tmp_path: Path) -> None:
        """load_pet() returns None for nonexistent pet."""
        result = load_pet(tmp_path, "ghost_id")
        assert result is None

    def test_save_and_load_all_pets(self, tmp_path: Path) -> None:
        """save_all() and load_all_pets() round-trip correctly."""
        data_dir = Path(__file__).parent.parent / "aeloon" / "plugins" / "PetManor" / "data"
        blueprints = load_blueprints(data_dir / "blueprints.json")
        manor = Manor(blueprints=blueprints)

        pet1 = Pet.create("Pet1", "mammal")
        pet2 = Pet.create("Pet2", "bird")
        manor.pets[pet1.pet_id] = pet1
        manor.pets[pet2.pet_id] = pet2

        save_all(tmp_path, manor)
        loaded = load_all_pets(tmp_path)

        assert len(loaded) == 2
        names = {p.name for p in loaded}
        assert "Pet1" in names
        assert "Pet2" in names

    def test_load_all_reconstructs_manor(self, tmp_path: Path) -> None:
        """load_all() reconstructs Manor from disk."""
        data_dir = Path(__file__).parent.parent / "aeloon" / "plugins" / "PetManor" / "data"
        blueprints = load_blueprints(data_dir / "blueprints.json")

        # Save original manor
        original = Manor(blueprints=blueprints)
        pet = Pet.create("LoadTest", "mammal")
        original.pets[pet.pet_id] = pet
        save_all(tmp_path, original)

        # Load reconstructed manor
        loaded = load_all(tmp_path, blueprints)

        assert len(loaded.pets) == 1
        assert pet.pet_id in loaded.pets
        assert loaded.pets[pet.pet_id].name == "LoadTest"

    def test_save_and_load_buildings(self, tmp_path: Path) -> None:
        """save_building_placed() and load_buildings() round-trip correctly."""

        building = Building.create("forest_hut", 5, 5)
        save_building_placed(tmp_path, building)

        loaded = load_buildings(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].building_id == building.building_id


# ---------------------------------------------------------------------------
# Test Migration
# ---------------------------------------------------------------------------


class TestMigration:
    """Test storage migration."""

    def test_migrate_if_needed_no_old_path(self, tmp_path: Path, monkeypatch) -> None:
        """migrate_if_needed() returns False when no old path exists."""
        # Mock Path.home() to return tmp_path to avoid actual home dir checks
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "fake_home")
        result = migrate_if_needed(tmp_path)
        assert result is False

    def test_migrate_if_needed_new_exists(self, tmp_path: Path) -> None:
        """migrate_if_needed() returns False when new storage exists."""
        # Create new storage
        (tmp_path / "pets").mkdir(parents=True)
        result = migrate_if_needed(tmp_path)
        assert result is False


# ---------------------------------------------------------------------------
# Test Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    """Test manifest loading."""

    def test_load_bundled_manifest(self) -> None:
        """Load the actual aeloon.plugin.json from pet_manor plugin."""
        manifest_path = (
            Path(__file__).parent.parent / "aeloon" / "plugins" / "PetManor" / "aeloon.plugin.json"
        )
        if not manifest_path.exists():
            pytest.skip("Bundled manifest not found")
        m = load_manifest(manifest_path)
        assert m.id == "aeloon.pet_manor"
        assert "pet" in m.provides.commands
        assert "build" in m.provides.commands
