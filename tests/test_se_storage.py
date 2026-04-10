"""Tests for SE JSONL storage."""

from __future__ import annotations

from pathlib import Path

from aeloon.plugins.SoftwareEngineering.storage.jsonl import JsonlStorage
from aeloon.plugins.SoftwareEngineering.task import (
    Execution,
    ExecutionState,
    Project,
    ProjectStatus,
)


class TestJsonlStorage:
    def test_save_and_load_project(self, tmp_path: Path) -> None:
        storage = JsonlStorage(tmp_path / "se")
        project = Project(description="test project")
        storage.save_project(project)
        loaded = storage.load_project(project.project_id)
        assert loaded is not None
        assert loaded.description == "test project"
        assert loaded.project_id == project.project_id

    def test_save_and_load_execution(self, tmp_path: Path) -> None:
        storage = JsonlStorage(tmp_path / "se")
        project = Project(description="test")
        storage.save_project(project)
        ex = Execution(
            project_id=project.project_id,
            node_id="n1",
            output="result",
            state=ExecutionState.WAITING_VALIDATION,
        )
        storage.save_execution(ex)
        executions = storage.load_executions(project.project_id)
        assert len(executions) == 1
        assert executions[0].output == "result"

    def test_list_projects(self, tmp_path: Path) -> None:
        storage = JsonlStorage(tmp_path / "se")
        p1 = Project(description="first")
        p2 = Project(description="second")
        storage.save_project(p1)
        storage.save_project(p2)
        projects = storage.list_projects()
        assert len(projects) == 2

    def test_update_project_status(self, tmp_path: Path) -> None:
        storage = JsonlStorage(tmp_path / "se")
        project = Project(description="test")
        storage.save_project(project)
        storage.update_project_status(project.project_id, ProjectStatus.COMPLETED)
        loaded = storage.load_project(project.project_id)
        assert loaded.status == ProjectStatus.COMPLETED

    def test_load_nonexistent_project(self, tmp_path: Path) -> None:
        storage = JsonlStorage(tmp_path / "se")
        assert storage.load_project("nonexistent") is None

    def test_artifact_dir(self, tmp_path: Path) -> None:
        storage = JsonlStorage(tmp_path / "se")
        artifact_dir = storage.artifact_dir("proj_123")
        assert artifact_dir.exists()

    def test_project_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
        storage = JsonlStorage(tmp_path / "se")
        project = Project(
            description="full test",
            status=ProjectStatus.RUNNING,
        )
        storage.save_project(project)
        loaded = storage.load_project(project.project_id)
        assert loaded.description == "full test"
        assert loaded.status == ProjectStatus.RUNNING
        assert loaded.project_id == project.project_id
