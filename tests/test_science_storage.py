"""Tests for JsonlStorage in aeloon/plugins/science/storage/jsonl.py."""

from __future__ import annotations

from aeloon.plugins.ScienceResearch.storage.jsonl import JsonlStorage
from aeloon.plugins.ScienceResearch.task import Execution, Task, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(goal: str = "Test goal") -> Task:
    return Task(goal=goal)


def _make_execution(task_id: str, node_id: str = "n1", output: str = "result") -> Execution:
    return Execution(task_id=task_id, node_id=node_id, output=output)


# ---------------------------------------------------------------------------
# Task CRUD round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_task_round_trip(tmp_path):
    storage = JsonlStorage(tmp_path)
    task = _make_task("Summarise protein research")

    storage.save_task(task)
    loaded = storage.load_task(task.task_id)

    assert loaded is not None
    assert loaded.task_id == task.task_id
    assert loaded.goal == task.goal
    assert loaded.status == task.status


def test_load_nonexistent_task_returns_none(tmp_path):
    storage = JsonlStorage(tmp_path)
    result = storage.load_task("task_does_not_exist")
    assert result is None


# ---------------------------------------------------------------------------
# Execution CRUD round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_execution_round_trip(tmp_path):
    storage = JsonlStorage(tmp_path)
    task = _make_task()
    storage.save_task(task)

    execution = _make_execution(task.task_id, output="execution output")
    storage.save_execution(execution)

    executions = storage.load_executions(task.task_id)
    assert len(executions) == 1
    assert executions[0].execution_id == execution.execution_id
    assert executions[0].output == "execution output"


def test_load_executions_for_nonexistent_task_returns_empty(tmp_path):
    storage = JsonlStorage(tmp_path)
    result = storage.load_executions("nonexistent_task_id")
    assert result == []


def test_multiple_executions_all_loaded(tmp_path):
    storage = JsonlStorage(tmp_path)
    task = _make_task()
    storage.save_task(task)

    ex1 = _make_execution(task.task_id, node_id="n1", output="first")
    ex2 = _make_execution(task.task_id, node_id="n2", output="second")
    storage.save_execution(ex1)
    storage.save_execution(ex2)

    executions = storage.load_executions(task.task_id)
    assert len(executions) == 2
    outputs = {e.output for e in executions}
    assert "first" in outputs
    assert "second" in outputs


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


def test_list_tasks_returns_expected_keys(tmp_path):
    storage = JsonlStorage(tmp_path)
    task = _make_task("Test list tasks")
    storage.save_task(task)

    summaries = storage.list_tasks()
    assert len(summaries) >= 1

    summary = next(s for s in summaries if s["task_id"] == task.task_id)
    assert "task_id" in summary
    assert "goal" in summary
    assert "status" in summary
    assert "created_at" in summary
    assert "updated_at" in summary


def test_list_tasks_empty_storage(tmp_path):
    storage = JsonlStorage(tmp_path)
    assert storage.list_tasks() == []


# ---------------------------------------------------------------------------
# update_task_status
# ---------------------------------------------------------------------------


def test_update_task_status_changes_status_on_load(tmp_path):
    storage = JsonlStorage(tmp_path)
    task = _make_task("Status update test")
    storage.save_task(task)

    storage.update_task_status(task.task_id, TaskStatus.COMPLETED)

    loaded = storage.load_task(task.task_id)
    assert loaded is not None
    assert loaded.status == TaskStatus.COMPLETED


def test_update_task_status_for_nonexistent_task_does_not_raise(tmp_path):
    storage = JsonlStorage(tmp_path)
    # Should log a warning but not raise
    storage.update_task_status("ghost_task_id", TaskStatus.FAILED)


# ---------------------------------------------------------------------------
# Multiple saves of the same task -> load returns last saved state
# ---------------------------------------------------------------------------


def test_multiple_saves_append_to_jsonl_and_load_returns_last(tmp_path):
    storage = JsonlStorage(tmp_path)
    task = _make_task("Append test")
    storage.save_task(task)

    task.status = TaskStatus.RUNNING
    storage.save_task(task)

    task.status = TaskStatus.COMPLETED
    storage.save_task(task)

    # The JSONL file now has 3 lines; load_task returns the last task record
    loaded = storage.load_task(task.task_id)
    assert loaded is not None
    assert loaded.status == TaskStatus.COMPLETED

    jsonl_path = tmp_path / "tasks" / f"{task.task_id}.jsonl"
    lines = [line for line in jsonl_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
