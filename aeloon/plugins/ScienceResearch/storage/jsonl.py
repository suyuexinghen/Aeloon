"""JSONL-backed persistence for science tasks and executions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from ..task import Execution, Task, TaskStatus


class JsonlStorage:
    """Append-only JSONL storage for science tasks and their executions.

    Directory layout::

        <root>/
          tasks/
            <task_id>.jsonl    # one JSON line per event (task header + executions)
          artifacts/
            <task_id>/         # large artifact files referenced by path
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()
        self._tasks_dir = self.root / "tasks"
        self._artifacts_dir = self.root / "artifacts"
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    def save_task(self, task: Task) -> None:
        """Persist (or update) a task record."""
        path = self._task_path(task.task_id)
        record = {"_type": "task", **task.model_dump(mode="json")}
        self._append(path, record)
        logger.debug("Saved task {} (status={})", task.task_id, task.status)

    def save_execution(self, execution: Execution) -> None:
        """Append an execution record to the task's JSONL file."""
        path = self._task_path(execution.task_id)
        record = {"_type": "execution", **execution.model_dump(mode="json")}
        self._append(path, record)

    def load_task(self, task_id: str) -> Task | None:
        """Load the most-recent task record for a given task_id."""
        path = self._task_path(task_id)
        if not path.exists():
            return None
        last: dict | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if obj.get("_type") == "task":
                    last = obj
            except json.JSONDecodeError:
                continue
        if last is None:
            return None
        last.pop("_type", None)
        return Task.model_validate(last)

    def load_executions(self, task_id: str) -> list[Execution]:
        """Load all execution records for a task."""
        path = self._task_path(task_id)
        if not path.exists():
            return []
        results: list[Execution] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if obj.get("_type") == "execution":
                    obj.pop("_type", None)
                    results.append(Execution.model_validate(obj))
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("Could not parse execution record: {}", exc)
        return results

    def list_tasks(self) -> list[dict]:
        """Return a summary list of all stored tasks (most-recent state)."""
        summaries: list[dict] = []
        for jsonl_file in sorted(
            self._tasks_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            task = self.load_task(jsonl_file.stem)
            if task is None:
                continue
            summaries.append(
                {
                    "task_id": task.task_id,
                    "goal": task.goal[:80],
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                }
            )
        return summaries

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Shorthand: load task, update status, re-save."""
        task = self.load_task(task_id)
        if task is None:
            logger.warning("update_task_status: task {} not found", task_id)
            return
        task.status = status
        task.updated_at = datetime.now(UTC)
        self.save_task(task)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def artifact_dir(self, task_id: str) -> Path:
        """Return (and create) the artifact directory for a task."""
        d = self._artifacts_dir / task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _task_path(self, task_id: str) -> Path:
        return self._tasks_dir / f"{task_id}.jsonl"

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
