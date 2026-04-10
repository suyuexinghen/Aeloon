"""JSONL-backed persistence for PlanPackage records.

Append-only storage mirroring the SE JsonlStorage pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from ..models import PlanPackage


class PlanStore:
    """Append-only JSONL storage for PlanPackage records."""

    def __init__(self, storage_dir: Path | str) -> None:
        self._dir = Path(storage_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, pkg: PlanPackage) -> None:
        """Append a PlanPackage record to its project JSONL file."""
        path = self._project_path(pkg.project_id)
        record = {"_type": "plan_package", **pkg.model_dump(mode="json")}
        self._append(path, record)
        logger.debug("Saved PlanPackage for project {}", pkg.project_id)

    def load_latest(self, project_id: str) -> PlanPackage | None:
        """Load the most recent PlanPackage record for a project."""
        path = self._project_path(project_id)
        if not path.exists():
            return None
        last: dict | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if obj.get("_type") == "plan_package":
                    last = obj
            except json.JSONDecodeError:
                continue
        if last is None:
            return None
        last.pop("_type", None)
        return PlanPackage.model_validate(last)

    def list_project_ids(self) -> list[str]:
        """Return all project IDs that have stored records."""
        ids: list[str] = []
        for jsonl_file in sorted(self._dir.glob("*.jsonl")):
            ids.append(jsonl_file.stem)
        return ids

    def _project_path(self, project_id: str) -> Path:
        return self._dir / f"{project_id}.jsonl"

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
