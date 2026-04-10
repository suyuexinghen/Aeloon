"""AssetManager: reusable templates, failure patterns, and task similarity lookup.

Manages:
- Workflow templates extracted from completed tasks
- Failure pattern records for repeated tool/node failures
- Similarity index for "have we done this before?" lookups

Assets are stored as JSON files under ``<storage_dir>/assets/``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class WorkflowTemplate(BaseModel):
    """A reusable plan template extracted from a successfully completed task."""

    template_id: str
    source_task_id: str
    goal_summary: str
    scenario: str  # e.g. "literature_analysis"
    node_ids: list[str]
    node_objectives: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    use_count: int = 0


class FailurePattern(BaseModel):
    """A recorded failure event for a specific tool/node/scenario combination."""

    pattern_id: str
    task_id: str
    node_id: str
    tool_or_capability: str
    error_type: str
    error_summary: str
    scenario: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AssetManager:
    """Manages workflow templates, failure patterns, and similarity lookups.

    Directory layout::

        <storage_dir>/assets/
          templates/
            <template_id>.json
          failures/
            <task_id>_<node_id>.json
          index.json          # lightweight summary for quick queries
    """

    def __init__(self, storage_dir: Path | str) -> None:
        self._root = Path(storage_dir).expanduser() / "assets"
        self._templates_dir = self._root / "templates"
        self._failures_dir = self._root / "failures"
        self._index_path = self._root / "index.json"
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        self._failures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def extract_template(
        self,
        task_id: str,
        goal: str,
        scenario: str,
        node_ids: list[str],
        node_objectives: list[str],
    ) -> WorkflowTemplate:
        """Extract and persist a workflow template from a completed task."""
        template_id = f"tmpl_{task_id[:8]}"
        template = WorkflowTemplate(
            template_id=template_id,
            source_task_id=task_id,
            goal_summary=goal[:120],
            scenario=scenario,
            node_ids=node_ids,
            node_objectives=node_objectives,
        )
        path = self._templates_dir / f"{template_id}.json"
        path.write_text(
            json.dumps(template.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._update_index("template", template_id, {"scenario": scenario, "goal": goal[:80]})
        logger.debug("AssetManager: saved template {} from task {}", template_id, task_id)
        return template

    def list_templates(self, scenario: str | None = None) -> list[WorkflowTemplate]:
        """Return all stored templates, optionally filtered by scenario."""
        results: list[WorkflowTemplate] = []
        for path in sorted(self._templates_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                t = WorkflowTemplate.model_validate(data)
                if scenario is None or t.scenario == scenario:
                    results.append(t)
            except Exception as exc:
                logger.warning("AssetManager: could not load template {}: {}", path.name, exc)
        return results

    def find_similar(self, goal: str, scenario: str | None = None) -> WorkflowTemplate | None:
        """Find the most similar past template using keyword overlap heuristic."""
        templates = self.list_templates(scenario=scenario)
        if not templates:
            return None

        goal_tokens = set(_tokenize(goal))
        best: WorkflowTemplate | None = None
        best_score = 0.0

        for t in templates:
            tmpl_tokens = set(_tokenize(t.goal_summary))
            if not tmpl_tokens:
                continue
            overlap = len(goal_tokens & tmpl_tokens) / len(goal_tokens | tmpl_tokens)
            if overlap > best_score:
                best_score = overlap
                best = t

        if best and best_score >= 0.2:
            return best
        return None

    # ------------------------------------------------------------------
    # Failure patterns
    # ------------------------------------------------------------------

    def record_failure(
        self,
        task_id: str,
        node_id: str,
        tool_or_capability: str,
        error_type: str,
        error_summary: str,
        scenario: str = "unknown",
    ) -> FailurePattern:
        """Persist a failure pattern for later analysis."""
        pattern_id = f"{task_id[:8]}_{node_id[:12]}"
        pattern = FailurePattern(
            pattern_id=pattern_id,
            task_id=task_id,
            node_id=node_id,
            tool_or_capability=tool_or_capability,
            error_type=error_type,
            error_summary=error_summary[:200],
            scenario=scenario,
        )
        path = self._failures_dir / f"{pattern_id}.json"
        path.write_text(
            json.dumps(pattern.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._update_index(
            "failure",
            pattern_id,
            {"tool": tool_or_capability, "error_type": error_type},
        )
        logger.debug(
            "AssetManager: recorded failure {} ({} on {})", pattern_id, error_type, node_id
        )
        return pattern

    def list_failures(
        self,
        tool_or_capability: str | None = None,
        error_type: str | None = None,
    ) -> list[FailurePattern]:
        """Return stored failure patterns, optionally filtered."""
        results: list[FailurePattern] = []
        for path in sorted(self._failures_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                p = FailurePattern.model_validate(data)
                if tool_or_capability and p.tool_or_capability != tool_or_capability:
                    continue
                if error_type and p.error_type != error_type:
                    continue
                results.append(p)
            except Exception as exc:
                logger.warning("AssetManager: could not load failure {}: {}", path.name, exc)
        return results

    def failure_count(self, tool_or_capability: str) -> int:
        """Return how many times a specific tool/capability has failed."""
        return len(self.list_failures(tool_or_capability=tool_or_capability))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_index(self, kind: str, asset_id: str, meta: dict[str, Any]) -> None:
        """Update the lightweight assets index file."""
        index: dict[str, Any] = {}
        if self._index_path.exists():
            try:
                index = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                index = {}

        index.setdefault(kind, {})[asset_id] = {
            **meta,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        try:
            self._index_path.write_text(
                json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("AssetManager: could not update index: {}", exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "for",
        "and",
        "or",
        "of",
        "in",
        "on",
        "to",
        "with",
        "what",
        "how",
        "is",
        "are",
        "was",
        "were",
        "find",
        "search",
        "give",
        "me",
        "us",
        "can",
        "do",
    ]
)


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alpha tokens, excluding stop words."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
