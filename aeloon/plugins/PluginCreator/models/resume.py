"""Resume model — compact continuation unit for workflow resume."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ResumeBlock(BaseModel):
    """Compact snapshot enabling safe workflow resumption."""

    current_phase: str
    completed_artifacts: list[str] = Field(default_factory=list)
    open_blockers: list[str] = Field(default_factory=list)
    deferred_items: list[str] = Field(default_factory=list)
    next_safe_action: str
    next_prompt_suggestion: str
    resume_version: str = "1"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    latest_phase_summary: str | None = None
    active_gate_ids: list[str] = Field(default_factory=list)
    required_context_refs: list[str] = Field(default_factory=list)
    open_decision_point_ids: list[str] = Field(default_factory=list)
    resume_preconditions: list[str] = Field(default_factory=list)
