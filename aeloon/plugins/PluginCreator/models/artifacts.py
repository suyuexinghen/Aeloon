"""Artifact models — specification and reference types for workflow artifacts.

Artifacts are the tangible outputs produced and consumed by workflow phases.
They form an implicit chain via producer/consumer links on :class:`ArtifactSpec`.
"""

from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """Kinds of artifacts in the plugin development workflow."""

    ANALYSIS = "analysis"
    DESIGN = "design"
    PLAN = "plan"
    CODE = "code"
    VALIDATION = "validation"
    DELIVERY = "delivery"
    RESUME = "resume"
    REVIEW = "review"
    HISTORY = "history"


class ArtifactStatus(str, Enum):
    """Lifecycle status of a single artifact (Domain 3)."""

    PLANNED = "planned"
    PARTIAL = "partial"
    VALIDATED = "validated"
    MISSING = "missing"
    SUPERSEDED = "superseded"


class ArtifactRef(BaseModel):
    """Lightweight reference to an artifact by ID."""

    artifact_id: str
    role: str = ""


class EvidenceRef(BaseModel):
    """Reference to evidence used for verification or validation."""

    source: str
    description: str = ""


class ArtifactSpec(BaseModel):
    """Specification of a single workflow artifact.

    Artifacts are linked into chains via ``producer`` and ``consumers`` fields.
    """

    id: str = Field(default_factory=lambda: f"art_{uuid4().hex[:12]}")
    type: ArtifactType
    producer: str
    consumers: list[str] = Field(default_factory=list)
    contract: list[str] = Field(default_factory=list)
    path_hint: str | None = None
    format: str = "json"
    phase_id: str | None = None
    required_for_resume: bool = False
    required_for_review: bool = False
    required_for_delivery: bool = False
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    status: ArtifactStatus = ArtifactStatus.PLANNED
