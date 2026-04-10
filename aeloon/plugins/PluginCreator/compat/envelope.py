"""Compatibility envelope — bridges v1 legacy plan path to v3 kernel.

Sprint 1: define types only.  Active routing will be added in a later sprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CompatibilityMode(str, Enum):
    """Compatibility routing mode."""

    NONE = "none"
    V1_LEGACY = "v1_legacy"


@dataclass
class CompatibilityEnvelope:
    """Wraps plan routing for backward compatibility."""

    mode: CompatibilityMode = CompatibilityMode.NONE
    notes: list[str] = field(default_factory=list)
    command_routing: dict[str, str] = field(default_factory=dict)
