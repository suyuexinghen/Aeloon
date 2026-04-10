"""Aeloon Science Plugin -- AI4S science agent mode."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Task",
    "ScienceTaskGraph",
    "ScienceTaskNode",
    "Execution",
    "Validation",
    "SciencePipeline",
    "AssetManager",
]

from .assets import AssetManager
from .pipeline import SciencePipeline
from .task import (
    Execution,
    ScienceTaskGraph,
    ScienceTaskNode,
    Task,
    Validation,
)
