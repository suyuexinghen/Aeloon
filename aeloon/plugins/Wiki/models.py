"""Shared models for the wiki plugin."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class WikiPageType(StrEnum):
    """Supported wiki page types."""

    DOMAIN = "domain"
    SUMMARY = "summary"
    CONCEPT = "concept"


@dataclass(slots=True)
class RepoLayout:
    """Resolved repository layout under the plugin repo root."""

    root: Path
    state_dir: Path
    manifest_path: Path
    log_path: Path
    harness_path: Path
    raw_links: Path
    raw_files: Path
    raw_meta: Path
    wiki_domains: Path
    wiki_summaries: Path
    wiki_concepts: Path


@dataclass(slots=True)
class WikiStatus:
    """Basic runtime status for user-facing command output."""

    repo_root: Path
    initialized: bool
    raw_sources: int
    domains: int
    summaries: int
    concepts: int
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IngestedSource:
    """A raw source captured into the repo."""

    kind: str
    display_name: str
    raw_path: Path
    raw_rel_path: str
    meta_path: Path
    meta_rel_path: str
    duplicate: bool = False
    source_url: str = ""
    original_name: str = ""
    file_hash: str = ""
    title: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """Serialize to metadata written into the repo."""
        data = asdict(self)
        data["raw_path"] = str(self.raw_path)
        data["meta_path"] = str(self.meta_path)
        return data


@dataclass(slots=True)
class DigestArtifact:
    """A written wiki page artifact."""

    page_type: WikiPageType
    path: Path
    rel_path: str
    title: str
    summary: str


@dataclass(slots=True)
class DigestResult:
    """Digest output for one ingested source."""

    source: IngestedSource
    artifacts: list[DigestArtifact]
    summary_artifact: DigestArtifact | None


@dataclass(slots=True)
class EvidenceItem:
    """One wiki page selected as grounding evidence."""

    entry_id: str
    title: str
    rel_path: str
    summary: str
    score: int
    snippets: list[str]


@dataclass(slots=True)
class RelatedEntryOption:
    """One related wiki entry offered for follow-up exploration."""

    entry_id: str
    title: str
    rel_path: str
    summary: str
    score: int


@dataclass(slots=True)
class QueryResult:
    """Query retrieval output split into primary and related entries."""

    primary_evidence: list[EvidenceItem]
    related_entries: list[RelatedEntryOption]
