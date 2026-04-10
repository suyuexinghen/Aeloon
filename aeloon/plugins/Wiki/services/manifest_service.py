"""Manifest-backed tracking for the wiki plugin."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..models import IngestedSource
from .repo_service import RepoService


class ManifestService:
    """Read and update the knowledge-base manifest."""

    def __init__(self, repo_service: RepoService) -> None:
        self._repo_service = repo_service

    def load(self) -> dict[str, Any]:
        """Load the current manifest payload."""
        if not self._repo_service.layout.manifest_path.exists():
            return {"sources": []}
        try:
            payload = json.loads(
                self._repo_service.layout.manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            payload = {}
        sources = payload.get("sources")
        return {"sources": sources if isinstance(sources, list) else []}

    def save(self, payload: dict[str, Any]) -> None:
        """Persist a manifest payload."""
        self._repo_service.layout.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def append_log(self, event: str, **fields: Any) -> None:
        """Append one compact JSONL event to the state log."""
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        with self._repo_service.layout.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def find_by_dedupe_key(self, dedupe_key: str) -> dict[str, Any] | None:
        """Return the first source record matching the dedupe key."""
        for source in self.load()["sources"]:
            if source.get("dedupe_key") == dedupe_key:
                return source
        return None

    def find_by_raw_rel_path(self, raw_rel_path: str) -> dict[str, Any] | None:
        """Return the first source record matching the raw path."""
        for source in self.load()["sources"]:
            if source.get("path") == raw_rel_path:
                return source
        return None

    def register_source(self, source: IngestedSource, *, dedupe_key: str) -> None:
        """Add or refresh one source record in the manifest."""
        payload = self.load()
        now = datetime.now(UTC).isoformat()
        sources = payload["sources"]
        existing = self.find_by_dedupe_key(dedupe_key)
        if existing is not None:
            existing.update(
                {
                    "path": source.raw_rel_path,
                    "display_name": source.display_name,
                    "updated_at": now,
                }
            )
        else:
            sources.append(
                {
                    "id": self._source_id(source),
                    "kind": source.kind,
                    "path": source.raw_rel_path,
                    "display_name": source.display_name,
                    "dedupe_key": dedupe_key,
                    "status": "raw",
                    "summary_page": "",
                    "concept_pages": [],
                    "created_at": now,
                    "updated_at": now,
                }
            )
        self.save(payload)
        self.append_log("source_registered", path=source.raw_rel_path, kind=source.kind)

    def pending_raw_paths(self) -> list[str]:
        """Return raw source paths still awaiting digest."""
        return [
            str(source.get("path", ""))
            for source in self.load()["sources"]
            if source.get("status") == "raw"
        ]

    def mark_digested(
        self,
        source: IngestedSource,
        *,
        summary_page: str,
        concept_pages: list[str],
    ) -> None:
        """Mark one source as digested and store derived page paths."""
        payload = self.load()
        now = datetime.now(UTC).isoformat()
        for item in payload["sources"]:
            if item.get("path") != source.raw_rel_path:
                continue
            item["status"] = "digested"
            item["summary_page"] = summary_page
            item["concept_pages"] = concept_pages
            item["updated_at"] = now
            break
        self.save(payload)
        self.append_log(
            "source_digested",
            path=source.raw_rel_path,
            summary_page=summary_page,
            concept_pages=concept_pages,
        )

    def _source_id(self, source: IngestedSource) -> str:
        suffix = source.file_hash[:12] if source.file_hash else source.raw_path.stem
        return f"{source.kind}:{suffix}"
