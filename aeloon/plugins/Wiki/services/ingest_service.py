"""Raw ingest service for the wiki plugin."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aeloon.core.agent.tools.web import WebFetchTool

from ..config import WikiConfig
from ..models import IngestedSource
from .manifest_service import ManifestService
from .repo_service import RepoService


class IngestService:
    """Capture URLs and local files into the raw repo."""

    def __init__(
        self,
        repo_service: RepoService,
        manifest_service: ManifestService,
        config: WikiConfig,
        *,
        fetch_tool: Any | None = None,
    ) -> None:
        self._repo_service = repo_service
        self._manifest_service = manifest_service
        self._config = config
        self._fetch_tool = fetch_tool or WebFetchTool()

    async def ingest_input(self, raw_input: str | Path) -> list[IngestedSource]:
        """Ingest one file path or many URL-like references from free-form text."""
        if isinstance(raw_input, Path):
            return [await self.ingest_file(raw_input)]

        text = raw_input.strip()
        if not text:
            raise ValueError("Input cannot be empty")

        candidate = Path(text).expanduser()
        if "\n" not in text and candidate.exists() and candidate.is_file():
            return [await self.ingest_file(candidate)]

        urls = self.extract_source_urls(text)
        if not urls:
            raise ValueError("No URL or arXiv reference found in input")

        ingested: list[IngestedSource] = []
        for url in urls:
            ingested.append(await self.ingest_url(url))
        return ingested

    def extract_source_urls(self, text: str) -> list[str]:
        """Extract URL and arXiv references from free-form text or BibTeX."""
        candidates: list[tuple[int, str]] = []
        seen: set[str] = set()

        def _record(position: int, url: str) -> None:
            normalized = self._normalize_extracted_url(url)
            if normalized:
                candidates.append((position, normalized))

        for match in re.finditer(r"https?://[^\s<>{}\"']+", text, flags=re.IGNORECASE):
            _record(match.start(), match.group(0))

        for pattern in (
            r"\beprint\s*=\s*[{\"\s]*([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
            r"\bvolume\s*=\s*[{\"\s]*abs/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
            r"\barxiv\s*[:=]\s*[{\"\s]*([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
        ):
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                _record(match.start(), f"https://arxiv.org/abs/{match.group(1)}")

        found: list[str] = []
        for _position, url in sorted(candidates, key=lambda item: item[0]):
            if url in seen:
                continue
            seen.add(url)
            found.append(url)
        return found

    async def ingest_url(self, url: str) -> IngestedSource:
        """Fetch a URL and store it as raw markdown plus metadata."""
        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("URL cannot be empty")

        existing = self._find_existing_url(normalized_url)
        if existing is not None:
            existing.duplicate = True
            return existing

        self._repo_service.initialize()
        fetch_result = await self._fetch_tool.execute(url=normalized_url, extractMode="markdown")
        payload = json.loads(fetch_result)
        if payload.get("error"):
            raise RuntimeError(payload["error"])

        raw_text = str(payload.get("text", "")).strip()
        if not raw_text:
            raise RuntimeError(f"No readable content extracted from {normalized_url}")

        title = self._extract_title(raw_text) or normalized_url
        filename = f"{self._slug_from_url(normalized_url)}-{self._short_hash(normalized_url)}.md"
        raw_path = self._repo_service.layout.raw_links / filename
        raw_rel_path = self._repo_service.relative_path(raw_path)
        raw_path.write_text(raw_text, encoding="utf-8")

        meta_path = self._repo_service.layout.raw_meta / f"{raw_path.stem}.json"
        meta_rel_path = self._repo_service.relative_path(meta_path)
        source = IngestedSource(
            kind="url",
            display_name=normalized_url,
            raw_path=raw_path,
            raw_rel_path=raw_rel_path,
            meta_path=meta_path,
            meta_rel_path=meta_rel_path,
            source_url=normalized_url,
            title=title,
        )
        self._write_metadata(source)
        self._manifest_service.register_source(
            source,
            dedupe_key=f"url:{normalized_url}",
        )
        return source

    async def ingest_file(self, source_path: str | Path) -> IngestedSource:
        """Copy a local file into `raw/files/` and write metadata."""
        source = Path(source_path).expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"File not found: {source}")

        ext = source.suffix.lstrip(".").lower()
        if ext not in self._config.supported_formats:
            raise ValueError(f"Unsupported format: .{ext}")

        file_hash = self._sha256_file(source)
        existing = self._find_existing_file(file_hash)
        if existing is not None:
            existing.duplicate = True
            return existing

        self._repo_service.initialize()
        target_name = f"{self._slug(source.stem)}-{file_hash[:8]}{source.suffix.lower()}"
        raw_path = self._repo_service.layout.raw_files / target_name
        shutil.copy2(source, raw_path)
        raw_rel_path = self._repo_service.relative_path(raw_path)

        meta_path = self._repo_service.layout.raw_meta / f"{raw_path.stem}.json"
        meta_rel_path = self._repo_service.relative_path(meta_path)
        ingested = IngestedSource(
            kind="file",
            display_name=source.name,
            raw_path=raw_path,
            raw_rel_path=raw_rel_path,
            meta_path=meta_path,
            meta_rel_path=meta_rel_path,
            original_name=source.name,
            file_hash=file_hash,
            title=source.stem,
        )
        self._write_metadata(ingested)
        self._manifest_service.register_source(
            ingested,
            dedupe_key=f"file:{file_hash}",
        )
        return ingested

    def list_sources(self) -> list[IngestedSource]:
        """Load all known raw source metadata."""
        if not self._repo_service.layout.raw_meta.exists():
            return []

        sources: list[IngestedSource] = []
        for meta_path in sorted(self._repo_service.layout.raw_meta.glob("*.json")):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            sources.append(
                IngestedSource(
                    kind=str(payload.get("kind", "")),
                    display_name=str(payload.get("display_name", "")),
                    raw_path=Path(str(payload.get("raw_path", ""))),
                    raw_rel_path=str(payload.get("raw_rel_path", "")),
                    meta_path=Path(str(payload.get("meta_path", ""))),
                    meta_rel_path=str(payload.get("meta_rel_path", "")),
                    source_url=str(payload.get("source_url", "")),
                    original_name=str(payload.get("original_name", "")),
                    file_hash=str(payload.get("file_hash", "")),
                    title=str(payload.get("title", "")),
                )
            )
        return sources

    def discover_unmanaged_repo_files(self) -> list[IngestedSource]:
        """Register manually added files under `raw/files/` that lack metadata."""
        self._repo_service.initialize()
        existing_sources = self.list_sources()
        known_raw_paths = {source.raw_rel_path for source in existing_sources if source.kind == "file"}

        discovered: list[IngestedSource] = []
        for raw_path in sorted(self._repo_service.layout.raw_files.glob("*")):
            if not raw_path.is_file():
                continue
            ext = raw_path.suffix.lstrip(".").lower()
            if ext not in self._config.supported_formats:
                continue

            raw_rel_path = self._repo_service.relative_path(raw_path)
            if raw_rel_path in known_raw_paths:
                continue

            title = raw_path.stem
            if ext in {"md", "txt"}:
                try:
                    title = self._extract_title(raw_path.read_text(encoding="utf-8")) or raw_path.stem
                except OSError:
                    title = raw_path.stem

            meta_path = self._repo_service.layout.raw_meta / f"{raw_path.stem}.json"
            source = IngestedSource(
                kind="file",
                display_name=raw_path.name,
                raw_path=raw_path,
                raw_rel_path=raw_rel_path,
                meta_path=meta_path,
                meta_rel_path=self._repo_service.relative_path(meta_path),
                original_name=raw_path.name,
                file_hash=self._sha256_file(raw_path),
                title=title,
            )
            self._write_metadata(source)
            self._manifest_service.register_source(
                source,
                dedupe_key=f"file:{source.file_hash}",
            )
            discovered.append(source)
            known_raw_paths.add(raw_rel_path)

        return discovered

    def _write_metadata(self, source: IngestedSource) -> None:
        payload = {
            "kind": source.kind,
            "display_name": source.display_name,
            "raw_path": str(source.raw_path),
            "raw_rel_path": source.raw_rel_path,
            "meta_path": str(source.meta_path),
            "meta_rel_path": source.meta_rel_path,
            "source_url": source.source_url,
            "original_name": source.original_name,
            "file_hash": source.file_hash,
            "title": source.title,
        }
        source.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _find_existing_url(self, url: str) -> IngestedSource | None:
        manifest_match = self._manifest_service.find_by_dedupe_key(f"url:{url}")
        if manifest_match is not None:
            for source in self.list_sources():
                if source.raw_rel_path == str(manifest_match.get("path", "")):
                    return source
        for source in self.list_sources():
            if source.kind == "url" and source.source_url == url:
                return source
        return None

    def _normalize_extracted_url(self, url: str) -> str:
        return url.strip().rstrip(",.;:)}]>'\"")

    def _find_existing_file(self, file_hash: str) -> IngestedSource | None:
        manifest_match = self._manifest_service.find_by_dedupe_key(f"file:{file_hash}")
        if manifest_match is not None:
            for source in self.list_sources():
                if source.raw_rel_path == str(manifest_match.get("path", "")):
                    return source
        for source in self.list_sources():
            if source.kind == "file" and source.file_hash == file_hash:
                return source
        return None

    def _extract_title(self, markdown: str) -> str:
        for line in markdown.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return ""

    def _slug_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        host = self._slug(parsed.netloc or "link")
        path = self._slug(parsed.path.strip("/")) or "index"
        return f"{host}-{path}".strip("-")

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
        return slug or "item"

    def _short_hash(self, value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
