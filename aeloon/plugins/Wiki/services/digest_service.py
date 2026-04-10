"""Digest raw sources into wiki pages."""

from __future__ import annotations

import re
from typing import Any

from aeloon.plugins._sdk.runtime import PluginLLMProxy

from ..models import DigestArtifact, DigestResult, IngestedSource, WikiPageType
from ..processing.converter import ConverterFactory
from .ingest_service import IngestService
from .manifest_service import ManifestService
from .repo_service import RepoService


class DigestService:
    """Convert raw sources into wiki pages."""

    def __init__(
        self,
        repo_service: RepoService,
        manifest_service: ManifestService,
        ingest_service: IngestService,
        llm: PluginLLMProxy,
    ) -> None:
        self._repo_service = repo_service
        self._manifest_service = manifest_service
        self._ingest_service = ingest_service
        self._llm = llm
        self._converter_factory = ConverterFactory()

    async def digest_source(self, source: IngestedSource) -> DigestResult:
        """Digest one source into wiki pages."""
        raw_markdown = await self._load_markdown(source)
        compiled = await self._compile_payload(source, raw_markdown)

        summary_artifact = self._write_page(
            page_type=WikiPageType.SUMMARY,
            primary_domain=self._normalize_domain_ref(
                str(compiled["summary"].get("primary_domain", "")) or source.title or source.display_name
            ),
            domain_refs=self._normalize_domain_refs(compiled["summary"].get("domain_refs", [])),
            title=str(compiled["summary"]["title"]),
            summary=str(compiled["summary"]["summary"]),
            content=str(compiled["summary"]["content"]),
            sources=[source.raw_rel_path],
            links=list(compiled["summary"].get("links", [])),
            depends_on=list(compiled["summary"].get("depends_on", [])),
            derived_from=list(compiled["summary"].get("derived_from", [])),
        )

        artifacts = [summary_artifact]
        concept_paths: list[str] = []
        for item in compiled.get("concepts", []):
            artifact = self._write_page(
                page_type=WikiPageType.CONCEPT,
                primary_domain=self._normalize_domain_ref(
                    str(item.get("primary_domain", "")) or str(compiled["summary"].get("primary_domain", ""))
                    or source.title
                    or source.display_name
                ),
                domain_refs=self._normalize_domain_refs(item.get("domain_refs", [])),
                title=str(item["title"]),
                summary=str(item["summary"]),
                content=str(item["content"]),
                sources=[summary_artifact.rel_path],
                links=list(item.get("links", [])),
                depends_on=list(item.get("depends_on", [])),
                derived_from=list(item.get("derived_from", [])),
            )
            artifacts.append(artifact)
            concept_paths.append(artifact.rel_path)

        self._sync_domain_pages(artifacts)
        self._manifest_service.mark_digested(
            source,
            summary_page=summary_artifact.rel_path,
            concept_pages=concept_paths,
        )
        return DigestResult(source=source, artifacts=artifacts, summary_artifact=summary_artifact)

    async def digest_pending(self) -> list[DigestResult]:
        """Digest all pending manifest-tracked sources."""
        self._ingest_service.discover_unmanaged_repo_files()
        pending_paths = set(self._manifest_service.pending_raw_paths())
        if not pending_paths:
            return []

        indexed_sources = {
            source.raw_rel_path: source
            for source in self._ingest_service.list_sources()
        }
        results: list[DigestResult] = []
        for raw_rel_path in sorted(pending_paths):
            source = indexed_sources.get(raw_rel_path)
            if source is None:
                continue
            results.append(await self.digest_source(source))
        return results

    def format_status_table(self, results: list[DigestResult]) -> str:
        """Render digest results as a markdown table."""
        if not results:
            return "No pending raw sources to digest."

        lines = [
            "| 文件 | 位置 | 一句话概括 |",
            "| --- | --- | --- |",
        ]
        for result in results:
            summary_artifact = result.summary_artifact
            location = summary_artifact.rel_path if summary_artifact else "-"
            summary = summary_artifact.summary if summary_artifact else "-"
            lines.append(f"| {result.source.display_name} | {location} | {summary} |")
        return "\n".join(lines)

    async def _load_markdown(self, source: IngestedSource) -> str:
        if source.kind == "url":
            return source.raw_path.read_text(encoding="utf-8")

        ext = source.raw_path.suffix.lstrip(".").lower()
        converter = self._converter_factory.get(ext)
        return await converter.convert(source.raw_path)

    async def _compile_payload(self, source: IngestedSource, raw_markdown: str) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "object",
                    "properties": {
                        "primary_domain": {"type": "string"},
                        "domain_refs": {"type": "array", "items": {"type": "string"}},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "content": {"type": "string"},
                        "links": {"type": "array", "items": {"type": "string"}},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "derived_from": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "summary", "content"],
                },
                "concepts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "primary_domain": {"type": "string"},
                            "domain_refs": {"type": "array", "items": {"type": "string"}},
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "content": {"type": "string"},
                            "links": {"type": "array", "items": {"type": "string"}},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                            "derived_from": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "summary", "content"],
                    },
                },
            },
            "required": ["summary"],
        }
        prompt = [
            {
                "role": "system",
                "content": (
                    "You maintain a constrained local wiki with only two page types: "
                    "`summary` and `concept`. "
                    "Use the source markdown to produce one summary page and zero or more concept pages. "
                    "Assign each page one `primary_domain` and zero or more `domain_refs`. "
                    "Domains should be short reusable buckets such as `agent-systems`, "
                    "`research-automation`, or `memory-tooling`. "
                    "Do not invent topic pages."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Source display name: {source.display_name}\n"
                    f"Source kind: {source.kind}\n"
                    f"Raw source path: {source.raw_rel_path}\n\n"
                    f"Raw markdown:\n{raw_markdown[:12000]}"
                ),
            },
        ]
        try:
            payload = await self._llm.structured_output(prompt, schema)
        except Exception:
            payload = {}
        if payload and isinstance(payload, dict) and payload.get("summary"):
            return payload
        fallback_summary = self._fallback_summary(source, raw_markdown)
        return {"summary": fallback_summary, "concepts": []}

    def _fallback_summary(self, source: IngestedSource, raw_markdown: str) -> dict[str, Any]:
        lines = [line.strip() for line in raw_markdown.splitlines() if line.strip()]
        title = source.title or source.display_name
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        summary = ""
        for line in lines:
            cleaned = re.sub(r"^#+\s*", "", line).strip()
            if cleaned and cleaned != title:
                summary = cleaned[:180]
                break
        if not summary:
            summary = title[:180]
        content = raw_markdown[:4000].strip()
        return {
            "primary_domain": self._normalize_domain_ref(source.title or source.display_name),
            "domain_refs": [],
            "title": title,
            "summary": summary,
            "content": content,
            "links": [],
            "depends_on": [],
            "derived_from": [],
        }

    def _write_page(
        self,
        *,
        page_type: WikiPageType,
        primary_domain: str,
        domain_refs: list[str],
        title: str,
        summary: str,
        content: str,
        sources: list[str],
        links: list[str],
        depends_on: list[str],
        derived_from: list[str],
    ) -> DigestArtifact:
        directory = {
            WikiPageType.SUMMARY: self._repo_service.layout.wiki_summaries,
            WikiPageType.CONCEPT: self._repo_service.layout.wiki_concepts,
        }[page_type]
        slug = self._slug(title)
        filename = f"{page_type.value}-{slug}.md"
        path = directory / filename
        rel_path = self._repo_service.relative_path(path)
        frontmatter = [
            "---",
            f"id: {page_type.value}-{slug}",
            f"type: {page_type.value}",
            f"primary_domain: {primary_domain}",
            "domain_refs:",
            *[f"  - {item}" for item in domain_refs],
            f"title: {self._yaml_quote(title)}",
            f"summary: {self._yaml_quote(summary)}",
            "sources:",
            *[f"  - {item}" for item in sources],
            "links:",
            *[f"  - {item}" for item in links],
            "depends_on:",
            *[f"  - {item}" for item in depends_on],
            "derived_from:",
            *[f"  - {item}" for item in derived_from],
            "---",
            "",
        ]
        path.write_text("\n".join(frontmatter) + content.strip() + "\n", encoding="utf-8")
        return DigestArtifact(
            page_type=page_type,
            path=path,
            rel_path=rel_path,
            title=title,
            summary=summary,
        )

    def _sync_domain_pages(self, artifacts: list[DigestArtifact]) -> None:
        grouped: dict[str, list[str]] = {}
        for artifact in artifacts:
            payload = self._read_frontmatter(artifact.path)
            domain_ids = [
                str(payload.get("primary_domain", "")).strip(),
                *self._frontmatter_list(artifact.path.read_text(encoding="utf-8"), "domain_refs"),
            ]
            for domain_id in domain_ids:
                if not domain_id:
                    continue
                grouped.setdefault(domain_id, [])
                if artifact.rel_path not in grouped[domain_id]:
                    grouped[domain_id].append(artifact.rel_path)

        for domain_id, member_refs in grouped.items():
            self._upsert_domain_page(domain_id, member_refs)

    def _upsert_domain_page(self, domain_id: str, member_refs: list[str]) -> None:
        slug = domain_id.removeprefix("domain-")
        path = self._repo_service.layout.wiki_domains / f"{domain_id}.md"
        existing_members: list[str] = []
        if path.exists():
            existing_members = self._frontmatter_list(path.read_text(encoding="utf-8"), "member_refs")
        merged_members = sorted({*existing_members, *member_refs})
        title = self._titleize_domain(slug)
        frontmatter = [
            "---",
            f"id: {domain_id}",
            f"type: {WikiPageType.DOMAIN.value}",
            f'title: "{self._yaml_escape(title)}"',
            f'summary: "{self._yaml_escape(f"Domain grouping for {title}.")}"',
            "member_refs:",
            *[f"  - {item}" for item in merged_members],
            "---",
            "",
        ]
        path.write_text(
            "\n".join(frontmatter) + f"{title} groups related wiki entries.\n",
            encoding="utf-8",
        )

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
        return slug or "item"

    def _normalize_domain_ref(self, value: str) -> str:
        slug = self._slug(value)
        if slug.startswith("domain-"):
            return slug
        return f"domain-{slug}"

    def _normalize_domain_refs(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            domain_ref = self._normalize_domain_ref(value)
            if domain_ref not in normalized:
                normalized.append(domain_ref)
        return normalized

    def _titleize_domain(self, slug: str) -> str:
        return slug.replace("-", " ").title()

    def _yaml_escape(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _yaml_quote(self, value: str) -> str:
        return '"' + self._yaml_escape(value) + '"'

    def _read_frontmatter(self, path: Any) -> dict[str, str]:
        text = path.read_text(encoding="utf-8")
        payload: dict[str, str] = {}
        in_frontmatter = False
        for line in text.splitlines():
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if not in_frontmatter:
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip().strip('"')
        return payload

    def _frontmatter_list(self, text: str, field: str) -> list[str]:
        in_frontmatter = False
        current_list = ""
        values: list[str] = []
        for line in text.splitlines():
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if not in_frontmatter:
                break
            if line.startswith(f"{field}:"):
                current_list = field
                continue
            if current_list == field and line.startswith("  - "):
                values.append(line[4:].strip())
                continue
            if current_list == field and not line.startswith("  "):
                current_list = ""
        return values
