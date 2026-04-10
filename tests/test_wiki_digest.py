"""Tests for wiki digest materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from aeloon.plugins.Wiki.config import WikiConfig
from aeloon.plugins.Wiki.services.digest_service import DigestService
from aeloon.plugins.Wiki.services.ingest_service import IngestService
from aeloon.plugins.Wiki.services.manifest_service import ManifestService
from aeloon.plugins.Wiki.services.repo_service import RepoService


class _FakeFetchTool:
    async def execute(self, url: str, extractMode: str = "markdown") -> str:  # noqa: N803
        return '{"url":"https://example.com/path","text":"# Example\\n\\nA concise article body for testing."}'


class _FakeLLM:
    async def structured_output(self, messages: list[dict], schema: dict) -> dict:
        return {
            "summary": {
                "primary_domain": "research-automation",
                "domain_refs": ["agent-systems"],
                "title": "Example Digest",
                "summary": "A concise article body for testing.",
                "content": "## Notes\n\nGrounded summary content.",
                "links": ["https://example.com/path"],
                "depends_on": [],
                "derived_from": [],
            },
            "concepts": [
                {
                    "primary_domain": "agent-systems",
                    "domain_refs": ["research-automation"],
                    "title": "Example Concept",
                    "summary": "One concept summary.",
                    "content": "Concept details.",
                    "links": [],
                    "depends_on": [],
                    "derived_from": [],
                }
            ],
        }


@pytest.mark.asyncio
async def test_digest_pending_creates_summary_and_concept_pages(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    manifest = ManifestService(repo)
    ingest = IngestService(repo, manifest, WikiConfig(), fetch_tool=_FakeFetchTool())
    digest = DigestService(repo, manifest, ingest, _FakeLLM())  # type: ignore[arg-type]

    await ingest.ingest_url("https://example.com/path")
    results = await digest.digest_pending()

    assert len(results) == 1
    assert results[0].summary_artifact is not None
    assert (repo.layout.wiki_domains / "domain-research-automation.md").exists()
    assert (repo.layout.wiki_domains / "domain-agent-systems.md").exists()
    assert (repo.layout.wiki_summaries / "summary-example-digest.md").exists()
    assert (repo.layout.wiki_concepts / "concept-example-concept.md").exists()
    payload = manifest.load()
    assert payload["sources"][0]["status"] == "digested"


@pytest.mark.asyncio
async def test_digest_status_table_uses_repo_relative_paths(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    manifest = ManifestService(repo)
    ingest = IngestService(repo, manifest, WikiConfig(), fetch_tool=_FakeFetchTool())
    digest = DigestService(repo, manifest, ingest, _FakeLLM())  # type: ignore[arg-type]

    await ingest.ingest_url("https://example.com/path")
    results = await digest.digest_pending()
    table = digest.format_status_table(results)

    assert "| 文件 | 位置 | 一句话概括 |" in table
    assert "wiki/summaries/summary-example-digest.md" in table
    assert str(tmp_path) not in table


@pytest.mark.asyncio
async def test_digest_pending_discovers_manual_raw_files(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    repo.ensure_layout()
    raw_file = repo.layout.raw_files / "manual-note.md"
    raw_file.write_text("# Manual Note\n\nA manually copied markdown file.", encoding="utf-8")

    manifest = ManifestService(repo)
    ingest = IngestService(repo, manifest, WikiConfig(), fetch_tool=_FakeFetchTool())
    digest = DigestService(repo, manifest, ingest, _FakeLLM())  # type: ignore[arg-type]

    results = await digest.digest_pending()

    assert len(results) == 1
    assert results[0].source.raw_rel_path == "raw/files/manual-note.md"
    assert (repo.layout.raw_meta / "manual-note.json").exists()
    assert (repo.layout.wiki_summaries / "summary-example-digest.md").exists()
    assert manifest.pending_raw_paths() == []
