"""Tests for wiki raw ingest and repo setup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeloon.plugins.Wiki.config import WikiConfig
from aeloon.plugins.Wiki.services.ingest_service import IngestService
from aeloon.plugins.Wiki.services.manifest_service import ManifestService
from aeloon.plugins.Wiki.services.repo_service import RepoService


class _FakeFetchTool:
    async def execute(self, url: str, extractMode: str = "markdown") -> str:  # noqa: N803
        return json.dumps(
            {
                "url": url,
                "text": "# Example Article\n\nA concise article body for testing.",
            },
            ensure_ascii=False,
        )


@pytest.mark.asyncio
async def test_ingest_url_writes_raw_markdown_and_metadata(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    ingest = IngestService(repo, ManifestService(repo), WikiConfig(), fetch_tool=_FakeFetchTool())

    source = await ingest.ingest_url("https://example.com/path")

    assert source.raw_path.exists()
    assert source.meta_path.exists()
    assert source.raw_rel_path.startswith("raw/links/")
    meta = json.loads(source.meta_path.read_text(encoding="utf-8"))
    assert meta["source_url"] == "https://example.com/path"


@pytest.mark.asyncio
async def test_ingest_file_copies_source_into_raw_files(tmp_path: Path) -> None:
    source_file = tmp_path / "paper.txt"
    source_file.write_text("A local research note.", encoding="utf-8")

    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    ingest = IngestService(repo, ManifestService(repo), WikiConfig())

    source = await ingest.ingest_file(source_file)

    assert source.raw_path.exists()
    assert source.raw_rel_path.startswith("raw/files/")
    assert source.raw_path.read_text(encoding="utf-8") == "A local research note."


@pytest.mark.asyncio
async def test_duplicate_url_ingest_reuses_existing_source(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    ingest = IngestService(repo, ManifestService(repo), WikiConfig(), fetch_tool=_FakeFetchTool())

    first = await ingest.ingest_url("https://example.com/path")
    second = await ingest.ingest_url("https://example.com/path")

    assert second.duplicate is True
    assert first.raw_path == second.raw_path
    assert len(list(repo.layout.raw_links.glob("*.md"))) == 1


def test_initialize_is_idempotent(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())

    repo.initialize()
    repo.initialize()

    assert (repo.repo_root / "state" / "manifest.json").exists()
    assert (repo.repo_root / "WIKI_HARNESS.md").exists()


def test_discover_unmanaged_repo_files_creates_metadata(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    repo.ensure_layout()
    raw_file = repo.layout.raw_files / "manual-note.md"
    raw_file.write_text("# Manual Note\n\nA manually copied markdown file.", encoding="utf-8")

    ingest = IngestService(repo, ManifestService(repo), WikiConfig())

    discovered = ingest.discover_unmanaged_repo_files()

    assert len(discovered) == 1
    assert discovered[0].raw_path == raw_file
    assert discovered[0].meta_path.exists()
    meta = json.loads(discovered[0].meta_path.read_text(encoding="utf-8"))
    assert meta["raw_rel_path"] == "raw/files/manual-note.md"


def test_extract_source_urls_from_bibtex_like_text(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    ingest = IngestService(repo, ManifestService(repo), WikiConfig())

    text = """
@Article{Qin2024MooncakeAK,
 volume = {abs/2407.00079},
}

@misc{liu2025megascaleinferservingmixtureofexperts,
      eprint={2504.02263},
      url={https://arxiv.org/abs/2504.02263},
}
"""

    assert ingest.extract_source_urls(text) == [
        "https://arxiv.org/abs/2407.00079",
        "https://arxiv.org/abs/2504.02263",
    ]
