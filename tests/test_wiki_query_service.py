"""Tests for wiki query retrieval and related-entry ranking."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aeloon.plugins.Wiki.config import WikiConfig
from aeloon.plugins.Wiki.services.query_service import QueryService
from aeloon.plugins.Wiki.services.repo_service import RepoService


def test_search_returns_primary_evidence_and_related_entries(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    repo.ensure_layout()

    (repo.layout.wiki_domains / "domain-agent-systems.md").write_text(
        "---\n"
        "id: domain-agent-systems\n"
        "type: domain\n"
        'title: "Agent Systems"\n'
        'summary: "Domain grouping."\n'
        "member_refs:\n"
        "---\n"
        "Agent Systems groups related entries.\n",
        encoding="utf-8",
    )
    (repo.layout.wiki_concepts / "concept-agent-systems.md").write_text(
        "---\n"
        'id: concept-agent-systems\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        'title: "Agent Systems"\n'
        'summary: "Core architecture patterns for agent systems."\n'
        "depends_on:\n"
        "derived_from:\n"
        "links:\n"
        "---\n"
        "Agent systems coordinate tools, planning, and memory.\n",
        encoding="utf-8",
    )
    (repo.layout.wiki_concepts / "concept-tool-execution.md").write_text(
        "---\n"
        'id: concept-tool-execution\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        'title: "Tool Execution"\n'
        'summary: "How agents invoke tools safely."\n'
        "depends_on:\n"
        "derived_from:\n"
        "links:\n"
        "---\n"
        "Tool execution is a key concept in agent systems.\n",
        encoding="utf-8",
    )
    (repo.layout.wiki_summaries / "summary-agent-overview.md").write_text(
        "---\n"
        'id: summary-agent-overview\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        'title: "Agent Overview"\n'
        'summary: "Overview of agent systems and tool execution."\n'
        "depends_on:\n"
        "  - concept-agent-systems\n"
        "derived_from:\n"
        "links:\n"
        "  - concept-tool-execution\n"
        "---\n"
        "This summary explains agent systems and tool execution.\n",
        encoding="utf-8",
    )
    (repo.layout.wiki_concepts / "concept-random-agent-notes.md").write_text(
        "---\n"
        'id: concept-random-agent-notes\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        'title: "Random Agent Notes"\n'
        'summary: "Loose notes mentioning agents."\n'
        "depends_on:\n"
        "derived_from:\n"
        "links:\n"
        "---\n"
        "Miscellaneous agent notes.\n",
        encoding="utf-8",
    )

    result = asyncio.run(QueryService(repo).search("Explain agent systems"))

    assert result.primary_evidence
    assert result.primary_evidence[0].entry_id == "summary-agent-overview"
    assert [item.entry_id for item in result.related_entries][:2] == [
        "concept-agent-systems",
        "concept-tool-execution",
    ]
    assert all(item.entry_id != "summary-agent-overview" for item in result.related_entries)


def test_get_entry_resolves_by_entry_id(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    repo.ensure_layout()

    (repo.layout.wiki_concepts / "concept-agent-systems.md").write_text(
        "---\n"
        'id: concept-agent-systems\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        'title: "Agent Systems"\n'
        'summary: "Core architecture patterns for agent systems."\n'
        "depends_on:\n"
        "derived_from:\n"
        "links:\n"
        "---\n"
        "Agent systems coordinate tools, planning, and memory.\n",
        encoding="utf-8",
    )

    item = QueryService(repo).get_entry("concept-agent-systems")

    assert item is not None
    assert item.title == "Agent Systems"
    assert item.rel_path == "wiki/concepts/concept-agent-systems.md"


def test_format_map_renders_mermaid_graph(tmp_path: Path) -> None:
    repo = RepoService(tmp_path / "aeloon" / "wiki", WikiConfig())
    repo.ensure_layout()

    (repo.layout.wiki_domains / "domain-agent-systems.md").write_text(
        "---\n"
        "id: domain-agent-systems\n"
        "type: domain\n"
        'title: "Agent Systems"\n'
        'summary: "Domain grouping."\n'
        "member_refs:\n"
        "---\n"
        "Agent Systems groups related entries.\n",
        encoding="utf-8",
    )
    (repo.layout.wiki_domains / "domain-research-automation.md").write_text(
        "---\n"
        "id: domain-research-automation\n"
        "type: domain\n"
        'title: "Research Automation"\n'
        'summary: "Domain grouping."\n'
        "member_refs:\n"
        "---\n"
        "Research Automation groups related entries.\n",
        encoding="utf-8",
    )
    (repo.layout.wiki_concepts / "concept-agent-systems.md").write_text(
        "---\n"
        'id: concept-agent-systems\n'
        'type: concept\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        "  - domain-research-automation\n"
        'title: "Agent Systems"\n'
        'summary: "Core architecture patterns for agent systems."\n'
        "depends_on:\n"
        "derived_from:\n"
        "links:\n"
        "---\n"
        "Agent systems coordinate tools.\n",
        encoding="utf-8",
    )
    (repo.layout.wiki_summaries / "summary-agent-overview.md").write_text(
        "---\n"
        'id: summary-agent-overview\n'
        'type: summary\n'
        "primary_domain: domain-agent-systems\n"
        "domain_refs:\n"
        'title: "Agent Overview"\n'
        'summary: "Overview of agent systems."\n'
        "depends_on:\n"
        "  - concept-agent-systems\n"
        "derived_from:\n"
        "links:\n"
        "---\n"
        "This summary explains agent systems.\n",
        encoding="utf-8",
    )

    mermaid = QueryService(repo).format_map()

    assert mermaid.startswith("```mermaid")
    assert 'root["Wiki"]' in mermaid
    assert "domain-agent-systems" in mermaid
    assert "domain-research-automation" in mermaid
    assert "summary-agent-overview" in mermaid
    assert "concept-agent-systems" in mermaid
    assert "--> n_domain_agent_systems" in mermaid
    assert "-.->|domain_ref| n_concept_agent_systems" in mermaid
