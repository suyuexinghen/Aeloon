"""Query-time evidence retrieval for the wiki plugin."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..models import EvidenceItem, QueryResult, RelatedEntryOption
from .repo_service import RepoService


@dataclass(slots=True)
class WikiPageRecord:
    """One parsed wiki page available for retrieval."""

    entry_id: str
    page_type: str
    primary_domain: str
    domain_refs: list[str]
    title: str
    summary: str
    rel_path: str
    text: str
    depends_on: list[str]
    derived_from: list[str]
    links: list[str]


class QueryService:
    """Search the wiki tree and format grounding blocks."""

    def __init__(self, repo_service: RepoService) -> None:
        self._repo_service = repo_service

    async def search(
        self,
        query: str,
        *,
        max_results: int = 1,
        max_related: int = 3,
    ) -> QueryResult:
        """Search wiki pages for primary evidence and related follow-ups."""
        terms = self._tokenize(query)
        if not terms:
            return QueryResult(primary_evidence=[], related_entries=[])

        pages = self._wiki_pages()
        scored: list[tuple[WikiPageRecord, EvidenceItem]] = []
        for page in pages.values():
            snippets = self._collect_snippets(page.text, terms)
            score = self._score_page(
                page.title,
                page.summary,
                page.text,
                terms,
                snippets,
                page.page_type,
            )
            if score <= 0:
                continue
            scored.append(
                (
                    page,
                    EvidenceItem(
                        entry_id=page.entry_id,
                        title=page.title,
                        rel_path=page.rel_path,
                        summary=page.summary,
                        score=score,
                        snippets=snippets[:2],
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[1].score, item[1].rel_path))
        primary = [item for _page, item in scored[:max_results]]
        related = self._collect_related_entries(
            terms=terms,
            pages=pages,
            primary_pages=[page for page, _item in scored[:max_results]],
            max_related=max_related,
        )
        return QueryResult(primary_evidence=primary, related_entries=related)

    def list_entries(self) -> list[WikiPageRecord]:
        """Return all wiki entries in stable order."""
        return sorted(
            self._wiki_pages().values(),
            key=lambda item: (item.page_type, item.rel_path),
        )

    def get_entry(self, entry_ref: str) -> WikiPageRecord | None:
        """Resolve one wiki entry by id, title, stem, or relative path."""
        return self._wiki_pages().get(self._normalize_ref(entry_ref))

    def format_map(self, entry_ref: str | None = None) -> str:
        """Render the wiki relation graph as a tree with references."""
        pages = self._wiki_pages()
        if not pages:
            return ""

        page_records = {page.entry_id: page for page in pages.values()}
        if entry_ref:
            target = pages.get(self._normalize_ref(entry_ref))
            if target is None:
                return ""
            included_ids = self._map_scope(target, page_records)
            selected = [page_records[entry_id] for entry_id in sorted(included_ids) if entry_id in page_records]
        else:
            selected = sorted(page_records.values(), key=lambda item: (item.page_type, item.rel_path))

        lines = ["```mermaid", "graph TD"]
        lines.append('    root["Wiki"]')
        seen_edges: set[tuple[str, str, str, str]] = set()
        selected_lookup = {page.entry_id: page for page in selected}

        for page in selected:
            node_id = self._mermaid_node_id(page.entry_id)
            lines.append(f'    {node_id}["{self._mermaid_label(page)}"]')

        domains = [page for page in selected if page.page_type == "domain"]
        for page in domains:
            lines.append(f"    root --> {self._mermaid_node_id(page.entry_id)}")

        for page in selected:
            if page.page_type == "domain":
                continue
            parent = selected_lookup.get(page.primary_domain)
            if parent is not None:
                lines.append(
                    f"    {self._mermaid_node_id(parent.entry_id)} --> {self._mermaid_node_id(page.entry_id)}"
                )
            for domain_ref in page.domain_refs:
                domain_page = selected_lookup.get(domain_ref)
                if domain_page is None:
                    continue
                edge = (domain_ref, page.entry_id, "domain_ref", "secondary")
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                lines.append(
                    f"    {self._mermaid_node_id(domain_page.entry_id)} -.->|domain_ref| {self._mermaid_node_id(page.entry_id)}"
                )

        for page in selected:
            if page.page_type == "domain":
                continue
            for relation, refs in (
                ("depends_on", page.depends_on),
                ("derived_from", page.derived_from),
                ("links", page.links),
            ):
                for ref in refs:
                    related = pages.get(self._normalize_ref(ref))
                    if related is None or related.entry_id not in selected_lookup or related.page_type == "domain":
                        continue
                    edge = (page.entry_id, related.entry_id, relation, "relation")
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                    source_id = self._mermaid_node_id(page.entry_id)
                    target_id = self._mermaid_node_id(related.entry_id)
                    lines.append(f"    {source_id} -.->|{relation}| {target_id}")

        lines.append("```")
        return "\n".join(lines)

    def format_evidence_block(
        self,
        query: str,
        evidence: list[EvidenceItem],
        related_entries: list[RelatedEntryOption],
    ) -> str:
        """Render evidence plus related-entry guidance into one prompt block."""
        lines = [
            "## Wiki Evidence",
            f"Query: {query}",
            "Use the evidence below as the primary grounding source.",
            "Answer the user's question first. Keep claims anchored to the cited wiki pages.",
            "",
        ]
        for index, item in enumerate(evidence, start=1):
            lines.append(f"[W{index}] {item.title} ({item.rel_path})")
            if item.summary:
                lines.append(f"Summary: {item.summary}")
            for snippet in item.snippets:
                lines.append(f"- {snippet}")
            lines.append("")

        if related_entries:
            lines.extend(
                [
                    "## Related Wiki Entries",
                    "If the main answer is grounded and complete, you may append one footer titled `Related wiki entries:`.",
                    "Use only the numbered options below. Do not invent extra options.",
                    "",
                ]
            )
            for index, item in enumerate(related_entries, start=1):
                summary = f" - {item.summary}" if item.summary else ""
                lines.append(f"{index}. {item.title} ({item.rel_path}){summary}")

        return "\n".join(lines).strip()

    def format_gap_block(self, query: str) -> str:
        """Render a no-evidence instruction block."""
        return (
            "## Wiki Coverage Gap\n"
            f"Query: {query}\n"
            "The local wiki does not contain strong evidence for this question. "
            "State that the wiki lacks coverage instead of filling the answer from model priors."
        )

    def _wiki_pages(self) -> dict[str, WikiPageRecord]:
        layout = self._repo_service.layout
        pages: dict[str, WikiPageRecord] = {}
        for directory in (
            layout.wiki_domains,
            layout.wiki_summaries,
            layout.wiki_concepts,
        ):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.md")):
                page = self._parse_page(path)
                for key in {
                    self._normalize_ref(page.entry_id),
                    self._normalize_ref(page.title),
                    self._normalize_ref(path.stem),
                    self._normalize_ref(page.rel_path),
                }:
                    if key:
                        pages[key] = page
        return {self._normalize_ref(page.entry_id): page for page in pages.values()}

    def _parse_page(self, path: Path) -> WikiPageRecord:
        text = path.read_text(encoding="utf-8")
        entry_id = self._frontmatter_field(text, "id") or path.stem
        title = self._frontmatter_field(text, "title") or path.stem
        summary = self._frontmatter_field(text, "summary")
        page_type = self._frontmatter_field(text, "type") or entry_id.split("-", 1)[0]
        return WikiPageRecord(
            entry_id=entry_id,
            page_type=page_type,
            primary_domain=(
                self._frontmatter_field(text, "primary_domain")
                or ("domain-general" if page_type in {"summary", "concept"} else "")
            ),
            domain_refs=self._frontmatter_list(text, "domain_refs"),
            title=title,
            summary=summary,
            rel_path=self._repo_service.relative_path(path),
            text=text,
            depends_on=self._frontmatter_list(text, "depends_on"),
            derived_from=self._frontmatter_list(text, "derived_from"),
            links=self._frontmatter_list(text, "links"),
        )

    def _collect_related_entries(
        self,
        *,
        terms: list[str],
        pages: dict[str, WikiPageRecord],
        primary_pages: list[WikiPageRecord],
        max_related: int,
    ) -> list[RelatedEntryOption]:
        primary_ids = {self._normalize_ref(page.entry_id) for page in primary_pages}
        ranked: dict[str, RelatedEntryOption] = {}

        for rank, page in enumerate(primary_pages, start=1):
            relation_bonus = max(0, 90 - rank * 5)
            for ref in self._page_refs(page):
                normalized = self._normalize_ref(ref)
                related = pages.get(normalized)
                if related is None or normalized in primary_ids or related.page_type == "domain":
                    continue
                self._update_related_candidate(
                    ranked,
                    related,
                    relation_bonus + self._lexical_score(related, terms),
                )

        for candidate in pages.values():
            normalized = self._normalize_ref(candidate.entry_id)
            if normalized in primary_ids or candidate.page_type == "domain":
                continue
            inbound_bonus = 0
            for primary in primary_pages:
                if self._normalize_ref(primary.entry_id) in {
                    self._normalize_ref(ref) for ref in self._page_refs(candidate)
                }:
                    inbound_bonus = max(inbound_bonus, 55)
            lexical_bonus = self._lexical_score(candidate, terms)
            total = max(inbound_bonus + lexical_bonus, lexical_bonus)
            if total > 0:
                self._update_related_candidate(ranked, candidate, total)

        ordered = sorted(ranked.values(), key=lambda item: (-item.score, item.rel_path))
        return ordered[:max_related]

    def _update_related_candidate(
        self,
        ranked: dict[str, RelatedEntryOption],
        page: WikiPageRecord,
        score: int,
    ) -> None:
        score += self._type_priority(page.page_type)
        current = ranked.get(page.entry_id)
        if current is None or score > current.score:
            ranked[page.entry_id] = RelatedEntryOption(
                entry_id=page.entry_id,
                title=page.title,
                rel_path=page.rel_path,
                summary=page.summary,
                score=score,
            )

    def _page_refs(self, page: WikiPageRecord) -> list[str]:
        refs = [*page.depends_on, *page.derived_from, *page.links]
        if page.primary_domain:
            refs.append(page.primary_domain)
        refs.extend(page.domain_refs)
        return refs

    def _type_priority(self, page_type: str) -> int:
        return {"concept": 2, "summary": 1, "domain": 0}.get(page_type, 0)

    def _mermaid_node_id(self, entry_id: str) -> str:
        return "n_" + re.sub(r"[^a-zA-Z0-9_]", "_", entry_id)

    def _mermaid_label(self, page: WikiPageRecord) -> str:
        return f"{page.title}\\n[{page.entry_id}]"

    def _map_scope(
        self,
        target: WikiPageRecord,
        page_records: dict[str, WikiPageRecord],
    ) -> set[str]:
        included_ids = {target.entry_id}
        for ref in self._page_refs(target):
            related = page_records.get(ref)
            if related is not None:
                included_ids.add(related.entry_id)
            else:
                normalized = self._normalize_ref(ref)
                for candidate in page_records.values():
                    if self._normalize_ref(candidate.entry_id) == normalized:
                        included_ids.add(candidate.entry_id)
                        break
        for candidate in page_records.values():
            if target.entry_id in self._page_refs(candidate):
                included_ids.add(candidate.entry_id)
        for entry_id in list(included_ids):
            page = page_records.get(entry_id)
            if page is None:
                continue
            if page.page_type == "domain":
                for candidate in page_records.values():
                    if candidate.primary_domain == page.entry_id or page.entry_id in candidate.domain_refs:
                        included_ids.add(candidate.entry_id)
            else:
                if page.primary_domain:
                    included_ids.add(page.primary_domain)
                included_ids.update(page.domain_refs)
        return included_ids

    def _lexical_score(self, page: WikiPageRecord, terms: list[str]) -> int:
        snippets = self._collect_snippets(page.text, terms)
        return self._score_page(
            page.title,
            page.summary,
            page.text,
            terms,
            snippets,
            page.page_type,
        )

    def _tokenize(self, query: str) -> list[str]:
        terms = re.findall(r"[a-z0-9]+", query.lower())
        stopwords = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "for", "with"}
        return [term for term in terms if len(term) > 1 and term not in stopwords]

    def _frontmatter_field(self, text: str, field: str) -> str:
        in_frontmatter = False
        for line in text.splitlines():
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if not in_frontmatter:
                if text.splitlines() and text.splitlines()[0].strip() != "---":
                    break
                continue
            if line.startswith(f"{field}:"):
                return line.split(":", 1)[1].strip().strip('"')
        return ""

    def _frontmatter_list(self, text: str, field: str) -> list[str]:
        in_frontmatter = False
        current_list = ""
        values: list[str] = []
        for line in text.splitlines():
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if not in_frontmatter:
                if text.splitlines() and text.splitlines()[0].strip() != "---":
                    break
                continue
            if line.startswith(f"{field}:"):
                current_list = field
                continue
            if current_list == field and line.startswith("  - "):
                values.append(line[4:].strip())
                continue
            if current_list == field and not line.startswith("  "):
                current_list = ""
        return values

    def _collect_snippets(self, text: str, terms: list[str]) -> list[str]:
        snippets: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped == "---":
                continue
            lowered = stripped.lower()
            if any(term in lowered for term in terms):
                snippets.append(stripped[:200])
        return snippets

    def _score_page(
        self,
        title: str,
        summary: str,
        text: str,
        terms: list[str],
        snippets: list[str],
        page_type: str,
    ) -> int:
        lowered_title = title.lower()
        lowered_summary = summary.lower()
        lowered_text = text.lower()
        score = 0
        for term in terms:
            if term in lowered_title:
                score += 6
            if term in lowered_summary:
                score += 4
            if term in lowered_text:
                score += 1
        score += min(len(snippets), 3)
        score += {
            "summary": 8,
            "topic": 1,
            "concept": 2,
        }.get(page_type, 0)
        return score

    def _normalize_ref(self, value: str) -> str:
        lowered = value.lower()
        lowered = re.sub(r"\.md$", "", lowered)
        return re.sub(r"[^a-z0-9]+", "", lowered)
