# Wiki Harness

This knowledge base is managed by the Wiki plugin.

Rules:

1. `raw/` contains source material. Treat it as input, not as the answer surface.
2. `wiki/summaries/` contains source-level summaries.
3. `wiki/concepts/` contains cross-source concepts.
4. `wiki/domains/` contains organizing domains used as the tree backbone.
5. Summary and concept pages should declare one `primary_domain` and may declare extra `domain_refs`.
6. `state/manifest.json` is the source of truth for tracked sources and derived pages.
7. Do not scan the filesystem on your own to discover new files.
8. Do not invent unmanaged files or directories.
9. Use wiki pages as the primary grounding layer for answers.
10. When the wiki lacks coverage, say so explicitly.
