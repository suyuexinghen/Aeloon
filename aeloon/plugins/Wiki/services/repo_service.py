"""Repository layout service for the wiki plugin."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import WikiConfig
from ..models import RepoLayout, WikiStatus


class RepoService:
    """Resolve and manage the on-disk wiki root."""

    def __init__(self, storage_path: Path, config: WikiConfig) -> None:
        self._storage_path = storage_path
        self._config = config
        self._selection_state_path = self._storage_path / "repo_state.json"
        self._layout = self._build_layout(self._resolve_root())
        self._harness_template = (
            Path(__file__).resolve().parent.parent / "templates" / "WIKI_HARNESS.md"
        )

    @property
    def layout(self) -> RepoLayout:
        """Return the resolved repository layout."""
        return self._layout

    @property
    def repo_root(self) -> Path:
        """Return the repo root path."""
        return self._layout.root

    def is_initialized(self) -> bool:
        """Return whether the knowledge base exists on disk."""
        return (
            self.repo_root.exists()
            and self._layout.state_dir.exists()
            and self._layout.manifest_path.exists()
            and self._layout.harness_path.exists()
        )

    def ensure_layout(self) -> RepoLayout:
        """Backward-compatible explicit initializer for the repo layout."""
        return self.initialize()

    def initialize(self, repo_root: str | Path | None = None) -> RepoLayout:
        """Create a knowledge base at the chosen or configured path."""
        if repo_root is not None and str(repo_root).strip():
            root = Path(repo_root).expanduser().resolve()
            self._set_root(root)
        else:
            self._set_root(self.repo_root)

        for path in (
            self._layout.root,
            self._layout.state_dir,
            self._layout.raw_links,
            self._layout.raw_files,
            self._layout.raw_meta,
            self._layout.wiki_domains,
            self._layout.wiki_summaries,
            self._layout.wiki_concepts,
        ):
            path.mkdir(parents=True, exist_ok=True)

        if not self._layout.manifest_path.exists():
            self._layout.manifest_path.write_text(
                json.dumps({"sources": []}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self._layout.log_path.touch(exist_ok=True)
        if not self._layout.harness_path.exists():
            self._layout.harness_path.write_text(
                self._harness_template.read_text(encoding="utf-8").rstrip() + "\n",
                encoding="utf-8",
            )
        self._persist_root(self.repo_root)
        return self._layout

    def remove_knowledge_base(self) -> Path:
        """Delete the whole initialized knowledge base."""
        root = self.repo_root
        if root.exists():
            shutil.rmtree(root)
        if not self._config.repo_root and self._selection_state_path.exists():
            self._selection_state_path.unlink()
        self._layout = self._build_layout(self._resolve_root())
        return root

    def build_status(self) -> WikiStatus:
        """Build a lightweight status model for command responses."""
        notes = []
        initialized = self.is_initialized()
        if not initialized:
            notes.append("Knowledge base is not initialized.")
        return WikiStatus(
            repo_root=self.repo_root,
            initialized=initialized,
            raw_sources=self._count_files(self._layout.raw_links) + self._count_files(self._layout.raw_files),
            domains=self._count_files(self._layout.wiki_domains),
            summaries=self._count_files(self._layout.wiki_summaries),
            concepts=self._count_files(self._layout.wiki_concepts),
            notes=notes,
        )

    def relative_path(self, path: Path) -> str:
        """Return a repo-relative path string."""
        return str(path.relative_to(self.repo_root))

    def _resolve_root(self) -> Path:
        if self._config.repo_root:
            return Path(self._config.repo_root).expanduser().resolve()
        if self._selection_state_path.exists():
            try:
                payload = json.loads(self._selection_state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            stored = str(payload.get("repo_root", "")).strip()
            if stored:
                return Path(stored).expanduser().resolve()
        return (self._storage_path / "repo").resolve()

    def _set_root(self, root: Path) -> None:
        self._layout = self._build_layout(root)

    def _persist_root(self, root: Path) -> None:
        if self._config.repo_root:
            return
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._selection_state_path.write_text(
            json.dumps({"repo_root": str(root)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _build_layout(self, root: Path) -> RepoLayout:
        return RepoLayout(
            root=root,
            state_dir=root / "state",
            manifest_path=root / "state" / "manifest.json",
            log_path=root / "state" / "log.jsonl",
            harness_path=root / "WIKI_HARNESS.md",
            raw_links=root / "raw" / "links",
            raw_files=root / "raw" / "files",
            raw_meta=root / "raw" / "meta",
            wiki_domains=root / "wiki" / "domains",
            wiki_summaries=root / "wiki" / "summaries",
            wiki_concepts=root / "wiki" / "concepts",
        )

    def _count_files(self, path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for item in path.iterdir() if item.is_file())
