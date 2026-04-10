"""Tests for ArxivTool: search, abstract, fetch, URL normalization, and WebFetchTool integration."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from aeloon.plugins.ScienceResearch.config import ArxivConfig
from aeloon.plugins.ScienceResearch.tools.arxiv import (
    ArxivTool,
    _is_arxiv_url,
    _normalize_arxiv_id,
)

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


class TestIsArxivUrl:
    def test_abs_url(self) -> None:
        assert _is_arxiv_url("https://arxiv.org/abs/2401.12345")

    def test_pdf_url(self) -> None:
        assert _is_arxiv_url("https://arxiv.org/pdf/2401.12345")

    def test_www_prefix(self) -> None:
        assert _is_arxiv_url("https://www.arxiv.org/abs/2401.12345")

    def test_non_arxiv(self) -> None:
        assert not _is_arxiv_url("https://nature.com/articles/123")

    def test_empty(self) -> None:
        assert not _is_arxiv_url("")

    def test_garbage(self) -> None:
        assert not _is_arxiv_url("not a url at all")


class TestNormalizeArxivId:
    def test_bare_id(self) -> None:
        assert _normalize_arxiv_id("2401.12345") == "2401.12345"

    def test_abs_url(self) -> None:
        assert _normalize_arxiv_id("https://arxiv.org/abs/2401.12345") == "2401.12345"

    def test_pdf_url(self) -> None:
        assert _normalize_arxiv_id("https://arxiv.org/pdf/2401.12345") == "2401.12345"

    def test_version_suffix(self) -> None:
        assert _normalize_arxiv_id("2401.12345v2") == "2401.12345"

    def test_arxiv_prefix(self) -> None:
        assert _normalize_arxiv_id("arxiv:2401.12345") == "2401.12345"

    def test_pdf_extension(self) -> None:
        assert _normalize_arxiv_id("2401.12345.pdf") == "2401.12345"

    def test_old_style_id(self) -> None:
        assert _normalize_arxiv_id("hep-th/9901001") == "hep-th/9901001"

    def test_www_prefix(self) -> None:
        assert _normalize_arxiv_id("https://www.arxiv.org/abs/2401.12345") == "2401.12345"


# ---------------------------------------------------------------------------
# ArxivTool — search
# ---------------------------------------------------------------------------

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345</id>
    <title>Perovskite Solar Cells: A Review</title>
    <summary>A comprehensive review of perovskite solar cells.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cond-mat.mtrl-sci"/>
    <published>2024-01-15T00:00:00Z</published>
    <updated>2024-02-01T00:00:00Z</updated>
    <link title="pdf" href="https://arxiv.org/pdf/2401.12345" rel="related" type="application/pdf"/>
  </entry>
</feed>"""


@pytest.mark.asyncio
async def test_search_parses_atom_results() -> None:
    tool = ArxivTool(config=ArxivConfig())

    async def _mock_get(self_client, url, **kwargs):
        r = MagicMock()
        r.text = SAMPLE_ATOM_XML
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.AsyncClient.get", _mock_get):
        result = await tool.execute(action="search", query="perovskite solar cells")

    data = json.loads(result)
    assert data["count"] == 1
    paper = data["papers"][0]
    assert paper["arxiv_id"] == "2401.12345"
    assert paper["title"] == "Perovskite Solar Cells: A Review"
    assert paper["authors"] == ["Alice Smith", "Bob Jones"]
    assert paper["categories"] == ["cond-mat.mtrl-sci"]


@pytest.mark.asyncio
async def test_search_empty_query_returns_error() -> None:
    tool = ArxivTool(config=ArxivConfig())
    result = await tool.execute(action="search", query="")
    data = json.loads(result)
    assert "error" in data


# ---------------------------------------------------------------------------
# ArxivTool — abstract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abstract_returns_paper_metadata() -> None:
    tool = ArxivTool(config=ArxivConfig())

    async def _mock_get(self_client, url, **kwargs):
        r = MagicMock()
        r.text = SAMPLE_ATOM_XML
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.AsyncClient.get", _mock_get):
        result = await tool.execute(action="abstract", paper_id="2401.12345")

    data = json.loads(result)
    assert data["paper_id"] == "2401.12345"
    assert data["extractor"] == "arxiv_api"
    assert data["title"] == "Perovskite Solar Cells: A Review"
    assert "Alice Smith" in data["authors"]


@pytest.mark.asyncio
async def test_abstract_accepts_url() -> None:
    tool = ArxivTool(config=ArxivConfig())

    async def _mock_get(self_client, url, **kwargs):
        r = MagicMock()
        r.text = SAMPLE_ATOM_XML
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.AsyncClient.get", _mock_get):
        result = await tool.execute(action="abstract", url="https://arxiv.org/abs/2401.12345")

    data = json.loads(result)
    assert data["paper_id"] == "2401.12345"


@pytest.mark.asyncio
async def test_abstract_missing_id_returns_error() -> None:
    tool = ArxivTool(config=ArxivConfig())
    result = await tool.execute(action="abstract")
    data = json.loads(result)
    assert "error" in data


# ---------------------------------------------------------------------------
# ArxivTool — fetch (full text)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_text_from_pdf() -> None:
    config = ArxivConfig(cache_dir="/tmp/test_arxiv_cache")
    tool = ArxivTool(config=config)

    call_count = 0

    async def _mock_get(self_client, url, **kwargs):
        nonlocal call_count
        call_count += 1
        r = MagicMock()
        if "api/query" in url or "export.arxiv.org" in url:
            r.text = SAMPLE_ATOM_XML
        else:
            # PDF download
            r.content = b"fake pdf bytes"
        r.raise_for_status = MagicMock()
        r.status_code = 200
        return r

    with (
        patch("httpx.AsyncClient.get", _mock_get),
        patch.object(tool, "_extract_pdf_text", return_value="Full paper text here."),
    ):
        result = await tool.execute(action="fetch", paper_id="2401.12345")

    data = json.loads(result)
    assert data["extractor"] == "arxiv_pdf"
    assert "Full paper text here" in data["text"]
    assert data["paper_id"] == "2401.12345"


@pytest.mark.asyncio
async def test_fetch_abstract_fallback_on_pdf_failure() -> None:
    config = ArxivConfig(cache_dir="/tmp/test_arxiv_cache_fb")
    tool = ArxivTool(config=config)

    async def _mock_get(self_client, url, **kwargs):
        r = MagicMock()
        if "api/query" in url or "export.arxiv.org" in url:
            r.text = SAMPLE_ATOM_XML
        else:
            raise httpx.TimeoutException("pdf timeout")
        r.raise_for_status = MagicMock()
        return r

    with (
        patch("httpx.AsyncClient.get", _mock_get),
    ):
        result = await tool.execute(action="fetch", paper_id="2401.12345")

    data = json.loads(result)
    assert data["extractor"] == "arxiv_api_fallback"
    assert "warning" in data


# ---------------------------------------------------------------------------
# Tool interface compliance
# ---------------------------------------------------------------------------


def test_arxiv_tool_schema() -> None:
    tool = ArxivTool(config=ArxivConfig())
    schema = tool.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "arxiv"
    assert "action" in schema["function"]["parameters"]["properties"]


def test_arxiv_tool_concurrency_mode() -> None:
    tool = ArxivTool(config=ArxivConfig())
    assert tool.concurrency_mode == "read_only"


def test_arxiv_tool_validates_required_action() -> None:
    tool = ArxivTool(config=ArxivConfig())
    errors = tool.validate_params({})
    assert any("action" in e for e in errors)
