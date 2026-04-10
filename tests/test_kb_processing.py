"""Tests for KB document processing pipeline."""

from pathlib import Path

import pytest
from aeloon.plugins.KnowledgeBase.config import KBConfig
from aeloon.plugins.KnowledgeBase.processing.chunker import SemanticChunker
from aeloon.plugins.KnowledgeBase.processing.converter import (
    ConverterFactory,
    CsvConverter,
    TextConverter,
)
from aeloon.plugins.KnowledgeBase.processing.extractor import (
    extract_keywords,
    extract_toc,
    generate_summary,
)
from aeloon.plugins.KnowledgeBase.processing.pipeline import DocumentProcessor

# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------


async def test_text_converter(tmp_path: Path) -> None:
    p = tmp_path / "test.txt"
    p.write_text("Hello world", encoding="utf-8")
    result = await TextConverter().convert(p)
    assert result == "Hello world"


async def test_csv_converter(tmp_path: Path) -> None:
    p = tmp_path / "test.csv"
    p.write_text("Name,Age\nAlice,30\nBob,25", encoding="utf-8")
    result = await CsvConverter().convert(p)
    assert "| Name | Age |" in result
    assert "| --- |" in result
    assert "| Alice | 30 |" in result


async def test_converter_factory_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        ConverterFactory().get("xyz")


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------


def test_extract_keywords_english() -> None:
    text = "The transformer architecture revolutionized natural language processing."
    kw = extract_keywords(text)
    assert "transformer" in kw
    assert "architecture" in kw


def test_extract_keywords_chinese() -> None:
    text = "深度学习技术在自然语言处理领域取得了重大突破"
    kw = extract_keywords(text)
    assert any("深度" in k or "学习" in k for k in kw)


def test_extract_keywords_empty() -> None:
    assert extract_keywords("") == []


def test_generate_summary() -> None:
    text = "# Title\n\nThis is the first paragraph. It has useful content.\n\nSecond paragraph."
    summary = generate_summary(text)
    assert "first paragraph" in summary


def test_generate_summary_empty() -> None:
    assert generate_summary("") == ""


def test_extract_toc() -> None:
    md = "# Main\n\n## Section 1\n\n### Sub 1.1\n\n## Section 2"
    toc = extract_toc(md)
    assert len(toc) == 4
    assert toc[0]["level"] == 1
    assert toc[0]["title"] == "Main"


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


def test_chunker_small_text() -> None:
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk("Short text.")
    assert len(chunks) == 1
    assert chunks[0].content == "Short text."


def test_chunker_heading_split() -> None:
    text = "Intro paragraph.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2


def test_chunker_empty() -> None:
    chunker = SemanticChunker()
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_chunker_large_section() -> None:
    # Create a section larger than chunk_size
    text = "## Big\n\n" + "Word. " * 200
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


async def test_pipeline_text(tmp_path: Path) -> None:
    config = KBConfig(chunk_size=512, chunk_overlap=64)
    processor = DocumentProcessor(config)

    p = tmp_path / "test.txt"
    p.write_text("# Hello\n\nWorld of knowledge.", encoding="utf-8")

    result = await processor.process(p)
    assert result.markdown
    assert len(result.keywords) > 0
    assert result.summary


async def test_pipeline_process_text() -> None:
    config = KBConfig(chunk_size=512, chunk_overlap=64)
    processor = DocumentProcessor(config)

    result = await processor.process_text("# Test\n\nSome content here.")
    assert result.markdown
    assert "Test" in result.markdown


async def test_pipeline_empty_file(tmp_path: Path) -> None:
    config = KBConfig()
    processor = DocumentProcessor(config)

    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no content"):
        await processor.process(p)
