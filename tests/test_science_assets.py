"""Tests for AssetManager (SP6 P6.2)."""

from __future__ import annotations

from pathlib import Path

from aeloon.plugins.ScienceResearch.assets import AssetManager

# ---------------------------------------------------------------------------
# AssetManager — templates
# ---------------------------------------------------------------------------


def test_extract_template_creates_file(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    tmpl = mgr.extract_template(
        task_id="abc12345",
        goal="Find papers on perovskite solar cells",
        scenario="literature_analysis",
        node_ids=["search", "fetch", "synthesize"],
        node_objectives=["search papers", "fetch content", "write summary"],
    )
    assert tmpl.template_id == "tmpl_abc12345"
    assert tmpl.scenario == "literature_analysis"
    assert (tmp_path / "assets" / "templates" / f"{tmpl.template_id}.json").exists()


def test_list_templates_empty(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    assert mgr.list_templates() == []


def test_list_templates_returns_saved(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    mgr.extract_template(
        task_id="task0001",
        goal="Survey quantum error correction",
        scenario="literature_analysis",
        node_ids=["search"],
        node_objectives=["find papers"],
    )
    templates = mgr.list_templates()
    assert len(templates) == 1
    assert templates[0].scenario == "literature_analysis"


def test_list_templates_filter_by_scenario(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    mgr.extract_template(
        task_id="t1000000",
        goal="High-entropy alloys",
        scenario="materials_science",
        node_ids=["search"],
        node_objectives=["search"],
    )
    mgr.extract_template(
        task_id="t2000000",
        goal="Perovskite solar cells",
        scenario="literature_analysis",
        node_ids=["search"],
        node_objectives=["search"],
    )
    lit = mgr.list_templates(scenario="literature_analysis")
    mat = mgr.list_templates(scenario="materials_science")
    assert len(lit) == 1
    assert len(mat) == 1


def test_find_similar_returns_closest_match(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    mgr.extract_template(
        task_id="sim10000",
        goal="recent papers on perovskite solar efficiency",
        scenario="literature_analysis",
        node_ids=["s", "f", "syn"],
        node_objectives=["search", "fetch", "synthesize"],
    )
    result = mgr.find_similar("perovskite solar cell efficiency review")
    assert result is not None
    assert result.template_id == "tmpl_sim10000"


def test_find_similar_returns_none_when_no_match(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    mgr.extract_template(
        task_id="nm100000",
        goal="quantum computing error correction",
        scenario="quantum",
        node_ids=["s"],
        node_objectives=["search"],
    )
    result = mgr.find_similar("protein folding deep learning")
    assert result is None


# ---------------------------------------------------------------------------
# AssetManager — failure patterns
# ---------------------------------------------------------------------------


def test_record_failure_creates_file(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    p = mgr.record_failure(
        task_id="fail1234",
        node_id="search_node",
        tool_or_capability="web_search",
        error_type="TimeoutError",
        error_summary="Request timed out after 30s",
        scenario="literature_analysis",
    )
    assert p.pattern_id == "fail1234_search_node"
    assert (tmp_path / "assets" / "failures" / f"{p.pattern_id}.json").exists()


def test_list_failures_empty(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    assert mgr.list_failures() == []


def test_failure_count_increments(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    mgr.record_failure(
        task_id="fc100000",
        node_id="n1",
        tool_or_capability="web_search",
        error_type="TimeoutError",
        error_summary="Timeout",
    )
    mgr.record_failure(
        task_id="fc200000",
        node_id="n2",
        tool_or_capability="web_search",
        error_type="ConnectionError",
        error_summary="Connection refused",
    )
    assert mgr.failure_count("web_search") == 2
    assert mgr.failure_count("web_fetch") == 0


def test_list_failures_filter_by_tool(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    mgr.record_failure("ft100000", "n1", "web_search", "Err", "msg")
    mgr.record_failure("ft200000", "n2", "web_fetch", "Err", "msg")
    results = mgr.list_failures(tool_or_capability="web_fetch")
    assert len(results) == 1
    assert results[0].tool_or_capability == "web_fetch"


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_index_file_created(tmp_path: Path) -> None:
    mgr = AssetManager(tmp_path)
    mgr.extract_template(
        task_id="idx10000",
        goal="index test",
        scenario="test",
        node_ids=[],
        node_objectives=[],
    )
    index_path = tmp_path / "assets" / "index.json"
    assert index_path.exists()
    import json

    data = json.loads(index_path.read_text())
    assert "template" in data
