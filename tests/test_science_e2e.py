"""End-to-end integration tests for the AI4S SciencePipeline.

All external calls (LLM / runtime) are mocked so no real network or
model access is needed.  Storage uses pytest's ``tmp_path`` fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aeloon.plugins._sdk.runtime import PluginRuntime
from aeloon.plugins.ScienceResearch.pipeline import SciencePipeline
from aeloon.plugins.ScienceResearch.task import TaskStatus

# ---------------------------------------------------------------------------
# Constants / shared helpers
# ---------------------------------------------------------------------------

# A plausible agent output that satisfies StructuralValidator:
#  - >= 100 chars
#  - Contains "Summary", "Key Findings", "Sources"
#  - Contains a URL
_MOCK_OUTPUT = """\
## Summary
This comprehensive analysis examines recent advances in perovskite solar cell efficiency,
covering the latest experimental results and theoretical models published in the last two years.
Multiple independent research groups have confirmed record-breaking efficiency improvements.

## Key Findings
- Efficiency records now exceed 26% for single-junction perovskite devices.
- Lead-free alternatives using tin and bismuth compounds show promise.
- Stability under humidity remains the main commercial barrier.
- Tandem architectures combining perovskite with silicon reach above 33%.

## Sources
- https://www.nature.com/articles/s41560-023-01234-5 — Nature Energy, 2023
- https://doi.org/10.1021/acsenergylett.3c00123 — ACS Energy Letters, 2023
- https://arxiv.org/abs/2312.01234 — ArXiv preprint on stability mechanisms
"""

_VALID_QUERY = "recent advances in perovskite solar cell efficiency and stability"


def _make_mock_runtime(return_value: str = _MOCK_OUTPUT) -> MagicMock:
    """Return a MagicMock runtime whose process_direct returns *return_value*."""
    runtime = MagicMock(spec=PluginRuntime)
    runtime.process_direct = AsyncMock(return_value=return_value)
    runtime.tool_execute = AsyncMock(return_value=json.dumps({"text": "fetched content"}))
    runtime.add_deep_profile_section = MagicMock()
    runtime.supports_async_tool_execute = True
    return runtime


def _make_pipeline(tmp_path: Path, runtime: MagicMock | None = None) -> SciencePipeline:
    """Construct a SciencePipeline backed by *tmp_path* storage."""
    if runtime is None:
        runtime = _make_mock_runtime()
    return SciencePipeline(runtime=runtime, storage_dir=tmp_path)


# ---------------------------------------------------------------------------
# 1. Full pipeline run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_full_run(tmp_path: Path) -> None:
    """Full pipeline with mocked runtime should complete successfully."""
    pipeline = _make_pipeline(tmp_path)
    progress_messages: list[str] = []

    async def capture_progress(msg: str, **_kwargs: object) -> None:
        progress_messages.append(msg)

    output, task = await pipeline.run(_VALID_QUERY, on_progress=capture_progress)

    # Output must contain the required sections
    assert "Summary" in output
    assert "Key Findings" in output
    assert "Sources" in output

    # Task must be marked completed
    assert task.status == TaskStatus.COMPLETED

    # Progress updates must have been emitted
    assert len(progress_messages) >= 1


# ---------------------------------------------------------------------------
# 2. History after a run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_history(tmp_path: Path) -> None:
    """get_history() should list the task after a successful run."""
    pipeline = _make_pipeline(tmp_path)
    _, task = await pipeline.run(_VALID_QUERY)

    history = pipeline.get_history()

    assert task.task_id in history
    # Either part of the goal should appear (storage truncates at 80 chars)
    assert "perovskite" in history.lower() or "solar" in history.lower()


# ---------------------------------------------------------------------------
# 3. Status after a run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_status(tmp_path: Path) -> None:
    """get_status() should return task id and goal after a run."""
    pipeline = _make_pipeline(tmp_path)
    _, task = await pipeline.run(_VALID_QUERY)

    status = pipeline.get_status()

    assert task.task_id in status
    # The goal should appear (possibly truncated)
    assert "perovskite" in status.lower() or "solar" in status.lower()


# ---------------------------------------------------------------------------
# 4. Vague query clarification hint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_vague_query_hint(tmp_path: Path) -> None:
    """A 2-word query should emit a clarification hint as the first progress update."""
    pipeline = _make_pipeline(tmp_path)
    progress_messages: list[str] = []

    async def capture_progress(msg: str, **_kwargs: object) -> None:
        progress_messages.append(msg)

    # Two words — below the _MIN_GOAL_WORDS=4 threshold
    await pipeline.run("solar cells", on_progress=capture_progress)

    assert len(progress_messages) >= 1
    first_msg = progress_messages[0].lower()
    # Should mention the short / vague nature of the query
    assert "short" in first_msg or "vague" in first_msg or "word" in first_msg


# ---------------------------------------------------------------------------
# 5. Failed orchestration propagates error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_failed_node_propagates(tmp_path: Path) -> None:
    """If the orchestrator raises an Exception, the pipeline returns an error string
    and marks the task as FAILED."""
    pipeline = _make_pipeline(tmp_path)

    with patch.object(
        pipeline._orchestrator,
        "run",
        new_callable=AsyncMock,
        side_effect=RuntimeError("mock orchestration failure"),
    ):
        output, task = await pipeline.run(_VALID_QUERY)

    assert output.startswith("Error:")
    assert "mock orchestration failure" in output
    assert task.status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# 6. JSONL task log complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jsonl_task_log_complete(tmp_path: Path) -> None:
    """After a successful run, the JSONL file for the task must exist and
    contain at least one task record and one execution record."""
    pipeline = _make_pipeline(tmp_path)
    _, task = await pipeline.run(_VALID_QUERY)

    jsonl_path = tmp_path / "tasks" / f"{task.task_id}.jsonl"
    assert jsonl_path.exists(), f"Expected JSONL file at {jsonl_path}"

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

    task_records = [r for r in records if r.get("_type") == "task"]
    execution_records = [r for r in records if r.get("_type") == "execution"]

    assert len(task_records) >= 1, "Expected at least one task record in JSONL"
    assert len(execution_records) >= 1, "Expected at least one execution record in JSONL"

    # Verify the last task record reflects the completed status
    last_task = task_records[-1]
    assert last_task["task_id"] == task.task_id
    assert last_task["status"] == TaskStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_pipeline_populates_deep_profile_sections(tmp_path: Path) -> None:
    """Science pipeline should call runtime.add_deep_profile_section for workflow stages."""
    runtime = _make_mock_runtime()
    pipeline = SciencePipeline(runtime=runtime, storage_dir=tmp_path)

    await pipeline.run(_VALID_QUERY)

    # Verify add_deep_profile_section was called with the expected stage names
    call_titles = [call.args[0] for call in runtime.add_deep_profile_section.call_args_list]
    assert any("Science · Interpret" in title for title in call_titles)
    assert any("Science · Plan" in title for title in call_titles)
    assert any("Science · Execute" in title for title in call_titles)
    assert any("Science · Validate" in title for title in call_titles)
    assert any("Science · Deliver" in title for title in call_titles)
