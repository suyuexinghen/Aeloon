"""Tests for StructuralValidator in aeloon/plugins/science/validator.py."""

from __future__ import annotations

from aeloon.plugins.ScienceResearch.task import (
    DeliverableSpec,
    Execution,
    NextAction,
    ValidationStatus,
)
from aeloon.plugins.ScienceResearch.validator import StructuralValidator


def _make_execution(output: str, task_id: str = "task_test") -> Execution:
    return Execution(task_id=task_id, node_id="n1", output=output)


def _make_deliverables(required_sections: list[str] | None = None) -> DeliverableSpec:
    return DeliverableSpec(
        expected_format="markdown",
        required_sections=required_sections or [],
    )


# ---------------------------------------------------------------------------
# PASSED: sufficient length + required sections + URL
# ---------------------------------------------------------------------------


def test_validate_passes_for_good_output():
    validator = StructuralValidator()
    good_output = (
        "## Summary\n"
        "This is a comprehensive overview of the topic covering many aspects. "
        "The research clearly shows that progress has been made.\n\n"
        "## Key Findings\n"
        "- Finding one\n- Finding two\n\n"
        "## Sources\n"
        "1. Paper A https://example.com/paper-a\n"
        "2. Paper B https://example.com/paper-b\n"
    )
    execution = _make_execution(good_output)
    deliverables = _make_deliverables(["summary", "key findings", "sources"])

    result = validator.validate(execution, deliverables)

    assert result.status == ValidationStatus.PASSED
    assert result.next_action == NextAction.DELIVER
    assert result.violations == []


# ---------------------------------------------------------------------------
# FAILED: output too short
# ---------------------------------------------------------------------------


def test_validate_fails_for_short_output():
    validator = StructuralValidator()
    execution = _make_execution("too short")
    deliverables = _make_deliverables()

    result = validator.validate(execution, deliverables)

    assert result.status == ValidationStatus.FAILED
    assert result.next_action == NextAction.RETRY


def test_validate_fails_below_min_chars_threshold():
    validator = StructuralValidator()
    # Exactly 99 chars — below the 100-char minimum
    short = "x" * 99
    execution = _make_execution(short)
    deliverables = _make_deliverables()

    result = validator.validate(execution, deliverables)

    assert result.status == ValidationStatus.FAILED


# ---------------------------------------------------------------------------
# PARTIAL: required sections + length OK, but no URL (warning only)
# ---------------------------------------------------------------------------


def test_validate_partial_for_output_with_no_url():
    validator = StructuralValidator()
    no_url_output = (
        "## Summary\n"
        "This is a comprehensive overview with no URLs at all. "
        "The research is very detailed and covers many topics in depth.\n\n"
        "## Key Findings\n"
        "- Finding one is important\n- Finding two is also important\n\n"
        "## Sources\n"
        "Author A (2024). Title of paper. Journal, volume.\n"
    )
    execution = _make_execution(no_url_output)
    deliverables = _make_deliverables(["summary", "key findings", "sources"])

    result = validator.validate(execution, deliverables)

    assert result.status == ValidationStatus.PARTIAL
    assert result.next_action == NextAction.DELIVER


# ---------------------------------------------------------------------------
# Required section check
# ---------------------------------------------------------------------------


def test_validate_fails_when_required_section_missing():
    validator = StructuralValidator()
    output = (
        "## Key Findings\n"
        "These are the findings. They are very important and well researched. "
        "There are many details here. https://example.com\n"
    )
    execution = _make_execution(output)
    deliverables = _make_deliverables(["Summary"])  # "Summary" not in output

    result = validator.validate(execution, deliverables)

    assert result.status == ValidationStatus.FAILED
    violation_rules = [v.rule for v in result.violations]
    assert "required_section_missing" in violation_rules


def test_validate_violation_message_mentions_missing_section():
    validator = StructuralValidator()
    output = "## Other section\n" + "x" * 200 + " https://example.com"
    execution = _make_execution(output)
    deliverables = _make_deliverables(["Summary"])

    result = validator.validate(execution, deliverables)

    msgs = [v.msg for v in result.violations]
    assert any("Summary" in m for m in msgs)


# ---------------------------------------------------------------------------
# evidence is populated after validate()
# ---------------------------------------------------------------------------


def test_execution_evidence_populated_after_validate():
    validator = StructuralValidator()
    output = "## Summary\n" + "x" * 200 + " https://example.com"
    execution = _make_execution(output)
    deliverables = _make_deliverables(["summary"])

    assert execution.evidence is None
    validator.validate(execution, deliverables)
    assert execution.evidence is not None


def test_execution_evidence_has_validation_log():
    validator = StructuralValidator()
    execution = _make_execution("short")
    deliverables = _make_deliverables()

    validator.validate(execution, deliverables)

    assert execution.evidence is not None
    # evidence.validation_log may be empty if no violations, or populated if there are
    assert isinstance(execution.evidence.validation_log, list)


# ---------------------------------------------------------------------------
# next_action mapping
# ---------------------------------------------------------------------------


def test_next_action_is_deliver_for_passed():
    validator = StructuralValidator()
    good = "## Summary\nLong enough content with lots of detail. " * 5 + " https://example.com"
    execution = _make_execution(good)
    result = validator.validate(execution, _make_deliverables(["summary"]))
    assert result.next_action == NextAction.DELIVER


def test_next_action_is_retry_for_failed():
    validator = StructuralValidator()
    execution = _make_execution("x" * 5)  # too short, guaranteed fail
    result = validator.validate(execution, _make_deliverables())
    assert result.next_action == NextAction.RETRY
