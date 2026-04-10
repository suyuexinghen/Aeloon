"""Science output validators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .task import (
    DeliverableSpec,
    Evidence,
    Execution,
    NextAction,
    Validation,
    ValidationStatus,
    Violation,
)

# Order used when picking the "worst" next_action across composite results.
_NEXT_ACTION_SEVERITY: dict[NextAction, int] = {
    NextAction.DELIVER: 0,
    NextAction.ESCALATE: 1,
    NextAction.SUBSTITUTE: 2,
    NextAction.REPLAN: 3,
    NextAction.RETRY: 4,
}


class Validator(ABC):
    """Abstract base class for science output validators."""

    @abstractmethod
    def validate(
        self,
        execution: Execution,
        deliverables: DeliverableSpec,
    ) -> Validation:
        """Validate a node execution output against delivery expectations."""
        ...


class StructuralValidator(Validator):
    """Validates that output is non-empty, reasonably long, and contains
    required sections and at least one source reference.

    This is the v0.1 walking-skeleton validator.  It performs structural
    checks only — semantic and domain validation are added in Phase 2.
    """

    MIN_OUTPUT_CHARS = 100
    URL_PATTERNS = ("http://", "https://", "doi:", "arxiv:", "www.")

    def validate(
        self,
        execution: Execution,
        deliverables: DeliverableSpec,
        **kwargs: object,
    ) -> Validation:
        output = execution.output or ""
        violations: list[Violation] = []
        evidence: dict[str, object] = {}

        # --- Check 1: non-empty and minimum length ---
        evidence["output_length"] = len(output)
        if len(output) < self.MIN_OUTPUT_CHARS:
            violations.append(
                Violation(
                    rule="min_output_length",
                    msg=(
                        f"Output is too short ({len(output)} chars, "
                        f"minimum {self.MIN_OUTPUT_CHARS})."
                    ),
                )
            )

        # --- Check 2: required sections present ---
        missing_sections: list[str] = []
        for section in deliverables.required_sections:
            normalized = section.lower()
            if normalized not in output.lower():
                missing_sections.append(section)
        evidence["missing_sections"] = missing_sections
        for section in missing_sections:
            violations.append(
                Violation(
                    rule="required_section_missing",
                    msg=f"Required section not found in output: '{section}'",
                )
            )

        # --- Check 3: at least one source/URL present ---
        has_source = any(pattern in output for pattern in self.URL_PATTERNS)
        evidence["has_source_reference"] = has_source
        if not has_source and deliverables.required_sections:
            # Only enforce source check when we asked for sections (literature tasks)
            violations.append(
                Violation(
                    rule="no_source_reference",
                    msg="Output does not contain any source URLs or references.",
                    severity="warning",
                )
            )

        # --- Determine overall status ---
        error_violations = [v for v in violations if v.severity == "error"]
        warning_violations = [v for v in violations if v.severity == "warning"]

        if error_violations:
            status = ValidationStatus.FAILED
            next_action = NextAction.RETRY
        elif warning_violations:
            status = ValidationStatus.PARTIAL
            next_action = NextAction.DELIVER
        else:
            status = ValidationStatus.PASSED
            next_action = NextAction.DELIVER

        criteria = ["min_output_length", "required_sections", "source_reference"]
        ev = Evidence(
            sources=[],
            validation_log=[f"{v.rule}: {v.msg}" for v in violations],
        )
        execution.evidence = ev

        return Validation(
            criteria=criteria,
            evidence=dict(evidence),
            status=status,
            confidence=1.0 - 0.25 * len(error_violations),
            violations=violations,
            next_action=next_action,
        )


class SemanticValidator(Validator):
    """Validates that output semantically addresses the original task goal.

    Uses a simple keyword-overlap heuristic (v0.2): extracts significant words
    from the task goal and checks what fraction appear in the output.  If fewer
    than 30 % of goal keywords are found, a ``semantic_coverage`` warning is
    added.  No LLM call is made.
    """

    COVERAGE_THRESHOLD: float = 0.3
    MIN_WORD_LEN: int = 4

    def validate(  # type: ignore[override]
        self,
        execution: Execution,
        deliverables: DeliverableSpec,
        task_goal: str = "",
    ) -> Validation:
        """Validate semantic coverage of *output* against *task_goal*.

        The extra *task_goal* kwarg is intentional — it extends the base
        ``Validator`` signature with a default so callers that only pass the
        two positional arguments still work.
        """
        output = (execution.output or "").lower()
        violations: list[Violation] = []
        evidence: dict[str, object] = {}

        keywords: list[str] = [w.lower() for w in task_goal.split() if len(w) > self.MIN_WORD_LEN]
        evidence["goal_keywords"] = keywords

        if keywords:
            found = [kw for kw in keywords if kw in output]
            coverage = len(found) / len(keywords)
        else:
            found = []
            coverage = 1.0  # no goal keywords — nothing to check

        evidence["keywords_found"] = found
        evidence["coverage"] = coverage

        if coverage < self.COVERAGE_THRESHOLD:
            violations.append(
                Violation(
                    rule="semantic_coverage",
                    msg=(
                        f"Output covers only {coverage:.0%} of goal keywords "
                        f"(threshold {self.COVERAGE_THRESHOLD:.0%})."
                    ),
                    severity="warning",
                )
            )

        warning_violations = [v for v in violations if v.severity == "warning"]
        if warning_violations:
            status = ValidationStatus.PARTIAL
            next_action = NextAction.DELIVER
        else:
            status = ValidationStatus.PASSED
            next_action = NextAction.DELIVER

        return Validation(
            criteria=["semantic_coverage"],
            evidence=dict(evidence),
            status=status,
            confidence=coverage,
            violations=violations,
            next_action=next_action,
        )


class CompositeValidator(Validator):
    """Runs multiple validators in sequence and merges their results.

    Status precedence (worst-wins): FAILED > PARTIAL > PASSED.
    next_action precedence (worst-wins): RETRY > REPLAN > SUBSTITUTE > ESCALATE > DELIVER.
    confidence: minimum across all validators.
    """

    def __init__(self, validators: list[Validator]) -> None:
        self._validators = validators

    def validate(
        self,
        execution: Execution,
        deliverables: DeliverableSpec,
        **kwargs: object,
    ) -> Validation:
        """Run each validator and merge results into a single ``Validation``."""
        all_violations: list[Violation] = []
        all_criteria: list[str] = []
        all_evidence: dict[str, object] = {}
        statuses: list[ValidationStatus] = []
        next_actions: list[NextAction] = []
        confidences: list[float] = []

        for validator in self._validators:
            result = validator.validate(execution, deliverables, **kwargs)  # type: ignore[call-arg]
            all_violations.extend(result.violations)
            all_criteria.extend(result.criteria)
            all_evidence.update(result.evidence)
            statuses.append(result.status)
            next_actions.append(result.next_action)
            confidences.append(result.confidence)

        # Determine overall status
        if ValidationStatus.FAILED in statuses:
            merged_status = ValidationStatus.FAILED
        elif ValidationStatus.PARTIAL in statuses:
            merged_status = ValidationStatus.PARTIAL
        else:
            merged_status = ValidationStatus.PASSED

        # Determine overall next_action (worst wins)
        merged_next_action = max(
            next_actions,
            key=lambda a: _NEXT_ACTION_SEVERITY.get(a, 0),
        )

        merged_confidence = min(confidences) if confidences else 1.0

        return Validation(
            criteria=all_criteria,
            evidence=all_evidence,
            status=merged_status,
            confidence=merged_confidence,
            violations=all_violations,
            next_action=merged_next_action,
        )


def make_default_validator() -> CompositeValidator:
    """Return the default validator chain for science tasks."""
    return CompositeValidator([StructuralValidator(), SemanticValidator()])
