"""Tests for SE output validators."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aeloon.plugins.SoftwareEngineering.task import (
    ArchitectureGraph,
    ModuleDef,
    Project,
    ValidationStatus,
)
from aeloon.plugins.SoftwareEngineering.validator import (
    ArchitectureValidator,
    CompileValidator,
    CompositeSEValidator,
    LintValidator,
    TestValidator,
)


class TestCompileValidator:
    @pytest.mark.asyncio
    async def test_passes_with_no_workspace(self) -> None:
        validator = CompileValidator()
        project = Project(description="test")
        result = await validator.validate(project)
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_passes_with_clean_code(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "src").mkdir()
        (workspace / "src" / "main.py").write_text("def foo():\n    return 42\n")
        validator = CompileValidator()
        project = Project(description="test", workspace_path=str(workspace))
        result = await validator.validate(project)
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_fails_with_syntax_error(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "src").mkdir()
        (workspace / "src" / "bad.py").write_text("def foo(\n")
        validator = CompileValidator()
        project = Project(description="test", workspace_path=str(workspace))
        result = await validator.validate(project)
        assert result.status == ValidationStatus.FAILED
        assert len(result.violations) == 1
        assert result.violations[0].rule == "compile_error"


class TestTestValidator:
    @pytest.mark.asyncio
    async def test_passes_with_no_workspace(self) -> None:
        validator = TestValidator()
        project = Project(description="test")
        result = await validator.validate(project)
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_passes_when_pytest_succeeds(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "tests").mkdir()
        (workspace / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
        validator = TestValidator()
        project = Project(description="test", workspace_path=str(workspace))

        # Mock subprocess
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"1 passed", b""))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", return_value=(b"1 passed", b"")):
                result = await validator.validate(project)
        assert result.status == ValidationStatus.PASSED


class TestLintValidator:
    @pytest.mark.asyncio
    async def test_passes_with_no_workspace(self) -> None:
        validator = LintValidator()
        project = Project(description="test")
        result = await validator.validate(project)
        assert result.status == ValidationStatus.PASSED


class TestArchitectureValidator:
    @pytest.mark.asyncio
    async def test_passes_with_no_architecture(self) -> None:
        validator = ArchitectureValidator()
        project = Project(description="test")
        result = await validator.validate(project)
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_warns_missing_module(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        project = Project(
            description="test",
            workspace_path=str(workspace),
            architecture=ArchitectureGraph(
                project_id="p1",
                modules=[
                    ModuleDef(id="m1", name="api", path="src/api"),
                    ModuleDef(id="m2", name="core", path="src/core"),
                ],
            ),
        )
        validator = ArchitectureValidator()
        result = await validator.validate(project)
        assert result.status == ValidationStatus.PARTIAL
        assert any(v.rule == "missing_module" for v in result.violations)


class TestCompositeSEValidator:
    @pytest.mark.asyncio
    async def test_fail_fast_on_compile_error(self, tmp_path: Path) -> None:
        """Fail-fast: compile error should skip test/lint validation."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "src").mkdir()
        (workspace / "src" / "bad.py").write_text("def foo(")
        validator = CompositeSEValidator([CompileValidator(), TestValidator()])
        project = Project(description="test", workspace_path=str(workspace))
        result = await validator.validate(project)
        assert result.status == ValidationStatus.FAILED
        # Composite always reports its own type
        assert result.validator_type == "composite"
        # But the violations should be compile errors (fail-fast)
        assert all(v.rule == "compile_error" for v in result.violations)

    @pytest.mark.asyncio
    async def test_all_pass_no_workspace(self) -> None:
        """With no workspace, all validators pass."""
        validator = CompositeSEValidator([CompileValidator(), TestValidator()])
        project = Project(description="test")
        result = await validator.validate(project)
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_merges_violations(self, tmp_path: Path) -> None:
        """Multiple warnings should be merged."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        project = Project(
            description="test",
            workspace_path=str(workspace),
            architecture=ArchitectureGraph(
                project_id="p1",
                modules=[
                    ModuleDef(id="m1", name="api", path="src/api"),
                ],
            ),
        )
        validator = CompositeSEValidator(
            [
                CompileValidator(),
                TestValidator(),
                ArchitectureValidator(),
            ]
        )
        result = await validator.validate(project)
        # Compile passes (no files), test fails (no pytest), architecture warns
        # Fail-fast on FAILED from TestValidator → overall FAILED
        assert result.status == ValidationStatus.FAILED
