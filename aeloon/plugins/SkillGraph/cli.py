"""CLI builder for the SkillGraph compiler plugin."""

from __future__ import annotations

import shlex

import typer


def build_skill_compiler_cli_builder(plugin_command: str = "skill_compiler"):
    """Return a Typer builder for `aeloon ext skill_compiler ...`."""

    def _builder(app: typer.Typer) -> None:
        @app.command(name="skill_compiler")
        def _compile(  # type: ignore[misc]
            skill_path: str = typer.Argument(..., help="Path to the skill directory"),
            model: str | None = typer.Option(None, "--model", help="Analyzer model override"),
            runtime_model: str | None = typer.Option(
                None, "--runtime-model", help="Runtime model override"
            ),
            strict_validate: bool = typer.Option(
                False,
                "--strict-validate",
                help="Fail on strict validation checks",
            ),
            session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
            workspace: str | None = typer.Option(
                None, "--workspace", "-w", help="Workspace directory"
            ),
            config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
        ) -> None:
            """Run the SkillGraph compiler through the shared plugin CLI runner."""
            from aeloon.cli.plugins import run_plugin_cli_command

            parts = [shlex.quote(skill_path)]
            if model:
                parts.extend(["--model", shlex.quote(model)])
            if runtime_model:
                parts.extend(["--runtime-model", shlex.quote(runtime_model)])
            if strict_validate:
                parts.append("--strict-validate")

            run_plugin_cli_command(
                plugin_command=plugin_command,
                args=" ".join(parts),
                session_id=session,
                workspace=workspace,
                config=config,
            )

    return _builder
