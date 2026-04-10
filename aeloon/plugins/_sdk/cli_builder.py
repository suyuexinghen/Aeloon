"""Helpers for building Typer commands from declarative plugin CLI specs."""

from __future__ import annotations

from collections.abc import Iterable

import typer

from aeloon.plugins._sdk.types import CLICommandSpec, CLIFlagSpec


def build_cli_group_builder(
    plugin_id: str,
    specs: Iterable[CLICommandSpec],
):
    """Return a Typer builder for one plugin CLI group."""
    spec_list = list(specs)
    if not spec_list:
        raise ValueError(f"No CLI command specs registered for {plugin_id}")

    group_name = spec_list[0].group_name
    group_help = spec_list[0].group_help or spec_list[0].help

    def _builder(app: typer.Typer) -> None:
        group_app = typer.Typer(help=group_help)

        for spec in spec_list:
            _register_command(group_app, spec)

        app.add_typer(group_app, name=group_name)

    return _builder


def _register_command(group_app: typer.Typer, spec: CLICommandSpec) -> None:
    """Attach one declared CLI command to a Typer group."""
    if len(spec.flags) > 1:
        raise ValueError(
            f"Command '{spec.group_name} {spec.command_name}' supports at most one flag"
        )

    if spec.message is None and not spec.flags:

        @group_app.command(spec.command_name)
        def _command_without_message(  # type: ignore[misc]
            session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
            workspace: str | None = typer.Option(
                None, "--workspace", "-w", help="Workspace directory"
            ),
            config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
        ) -> None:
            """Run one plugin command through the shared plugin CLI runner."""
            _run_plugin_command(
                spec,
                session=session,
                workspace=workspace,
                config=config,
            )

        _command_without_message.__doc__ = spec.help
        return

    flag = spec.flags[0] if spec.flags else None
    if spec.message is None:

        @group_app.command(spec.command_name)
        def _command_with_flag(  # type: ignore[misc]
            enabled: bool = typer.Option(
                flag.default,
                *flag.flags,
                help=flag.help,
            ),
            session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
            workspace: str | None = typer.Option(
                None, "--workspace", "-w", help="Workspace directory"
            ),
            config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
        ) -> None:
            """Run one plugin command through the shared plugin CLI runner."""
            _run_plugin_command(
                spec,
                session=session,
                workspace=workspace,
                config=config,
                **_flag_template_values(flag, enabled),
            )

        _command_with_flag.__doc__ = spec.help
        return

    if spec.message.parameter_kind == "argument" and flag is None:
        if spec.message.required:

            @group_app.command(spec.command_name)
            def _command_with_required_argument(  # type: ignore[misc]
                message: str = typer.Argument(..., help=spec.message.help),
                session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
                workspace: str | None = typer.Option(
                    None, "--workspace", "-w", help="Workspace directory"
                ),
                config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
            ) -> None:
                """Run one plugin command through the shared plugin CLI runner."""
                _run_plugin_command(
                    spec,
                    session=session,
                    workspace=workspace,
                    config=config,
                    message=message,
                )

            _command_with_required_argument.__doc__ = spec.help
            return

        @group_app.command(spec.command_name)
        def _command_with_optional_argument(  # type: ignore[misc]
            message: str | None = typer.Argument(None, help=spec.message.help),
            session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
            workspace: str | None = typer.Option(
                None, "--workspace", "-w", help="Workspace directory"
            ),
            config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
        ) -> None:
            """Run one plugin command through the shared plugin CLI runner."""
            _run_plugin_command(
                spec,
                session=session,
                workspace=workspace,
                config=config,
                message=message or spec.message.default,
            )

        _command_with_optional_argument.__doc__ = spec.help
        return

    if spec.message.parameter_kind == "argument":
        if spec.message.required:

            @group_app.command(spec.command_name)
            def _command_with_argument_and_flag(  # type: ignore[misc]
                message: str = typer.Argument(..., help=spec.message.help),
                enabled: bool = typer.Option(
                    flag.default,
                    *flag.flags,
                    help=flag.help,
                ),
                session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
                workspace: str | None = typer.Option(
                    None, "--workspace", "-w", help="Workspace directory"
                ),
                config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
            ) -> None:
                """Run one plugin command through the shared plugin CLI runner."""
                _run_plugin_command(
                    spec,
                    session=session,
                    workspace=workspace,
                    config=config,
                    message=message,
                    **_flag_template_values(flag, enabled),
                )

            _command_with_argument_and_flag.__doc__ = spec.help
            return

        @group_app.command(spec.command_name)
        def _command_with_optional_argument_and_flag(  # type: ignore[misc]
            message: str | None = typer.Argument(None, help=spec.message.help),
            enabled: bool = typer.Option(
                flag.default,
                *flag.flags,
                help=flag.help,
            ),
            session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
            workspace: str | None = typer.Option(
                None, "--workspace", "-w", help="Workspace directory"
            ),
            config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
        ) -> None:
            """Run one plugin command through the shared plugin CLI runner."""
            _run_plugin_command(
                spec,
                session=session,
                workspace=workspace,
                config=config,
                message=message or spec.message.default,
                **_flag_template_values(flag, enabled),
            )

        _command_with_optional_argument_and_flag.__doc__ = spec.help
        return

    if flag is None:

        @group_app.command(spec.command_name)
        def _command_with_option(  # type: ignore[misc]
            message: str = typer.Option(
                ... if spec.message.required else spec.message.default,
                *spec.message.flags,
                help=spec.message.help,
            ),
            session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
            workspace: str | None = typer.Option(
                None, "--workspace", "-w", help="Workspace directory"
            ),
            config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
        ) -> None:
            """Run one plugin command through the shared plugin CLI runner."""
            _run_plugin_command(
                spec,
                session=session,
                workspace=workspace,
                config=config,
                message=message,
            )

        _command_with_option.__doc__ = spec.help
        return

    @group_app.command(spec.command_name)
    def _command_with_option_and_flag(  # type: ignore[misc]
        message: str = typer.Option(
            ... if spec.message.required else spec.message.default,
            *spec.message.flags,
            help=spec.message.help,
        ),
        enabled: bool = typer.Option(
            flag.default,
            *flag.flags,
            help=flag.help,
        ),
        session: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
        workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
        config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
    ) -> None:
        """Run one plugin command through the shared plugin CLI runner."""
        _run_plugin_command(
            spec,
            session=session,
            workspace=workspace,
            config=config,
            message=message,
            **_flag_template_values(flag, enabled),
        )

    _command_with_option_and_flag.__doc__ = spec.help


def _flag_template_values(flag: CLIFlagSpec, enabled: bool) -> dict[str, str]:
    return {flag.name: flag.value_when_true if enabled else flag.value_when_false}


def _run_plugin_command(
    spec: CLICommandSpec,
    *,
    session: str | None,
    workspace: str | None,
    config: str | None,
    **template_values: str,
) -> None:
    """Run one plugin command through the shared plugin CLI runner."""
    from aeloon.cli.plugins import run_plugin_cli_command

    values = {"message": "", **template_values}
    args = spec.args_template.format(**values).strip()
    run_plugin_cli_command(
        plugin_command=spec.plugin_command,
        args=args,
        session_id=session,
        workspace=workspace,
        config=config,
    )
