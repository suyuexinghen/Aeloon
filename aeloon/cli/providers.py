"""Provider authentication CLI commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import typer

from aeloon import __logo__
from aeloon.cli.app import app, console

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")

_LOGIN_HANDLERS: dict[str, Callable[[], None]] = {}


def _register_login(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Register one provider login handler."""

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        _LOGIN_HANDLERS[name] = fn
        return fn

    return decorator


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(
        ...,
        help="OAuth provider (e.g. 'openai-codex', 'github-copilot')",
    ),
) -> None:
    """Authenticate with an OAuth provider."""
    from aeloon.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((spec for spec in PROVIDERS if spec.name == key and spec.is_oauth), None)
    if not spec:
        names = ", ".join(spec.name.replace("_", "-") for spec in PROVIDERS if spec.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive

        token = None
        try:
            token = get_token()
        except Exception:
            pass
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = login_oauth_interactive(
                print_fn=lambda text: console.print(text),
                prompt_fn=lambda text: typer.prompt(text),
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]"
        )
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    async def _trigger() -> None:
        from litellm import acompletion

        await acompletion(
            model="github_copilot/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )

    try:
        asyncio.run(_trigger())
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
    except Exception as exc:
        console.print(f"[red]Authentication error: {exc}[/red]")
        raise typer.Exit(1)
