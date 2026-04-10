"""Display helpers for interactive CLI output."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

from prompt_toolkit import print_formatted_text
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from aeloon import __logo__


def make_console() -> Console:
    """Build a console writing to stdout."""
    return Console(file=sys.stdout)


def try_render_inline_image(path: str | Path) -> bool:
    """Try to render an image inline in the terminal using native protocols."""
    target = Path(path).expanduser()
    if not target.exists():
        return False

    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")

    if term == "xterm-kitty":
        try:
            subprocess.run(["kitty", "+kitten", "icat", str(target)], check=True, timeout=5)
            return True
        except Exception:
            return False

    if term_program == "iTerm.app":
        try:
            data = base64.b64encode(target.read_bytes()).decode("ascii")
            sys.stdout.write(f"\033]1337;File=inline=1;size={target.stat().st_size}:{data}\a")
            sys.stdout.flush()
            return True
        except Exception:
            return False

    return False


def try_open_media_file(path: str | Path) -> bool:
    """Try to open a local media file with a platform-native viewer."""
    target = Path(path).expanduser()
    if not target.exists():
        return False

    target_str = str(target)
    if sys.platform == "win32":
        startfile = getattr(os, "startfile", None)
        if startfile is not None:
            try:
                startfile(target_str)
                return True
            except OSError:
                pass
        candidates = [
            ["powershell", "-NoProfile", "-Command", "Start-Process", "-FilePath", target_str],
            ["explorer", target_str],
        ]
    elif sys.platform == "darwin":
        candidates = [["open", target_str]]
    else:
        candidates = [
            ["xdg-open", target_str],
            ["gio", "open", target_str],
            ["gnome-open", target_str],
            ["kde-open", target_str],
            ["kde-open5", target_str],
            ["see", target_str],
        ]

    for command in candidates:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except OSError:
            continue

    return False


async def handle_interactive_media(
    media_paths: list[str],
    thinking,
    *,
    print_interactive_progress_line,
) -> None:
    """Try to render outbound media inline, or print path as fallback."""
    for media_path in media_paths:
        if try_render_inline_image(media_path):
            continue
        await print_interactive_progress_line(
            f"Unable to display image. File: {media_path}",
            thinking,
        )


def render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can print it safely."""
    ansi_console = Console(force_terminal=True, color_system="standard", width=Console().width)
    with ansi_console.capture() as capture:
        render_fn(ansi_console)
    return capture.get()


def print_agent_response(response: str, render_markdown: bool, *, console) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    local_console = make_console()
    local_console.print()
    local_console.print(f"[cyan]{__logo__} aeloon[/cyan]")
    local_console.print(body)
    local_console.print()


async def print_interactive_line(text: str) -> None:
    """Print async interactive updates with prompt_toolkit-safe Rich styling."""

    def _write() -> None:
        ansi = render_interactive_ansi(lambda c: c.print(f"  [dim]↳ {text}[/dim]"))
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


async def print_interactive_response(response: str, render_markdown: bool) -> None:
    """Print async interactive replies with prompt_toolkit-safe Rich styling."""

    def _write() -> None:
        content = response or ""
        ansi = render_interactive_ansi(
            lambda c: (
                c.print(),
                c.print(f"[cyan]{__logo__} aeloon[/cyan]"),
                c.print(Markdown(content) if render_markdown else Text(content)),
                c.print(),
            )
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


class ThinkingSpinner:
    """Spinner wrapper with pause support for clean progress output."""

    def __init__(self, enabled: bool, *, console):
        self._spinner = (
            console.status("[dim]aeloon is thinking...[/dim]", spinner="dots") if enabled else None
        )
        self._active = False

    def __enter__(self):
        if self._spinner:
            self._spinner.start()
        self._active = True
        return self

    def __exit__(self, *exc):
        self._active = False
        if self._spinner:
            self._spinner.stop()
        return False

    @contextmanager
    def pause(self):
        """Temporarily stop spinner while printing progress."""
        if self._spinner and self._active:
            self._spinner.stop()
        try:
            yield
        finally:
            if self._spinner and self._active:
                self._spinner.start()


def print_cli_progress_line(text: str, thinking, *, console) -> None:
    """Print a CLI progress line, pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        console.print(f"  [dim]↳ {text}[/dim]")


async def print_interactive_progress_line(text: str, thinking, *, print_interactive_line) -> None:
    """Print an interactive progress line, pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        await print_interactive_line(text)


async def print_interactive_profile_report(text: str, thinking) -> None:
    """Print an async profiling report panel in interactive mode."""

    def _write() -> None:
        ansi = render_interactive_ansi(
            lambda c: c.print(Panel(Text(text), title="Profiling", border_style="cyan"))
        )
        print_formatted_text(ANSI(ansi), end="")

    with thinking.pause() if thinking else nullcontext():
        await run_in_terminal(_write)


def print_stderr_profile_report(text: str) -> None:
    """Print profiling report to stderr."""
    err_console = Console(file=sys.stderr)
    err_console.print(Panel(Text(text), title="Profiling", border_style="cyan"))
