"""Slash-navigation helpers for the interactive CLI."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.containers import VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.widgets import Box, Frame

from aeloon.cli.app import command_catalog
from aeloon.cli.registry import SlashSegment
from aeloon.plugins._sdk.admin import suggest_plugin_entries


def get_slash_candidates(query: str) -> list[tuple[str, str]]:
    """Return slash candidates for one query."""
    return [
        (candidate.label, candidate.description) for candidate in get_navigation_segments(query)
    ]


def get_slash_segments(query: str) -> list[SlashSegment]:
    """Return immediate slash-navigation segments for one query."""
    return command_catalog.slash_segments(query)


def slash_active_prefix(query: str) -> str:
    """Return the in-progress segment being edited in one slash query."""
    raw = query.strip() if not query.endswith(" ") else query
    raw = raw.lstrip("/")
    if not raw.strip() or raw.endswith(" "):
        return ""
    return raw.split()[-1]


def session_switch_segments(
    query: str,
    *,
    agent_loop: Any = None,
    current_session_key: str | None = None,
) -> list[SlashSegment]:
    """Return dynamic session-key segments for resume/session switch flows."""
    if agent_loop is None:
        return []

    raw = query.lstrip("/")
    trailing_space = query.endswith(" ")
    parts = raw.split()
    if len(parts) < 2:
        return []

    base_command = parts[0].lower()
    subcommand = parts[1].lower()
    if base_command not in {"resume", "sessions"} or subcommand != "switch":
        return []

    prefix = ""
    if len(parts) > 2 and not trailing_space:
        prefix = parts[2].lower()

    segments: list[SlashSegment] = []
    for item in agent_loop.sessions.list_sessions()[:20]:
        key = str(item.get("key") or "")
        if not key:
            continue
        if prefix and not key.lower().startswith(prefix):
            continue
        updated_at = str(item.get("updated_at") or "unknown")
        description = f"{updated_at}{' (current)' if key == current_session_key else ''}"
        segments.append(
            SlashSegment(
                segment=key,
                description=description,
                path=(base_command, "switch", key),
            )
        )

    if segments:
        return segments

    if trailing_space or len(parts) == 2:
        return [
            SlashSegment(
                segment="<session-key>",
                description="Saved session key",
                path=(base_command, "switch", "<session-key>"),
            )
        ]
    return []


def plugin_name_segments(query: str) -> list[SlashSegment]:
    """Return dynamic plugin-name segments for plugin admin commands."""
    raw = query.lstrip("/")
    trailing_space = query.endswith(" ")
    parts = raw.split()
    if len(parts) < 2:
        return []

    base_command = parts[0].lower()
    action = parts[1].lower()
    if base_command != "plugin" or action not in {"activate", "deactivate", "remove", "error"}:
        return []

    prefix = ""
    if len(parts) > 2 and not trailing_space:
        prefix = parts[2].lower()

    entries = suggest_plugin_entries(action)
    segments = [
        SlashSegment(
            segment=entry.id,
            description=f"{entry.source} · {entry.status} · v{entry.version}",
            path=("plugin", action, entry.id),
        )
        for entry in entries
        if not prefix or entry.id.lower().startswith(prefix)
    ]
    if segments:
        return segments
    if trailing_space or len(parts) == 2:
        return [
            SlashSegment(
                segment="<name>",
                description="Installed plugin ID",
                path=("plugin", action, "<name>"),
            )
        ]
    return []


def get_navigation_segments(
    query: str,
    *,
    agent_loop: Any = None,
    current_session_key: str | None = None,
) -> list[SlashSegment]:
    """Return staged navigation segments, with dynamic runtime overlays when available."""
    dynamic_segments = session_switch_segments(
        query,
        agent_loop=agent_loop,
        current_session_key=current_session_key,
    )
    return dynamic_segments or plugin_name_segments(query) or get_slash_segments(query)


def should_open_slash_palette(
    command: str,
    *,
    agent_loop: Any = None,
    current_session_key: str | None = None,
) -> bool:
    """Return True when interactive slash navigation should open."""
    if command == "/":
        return True
    if not command.startswith("/"):
        return False
    segments = get_navigation_segments(
        command,
        agent_loop=agent_loop,
        current_session_key=current_session_key,
    )
    if not segments:
        return False
    normalized = command.lstrip("/").strip().lower()
    exact = next(
        (segment for segment in segments if " ".join(segment.path).lower() == normalized), None
    )
    if exact is not None and not exact.has_children:
        return False
    return True


def auto_descend_query(
    query: str,
    *,
    agent_loop: Any = None,
    current_session_key: str | None = None,
) -> str:
    """Advance an exact slash path to its first selectable child level."""
    current = query
    if current.endswith(" "):
        return current

    while current:
        segments = get_navigation_segments(
            current,
            agent_loop=agent_loop,
            current_session_key=current_session_key,
        )
        normalized = current.strip().lstrip("/").lower()
        exact = next(
            (segment for segment in segments if " ".join(segment.path).lower() == normalized),
            None,
        )
        if exact is None or not exact.has_children:
            return current
        current = " ".join(exact.path) + " "

    return current


def rank_slash_commands(query: str) -> list[tuple[str, str]]:
    """Rank slash commands using prefix matching only."""
    normalized = query.strip().lower()
    candidates = get_slash_candidates(query)
    if candidates:
        return candidates

    commands = command_catalog.slash_commands()
    if not normalized:
        return get_slash_candidates("/")

    return [
        (cmd, desc)
        for cmd, desc in commands
        if cmd.lower().startswith(f"/{normalized}") or cmd[1:].lower().startswith(normalized)
    ]


class SlashCommandCompleter(Completer):
    """Autocomplete slash commands one level at a time."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.lstrip().startswith("/"):
            return

        normalized = text.lstrip()
        segments = get_navigation_segments(normalized[1:] if normalized != "/" else "")
        if not segments:
            return

        prefix = slash_active_prefix(normalized)
        for segment in segments:
            yield Completion(
                segment.segment,
                start_position=-len(prefix),
                display=segment.segment,
                display_meta=segment.description,
            )


class PaletteInputProcessor(Processor):
    """Hide the leading slash while the slash palette filter is active."""

    def apply_transformation(self, transformation_input):
        fragments = transformation_input.fragments
        if fragments and fragments[0][1].startswith("/"):
            style, text, *rest = fragments[0]
            fragments = [tuple([style, text[1:], *rest])] + fragments[1:]
        return Transformation(fragments)


async def interactive_slash_palette(
    initial_query: str = "/",
    *,
    agent_loop: Any = None,
    current_session_key: str | None = None,
) -> str | None:
    """Prompt-toolkit slash command palette with staged navigation."""
    result: dict[str, str | None] = {"value": None}
    selected_index = 0
    initial_text = initial_query[1:] if initial_query.startswith("/") else initial_query
    initial_text = auto_descend_query(
        initial_text,
        agent_loop=agent_loop,
        current_session_key=current_session_key,
    )
    matches = get_navigation_segments(
        initial_text,
        agent_loop=agent_loop,
        current_session_key=current_session_key,
    )
    list_control = FormattedTextControl(focusable=False)
    filter_buffer = Buffer()

    def _set_selection(index: int) -> None:
        nonlocal selected_index
        if not matches:
            selected_index = 0
            return
        selected_index = max(0, min(index, len(matches) - 1))

    def _render_results() -> None:
        fragments: list[tuple[str, str]] = []
        if matches:
            label_width = min(max(len(match.segment) for match in matches), 28)
            for idx, match in enumerate(matches):
                selected = idx == selected_index
                style = "reverse" if selected else ""
                prefix = "❯ " if selected else "  "
                suffix = " ›" if match.has_children else ""
                fragments.extend(
                    [
                        (style, prefix),
                        (style, f"{match.segment:<{label_width}}"),
                        (style, suffix),
                        (style, " "),
                        (style, match.description),
                        ("", "\n"),
                    ]
                )
        else:
            fragments.append(("italic", "  No matching commands\n"))
        fragments.append(("dim", "  ↑/↓ navigate • Enter descend/select • Esc cancel"))
        list_control.text = fragments

    def _refresh() -> None:
        nonlocal matches
        query = filter_buffer.text
        matches = get_navigation_segments(
            query,
            agent_loop=agent_loop,
            current_session_key=current_session_key,
        )
        _set_selection(0)
        _render_results()

    def _accept() -> bool:
        if not matches:
            result["value"] = None
            app.exit()
            return True

        selected = matches[selected_index]
        if selected.has_children:
            filter_buffer.text = " ".join(selected.path) + " "
            filter_buffer.cursor_position = len(filter_buffer.text)
            _refresh()
            return True

        result["value"] = selected.label
        app.exit()
        return True

    filter_buffer.accept_handler = lambda _buf: _accept()
    filter_buffer.on_text_changed += lambda _buf: _refresh()
    filter_buffer.text = initial_text

    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:
        if matches:
            _set_selection(selected_index - 1)
            _render_results()

    @kb.add("down")
    def _down(event) -> None:
        if matches:
            _set_selection(selected_index + 1)
            _render_results()

    @kb.add("c-p")
    def _prev(event) -> None:
        _up(event)

    @kb.add("c-n")
    def _next(event) -> None:
        _down(event)

    @kb.add("enter")
    def _enter(event) -> None:
        _accept()

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:
        result["value"] = None
        event.app.exit()

    _refresh()
    filter_field = Window(content=FormattedTextControl(text=[("bold", "/")]), width=1)
    filter_input = Window(
        content=BufferControl(buffer=filter_buffer, input_processors=[PaletteInputProcessor()]),
        height=1,
    )
    results = Window(content=list_control, wrap_lines=False, height=Dimension(min=6))
    root_container = Box(
        Frame(
            HSplit(
                [
                    Window(
                        content=FormattedTextControl(
                            text=[("dim", "Type to filter slash commands")]
                        ),
                        height=1,
                    ),
                    Window(height=1, char=" "),
                    VSplit(
                        [
                            Window(
                                content=FormattedTextControl(text=[("bold", "Filter: ")]),
                                width=8,
                                dont_extend_width=True,
                            ),
                            filter_field,
                            filter_input,
                        ],
                        height=1,
                    ),
                    Window(height=1, char=" "),
                    results,
                ]
            ),
            title="Slash commands",
        ),
        padding=1,
    )
    app = Application(
        layout=Layout(root_container, focused_element=filter_input),
        key_bindings=kb,
        full_screen=False,
    )
    await app.run_async()
    return result["value"]
