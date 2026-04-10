"""StatusPlugin — Built-in status bar plugin.

Registers a status provider that shows the current model name and
context window usage in the CLI bottom toolbar.  This replaces the
previous hardcoded ``_build_bottom_toolbar`` in ``cli/commands.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aeloon.plugins._sdk import Plugin
from aeloon.plugins._sdk.types import StatusContext, StatusSegment

if TYPE_CHECKING:
    from aeloon.plugins._sdk.api import PluginAPI


class StatusPlugin(Plugin):
    """Built-in status bar — model name + context window usage."""

    def register(self, api: PluginAPI) -> None:
        api.register_status_provider("model_context", self._get_status, priority=0)

    async def activate(self, api: PluginAPI) -> None:
        pass

    async def deactivate(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Status provider (sync — called during prompt_toolkit render)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_status(ctx: StatusContext) -> list[StatusSegment]:
        """Return model and context segments."""
        # --- Model segment ---
        model_value = ctx.model
        if len(model_value) > 30:
            model_value = f"{model_value[:29]}\u2026"

        # --- Context segment ---
        total = ctx.context_tokens_total
        used = ctx.context_tokens_used
        if total > 0:
            ratio = used / total * 100
            pct_text = f"{ratio:.0f}%"
        else:
            ratio = 0.0
            pct_text = "0%"

        context_text = f"{used}/{total} ({pct_text})"

        if ratio >= 90:
            context_style = "bold ansired"
        elif ratio >= 75:
            context_style = "ansiyellow"
        else:
            context_style = ""

        return [
            StatusSegment(text=f"Model: {model_value}", style="bold", priority=10),
            StatusSegment(text=f"Context: {context_text}", style=context_style, priority=5),
        ]
