"""Status provider for the ACP Bridge plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aeloon.plugins._sdk.types import StatusContext, StatusSegment

if TYPE_CHECKING:
    from aeloon.plugins.acp_bridge.service import ACPConnectionService


def make_acp_status_provider(service: ACPConnectionService) -> type:
    """Create a status provider function bound to the connection service."""

    def acp_status(ctx: StatusContext) -> StatusSegment:
        health = service.health_check()
        state = health.get("state", "disconnected")

        if state == "connected":
            profile = health.get("profile", "unknown")
            text = f"ACP: {profile}"
            style = "bold ansigreen"
        elif state == "connecting":
            text = "ACP: connecting..."
            style = "ansiyellow"
        elif state == "error":
            text = "ACP: error"
            style = "bold ansired"
        else:
            return StatusSegment(text="", style="", priority=0)

        return StatusSegment(text=text, style=style, priority=10)

    return acp_status  # type: ignore[return-value]
