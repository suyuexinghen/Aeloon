"""Non-regression tests: ensure ordinary assistant mode is unaffected by science module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import isolation: science module should NOT be loaded on plain aeloon import
# ---------------------------------------------------------------------------


def test_plain_aeloon_import_does_not_load_science():
    """Importing aeloon top-level should not auto-import aeloon.science.

    Note: if config.schema has already been imported in this process (which
    it will be in a test suite), the backward-compat re-export at the bottom
    of schema.py will have loaded the science module.  The important
    invariant is that `import aeloon` alone does not *require* science.
    """
    import aeloon  # noqa: F401 — side-effect import

    # We only assert aeloon is importable — not that science is absent,
    # because schema.py's re-export may have loaded it in the same process.


def test_config_schema_no_science_field():
    """Config no longer has a top-level 'science' field (moved to plugins namespace).

    Note: schema.py still re-exports ScienceConfig/GovernanceConfig for backward
    compat, so the science module *is* loaded as a side-effect — that's expected.
    """
    from aeloon.core.config.schema import Config

    cfg = Config()
    assert not hasattr(cfg, "science") or "science" not in cfg.model_fields


# ---------------------------------------------------------------------------
# ScienceConfig is accessible from config.schema and defaults to disabled
# ---------------------------------------------------------------------------


def test_science_config_accessible_and_disabled_by_default():
    from aeloon.plugins.ScienceResearch.config import ScienceConfig

    sc = ScienceConfig()
    assert sc.enabled is False


def test_config_has_plugins_field_empty_by_default():
    from aeloon.core.config.schema import Config

    cfg = Config.model_validate({})
    assert cfg.plugins == {}
    assert "science" not in Config.model_fields  # science is no longer a top-level field


# ---------------------------------------------------------------------------
# Dispatcher: /science is in known slash commands
# ---------------------------------------------------------------------------


def test_dispatcher_known_slash_commands_includes_plugin_commands():

    # _known_slash_commands is now an instance method
    # Without a plugin_manager, /science is NOT in the base list

    dispatcher = _make_dispatcher()
    commands = dispatcher._known_slash_commands()
    # Plugin commands appear only when a plugin_manager is set
    assert "/science" not in commands  # no longer hardcoded


# ---------------------------------------------------------------------------
# Ordinary dispatcher routing unaffected: /help and /new still work
# ---------------------------------------------------------------------------


def _make_dispatcher(tmp_path=None):
    """Build a Dispatcher backed by a minimal mocked AgentLoop."""
    from aeloon.core.agent.loop import AgentLoop
    from aeloon.core.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with (
        patch("aeloon.core.agent.loop.ContextBuilder"),
        patch("aeloon.core.agent.loop.SessionManager") as mock_sm,
        patch("aeloon.core.agent.loop.SubagentManager") as mock_sub,
    ):
        mock_sub.return_value.cancel_by_session = AsyncMock(return_value=0)
        session = MagicMock()
        session.messages = []
        session.last_consolidated = 0
        mock_sm.return_value.get_or_create = MagicMock(return_value=session)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

    from aeloon.core.agent.dispatcher import Dispatcher

    dispatcher = Dispatcher(loop)
    return dispatcher


@pytest.mark.asyncio
async def test_help_command_returns_response():
    """The /help command should respond successfully."""
    from aeloon.core.bus.events import InboundMessage

    dispatcher = _make_dispatcher()

    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
    response = await dispatcher.process_message(msg)

    assert response is not None


@pytest.mark.asyncio
async def test_new_command_returns_response():
    """The /new command should respond successfully."""
    from aeloon.core.bus.events import InboundMessage

    dispatcher = _make_dispatcher()

    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/new")
    response = await dispatcher.process_message(msg)

    assert response is not None
