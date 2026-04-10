"""Tests for FeishuChannel tool hint code block formatting."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pytest import mark

from aeloon.channels.feishu import FeishuChannel
from aeloon.core.bus.events import OutboundMessage


@pytest.fixture
def mock_feishu_channel():
    """Create a FeishuChannel with mocked client."""
    config = MagicMock()
    config.app_id = "test_app_id"
    config.app_secret = "test_app_secret"
    config.encrypt_key = None
    config.verification_token = None
    bus = MagicMock()
    channel = FeishuChannel(config, bus)
    channel._client = MagicMock()  # Simulate initialized client
    return channel


@mark.asyncio
async def test_tool_hint_sends_code_message(mock_feishu_channel):
    """Tool hint messages should be sent as interactive cards with code blocks."""
    msg = OutboundMessage(
        channel="feishu",
        chat_id="oc_123456",
        content='web_search("test query")',
        metadata={"_tool_hint": True},
    )

    with patch.object(mock_feishu_channel, "_send_message_sync") as mock_send:
        await mock_feishu_channel.send(msg)

        # Verify interactive message with card was sent
        assert mock_send.call_count == 1
        call_args = mock_send.call_args[0]
        receive_id_type, receive_id, msg_type, content = call_args

        assert receive_id_type == "chat_id"
        assert receive_id == "oc_123456"
        assert msg_type == "interactive"

        # Parse content to verify card structure
        card = json.loads(content)
        assert card["config"]["wide_screen_mode"] is True
        assert len(card["elements"]) == 1
        assert card["elements"][0]["tag"] == "markdown"
        # Check that code block is properly formatted with language hint
        expected_md = '**Tool Calls**\n\n```text\nweb_search("test query")\n```'
        assert card["elements"][0]["content"] == expected_md


@mark.asyncio
async def test_tool_hint_empty_content_does_not_send(mock_feishu_channel):
    """Empty tool hint messages should not be sent."""
    msg = OutboundMessage(
        channel="feishu",
        chat_id="oc_123456",
        content="   ",  # whitespace only
        metadata={"_tool_hint": True},
    )

    with patch.object(mock_feishu_channel, "_send_message_sync") as mock_send:
        await mock_feishu_channel.send(msg)

        # Should not send any message
        mock_send.assert_not_called()


@mark.asyncio
async def test_tool_hint_without_metadata_sends_as_normal(mock_feishu_channel):
    """Regular messages without _tool_hint should use normal formatting."""
    msg = OutboundMessage(
        channel="feishu", chat_id="oc_123456", content="Hello, world!", metadata={}
    )

    with patch.object(mock_feishu_channel, "_send_message_sync") as mock_send:
        await mock_feishu_channel.send(msg)

        # Should send as text message (detected format)
        assert mock_send.call_count == 1
        call_args = mock_send.call_args[0]
        _, _, msg_type, content = call_args
        assert msg_type == "text"
        assert json.loads(content) == {"text": "Hello, world!"}


@mark.asyncio
async def test_tool_hint_multiple_tools_in_one_message(mock_feishu_channel):
    """Multiple tool calls should be displayed each on its own line in a code block."""
    msg = OutboundMessage(
        channel="feishu",
        chat_id="oc_123456",
        content='web_search("query"), read_file("/path/to/file")',
        metadata={"_tool_hint": True},
    )

    with patch.object(mock_feishu_channel, "_send_message_sync") as mock_send:
        await mock_feishu_channel.send(msg)

        call_args = mock_send.call_args[0]
        msg_type = call_args[2]
        content = json.loads(call_args[3])
        assert msg_type == "interactive"
        # Each tool call should be on its own line
        expected_md = (
            '**Tool Calls**\n\n```text\nweb_search("query"),\nread_file("/path/to/file")\n```'
        )
        assert content["elements"][0]["content"] == expected_md


@mark.asyncio
async def test_tool_hint_keeps_commas_inside_arguments(mock_feishu_channel):
    """Commas inside a single tool argument must not be split onto a new line."""
    msg = OutboundMessage(
        channel="feishu",
        chat_id="oc_123456",
        content='web_search("foo, bar"), read_file("/path/to/file")',
        metadata={"_tool_hint": True},
    )

    with patch.object(mock_feishu_channel, "_send_message_sync") as mock_send:
        await mock_feishu_channel.send(msg)

        content = json.loads(mock_send.call_args[0][3])
        expected_md = (
            '**Tool Calls**\n\n```text\nweb_search("foo, bar"),\nread_file("/path/to/file")\n```'
        )
        assert content["elements"][0]["content"] == expected_md


@mark.asyncio
async def test_progress_messages_are_throttled_per_chat(mock_feishu_channel):
    """Plain progress messages sent too close together should be suppressed."""
    first = OutboundMessage(
        channel="feishu",
        chat_id="oc_123456",
        content="step 1",
        metadata={"_progress": True},
    )
    second = OutboundMessage(
        channel="feishu",
        chat_id="oc_123456",
        content="step 2",
        metadata={"_progress": True},
    )

    with (
        patch.object(mock_feishu_channel, "_send_message_sync") as mock_send,
        patch.object(mock_feishu_channel, "_now_monotonic", side_effect=[100.0, 100.0, 100.2]),
    ):
        await mock_feishu_channel.send(first)
        await mock_feishu_channel.send(second)

    assert mock_send.call_count == 1
    sent_body = json.loads(mock_send.call_args[0][3])
    assert sent_body == {"text": "step 1"}


@mark.asyncio
async def test_final_message_bypasses_progress_throttle(mock_feishu_channel):
    """Final messages should still send even right after a progress update."""
    progress = OutboundMessage(
        channel="feishu",
        chat_id="oc_123456",
        content="thinking",
        metadata={"_progress": True},
    )
    final = OutboundMessage(
        channel="feishu",
        chat_id="oc_123456",
        content="done",
        metadata={},
    )

    with (
        patch.object(mock_feishu_channel, "_send_message_sync") as mock_send,
        patch.object(mock_feishu_channel, "_now_monotonic", return_value=100.0),
    ):
        await mock_feishu_channel.send(progress)
        await mock_feishu_channel.send(final)

    assert mock_send.call_count == 2
    first_body = json.loads(mock_send.call_args_list[0][0][3])
    second_body = json.loads(mock_send.call_args_list[1][0][3])
    assert first_body == {"text": "thinking"}
    assert second_body == {"text": "done"}


@mark.asyncio
async def test_send_rate_limit_waits_between_same_chat_messages(mock_feishu_channel):
    """All Feishu sends in the same chat should be spaced by the min interval."""
    first = OutboundMessage(channel="feishu", chat_id="oc_123456", content="one", metadata={})
    second = OutboundMessage(channel="feishu", chat_id="oc_123456", content="two", metadata={})
    sleep_mock = MagicMock()

    async def _fake_sleep(delay: float) -> None:
        sleep_mock(delay)

    with (
        patch.object(mock_feishu_channel, "_send_message_sync") as mock_send,
        patch.object(mock_feishu_channel, "_now_monotonic", side_effect=[100.0, 100.2, 101.0]),
        patch("aeloon.channels.feishu.asyncio.sleep", side_effect=_fake_sleep),
    ):
        await mock_feishu_channel.send(first)
        await mock_feishu_channel.send(second)

    assert mock_send.call_count == 2
    sleep_mock.assert_called_once()
    assert sleep_mock.call_args[0][0] == pytest.approx(0.8, rel=1e-3)


@mark.asyncio
async def test_send_rate_limit_is_isolated_per_chat(mock_feishu_channel):
    """Different chats should not block each other's first send."""
    first = OutboundMessage(channel="feishu", chat_id="oc_chat1", content="one", metadata={})
    second = OutboundMessage(channel="feishu", chat_id="oc_chat2", content="two", metadata={})

    async def _unexpected_sleep(_delay: float) -> None:
        raise AssertionError("sleep should not be called for different chats")

    with (
        patch.object(mock_feishu_channel, "_send_message_sync") as mock_send,
        patch.object(mock_feishu_channel, "_now_monotonic", side_effect=[100.0, 100.1]),
        patch("aeloon.channels.feishu.asyncio.sleep", side_effect=_unexpected_sleep),
    ):
        await mock_feishu_channel.send(first)
        await mock_feishu_channel.send(second)

    assert mock_send.call_count == 2
