import pytest

from aeloon.core.agent.tools.message import MessageTool
from aeloon.core.agent.turn import TurnContext


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
async def test_message_tool_accepts_media_only() -> None:
    sent = []

    async def _send(msg) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send, default_channel="wechat", default_chat_id="user-1")

    result = await tool.execute(media=["/tmp/demo.png"])

    assert result == "Message sent to wechat:user-1 with 1 attachments"
    assert sent[0].content == ""
    assert sent[0].media == ["/tmp/demo.png"]


@pytest.mark.asyncio
async def test_message_tool_rejects_empty_content_and_media() -> None:
    tool = MessageTool()
    result = await tool.execute()
    assert result == "Error: Either content or media must be provided"


def test_message_tool_description_mentions_local_file_sending() -> None:
    tool = MessageTool()

    assert "send, re-send, return, or forward a local image/file" in tool.description
    assert "media=[path]" in tool.description
    assert (
        "primary way to send a local image or file back to the user"
        in (tool.parameters["properties"]["media"]["description"])
    )


def test_message_tool_on_turn_start_updates_context_and_resets_sent_flag() -> None:
    tool = MessageTool()
    tool._sent_in_turn = True

    ctx = TurnContext(
        channel="feishu",
        chat_id="chat-1",
        message_id="mid-1",
        session_key="feishu:chat-1",
        sender_id="user-1",
    )
    tool.on_turn_start(ctx)

    assert tool.sent_in_turn is False
    assert tool._default_channel == "feishu"
    assert tool._default_chat_id == "chat-1"
    assert tool._default_message_id == "mid-1"
