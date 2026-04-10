from aeloon.core.agent.context import ContextBuilder
from aeloon.core.session.manager import Session, SessionManager


def _mk_manager(tmp_path) -> SessionManager:
    return SessionManager(tmp_path)


def test_save_turn_skips_multimodal_user_when_only_runtime_context(tmp_path) -> None:
    manager = _mk_manager(tmp_path)
    session = Session(key="test:runtime-only")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    manager.save_turn(
        session,
        [{"role": "user", "content": [{"type": "text", "text": runtime}]}],
        skip=0,
        runtime_context_tag=ContextBuilder._RUNTIME_CONTEXT_TAG,
    )
    assert session.messages == []


def test_save_turn_keeps_image_placeholder_with_path_after_runtime_strip(tmp_path) -> None:
    manager = _mk_manager(tmp_path)
    session = Session(key="test:image")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    manager.save_turn(
        session,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": runtime},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                        "_meta": {"path": "/media/feishu/photo.jpg"},
                    },
                ],
            }
        ],
        skip=0,
        runtime_context_tag=ContextBuilder._RUNTIME_CONTEXT_TAG,
    )
    assert session.messages[0]["content"] == [
        {"type": "text", "text": "[image: /media/feishu/photo.jpg]"}
    ]


def test_save_turn_keeps_image_placeholder_without_meta(tmp_path) -> None:
    manager = _mk_manager(tmp_path)
    session = Session(key="test:image-no-meta")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    manager.save_turn(
        session,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": runtime},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
        skip=0,
        runtime_context_tag=ContextBuilder._RUNTIME_CONTEXT_TAG,
    )
    assert session.messages[0]["content"] == [{"type": "text", "text": "[image]"}]


def test_save_turn_keeps_tool_results_under_16k(tmp_path) -> None:
    manager = _mk_manager(tmp_path)
    session = Session(key="test:tool-result")
    content = "x" * 12_000

    manager.save_turn(
        session,
        [{"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": content}],
        skip=0,
    )

    assert session.messages[0]["content"] == content
