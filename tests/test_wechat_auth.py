import json
import sys
import types
from pathlib import Path

import pytest

from aeloon.channels.wechat_ilink.auth import (
    download_qr_image,
    fetch_qrcode,
    load_all_credentials,
    poll_qrcode_status,
    poll_qrcode_until_confirmed,
    remove_all_credentials,
    save_credentials,
)
from aeloon.channels.wechat_ilink.types import Credentials


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *_args, **_kwargs):
        return _FakeResponse(self._responses.pop(0))


@pytest.mark.asyncio
async def test_fetch_qrcode_reads_qrcode_img_content(monkeypatch) -> None:
    monkeypatch.setattr(
        "aeloon.channels.wechat_ilink.auth.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeClient(
            [{"ret": 0, "qrcode": "qr-1", "qrcode_img_content": "aGVsbG8=", "expired_time": 123}]
        ),
    )

    result = await fetch_qrcode()

    assert result["qrcode"] == "qr-1"
    assert result["qrcode_img_content"] == "aGVsbG8="


@pytest.mark.asyncio
async def test_poll_qrcode_status_reads_top_level_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        "aeloon.channels.wechat_ilink.auth.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeClient(
            [
                {
                    "ret": 0,
                    "status": "confirmed",
                    "bot_token": "bot-token",
                    "ilink_bot_id": "bot-id",
                    "baseurl": "https://ilinkai.weixin.qq.com",
                    "ilink_user_id": "user-id",
                }
            ]
        ),
    )

    result = await poll_qrcode_status("qr-1")

    assert result["status"] == "confirmed"
    assert result["bot_token"] == "bot-token"
    assert result["ilink_bot_id"] == "bot-id"


@pytest.mark.asyncio
async def test_poll_qrcode_until_confirmed_builds_credentials_from_top_level(monkeypatch) -> None:
    async def _fake_poll(_qrcode: str, timeout: float = 40.0) -> dict:
        return {
            "status": "confirmed",
            "bot_token": "bot-token",
            "ilink_bot_id": "bot-id",
            "baseurl": "https://ilinkai.weixin.qq.com",
            "ilink_user_id": "user-id",
        }

    monkeypatch.setattr("aeloon.channels.wechat_ilink.auth.poll_qrcode_status", _fake_poll)

    credentials = await poll_qrcode_until_confirmed("qr-1", overall_timeout=5.0, poll_interval=0.0)

    assert credentials.bot_token == "bot-token"
    assert credentials.ilink_bot_id == "bot-id"


@pytest.mark.asyncio
async def test_download_qr_image_renders_png_via_qrcode_library(
    monkeypatch, tmp_path: Path
) -> None:
    saved = {}

    class _FakeImage:
        def save(self, path: Path) -> None:
            saved["path"] = path
            path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    fake_module = types.SimpleNamespace(
        make=lambda content: saved.setdefault("content", content) or _FakeImage()
    )

    # make needs to return image; avoid lambda short-circuit ambiguity
    def _fake_make(content):
        saved["content"] = content
        return _FakeImage()

    fake_module.make = _fake_make
    monkeypatch.setitem(sys.modules, "qrcode", fake_module)

    output = tmp_path / "qr.png"

    await download_qr_image("wechat://qr-content", output)

    assert saved["content"] == "wechat://qr-content"
    assert saved["path"] == output
    assert output.read_bytes().startswith(b"\x89PNG")


def test_save_load_remove_credentials_use_same_directory(tmp_path: Path) -> None:
    credentials = Credentials(
        bot_token="bot-token",
        ilink_bot_id="bot-id",
        base_url="https://ilinkai.weixin.qq.com",
        ilink_user_id="user-id",
    )

    path = save_credentials(credentials, tmp_path)
    assert path.exists()
    assert json.loads(path.read_text())["bot_token"] == "bot-token"

    loaded = load_all_credentials(tmp_path)
    assert [cred.ilink_bot_id for cred in loaded] == ["bot-id"]

    removed = remove_all_credentials(tmp_path)
    assert removed == 1
    assert load_all_credentials(tmp_path) == []
