"""Credential loading helpers for the native WeChat channel."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from aeloon.core.config.loader import get_aeloon_home

from .types import Credentials


def default_accounts_dir() -> Path:
    """Return the default iLink credentials directory (legacy path)."""
    return Path.home() / ".weclaw" / "accounts"


def new_accounts_dir() -> Path:
    """Return the new aeloon credentials directory."""
    return get_aeloon_home() / "accounts" / "wechat"


def get_qr_code_dir() -> Path:
    """Return the directory for saving QR code images during login."""
    qr_dir = get_aeloon_home() / "media" / "wechat-login"
    qr_dir.mkdir(parents=True, exist_ok=True)
    return qr_dir


def load_all_credentials(accounts_dir: str | Path | None = None) -> list[Credentials]:
    """
    Load all account credentials from disk.

    Supports dual-directory compatibility:
    - First tries ~/.aeloon/accounts/wechat/
    - If empty, falls back to ~/.weclaw/accounts/
    """
    # If explicit directory provided, use it
    if accounts_dir:
        directory = Path(accounts_dir).expanduser()
    else:
        # Try new directory first
        directory = new_accounts_dir()
        if not directory.exists() or not list(directory.glob("*.json")):
            # Fall back to legacy directory
            directory = default_accounts_dir()

    if not directory.exists():
        return []

    credentials: list[Credentials] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        creds = Credentials.from_dict(data)
        if creds.bot_token and creds.ilink_bot_id:
            credentials.append(creds)
    return credentials


def has_valid_credentials(accounts_dir: str | Path | None = None) -> bool:
    """Check if there are valid credentials available."""
    return len(load_all_credentials(accounts_dir)) > 0


def get_first_credential(accounts_dir: str | Path | None = None) -> Credentials | None:
    """Get the first available credential."""
    creds = load_all_credentials(accounts_dir)
    return creds[0] if creds else None


async def fetch_qrcode() -> dict[str, Any]:
    """
    Fetch a QR code from iLink for WeChat login.

    Endpoint: GET https://ilinkai.weixin.qq.com/ilink/bot/get_bot_qrcode?bot_type=3

    Returns:
        dict with keys:
        - qrcode: The QR code ID (e.g., "qr_1234567890")
        - qrcode_img_content: QR payload content used to render a scannable QR code
        - expired_time: Expiration timestamp
    """
    url = "https://ilinkai.weixin.qq.com/ilink/bot/get_bot_qrcode"
    params = {"bot_type": "3"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        if data.get("ret") != 0:
            raise RuntimeError(f"iLink API error: {data.get('errmsg', 'Unknown error')}")

        return {
            "qrcode": data.get("qrcode", ""),
            "qrcode_img_content": data.get("qrcode_img_content", ""),
            "expired_time": data.get("expired_time", 0),
        }


async def download_qr_image(qrcode_img_content: str, output_path: Path) -> None:
    """Render qrcode_img_content into a standard PNG image file."""
    if not qrcode_img_content:
        raise RuntimeError("Missing qrcode_img_content in QR code response")
    try:
        import qrcode
    except ImportError as exc:
        raise RuntimeError("qrcode package is required for WeChat login QR rendering") from exc

    image = qrcode.make(qrcode_img_content)
    image.save(output_path)
    os.chmod(output_path, 0o600)


async def poll_qrcode_status(qrcode: str, timeout: float = 40.0) -> dict[str, Any]:
    """
    Poll the QR code scan status from iLink.

    Endpoint: GET https://ilinkai.weixin.qq.com/ilink/bot/get_qrcode_status?qrcode=<id>

    Status values:
    - wait: Waiting for user to scan
    - scaned: User has scanned, waiting for confirmation
    - confirmed: Login successful, credentials available
    - expired: QR code has expired

    Args:
        qrcode: The QR code ID returned by fetch_qrcode()
        timeout: Long poll timeout in seconds (default: 40s)

    Returns:
        dict with keys:
        - status: One of "wait", "scaned", "confirmed", "expired"
        - bot_token / ilink_bot_id / baseurl / ilink_user_id when confirmed
    """
    url = "https://ilinkai.weixin.qq.com/ilink/bot/get_qrcode_status"
    params = {"qrcode": qrcode}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if data.get("ret") != 0:
            raise RuntimeError(f"iLink API error: {data.get('errmsg', 'Unknown error')}")

        status = data.get("status", "")
        result: dict[str, Any] = {"status": status}

        # If confirmed, extract top-level credential fields.
        if status == "confirmed":
            result.update(
                {
                    "bot_token": data.get("bot_token", ""),
                    "ilink_bot_id": data.get("ilink_bot_id", ""),
                    "baseurl": data.get("baseurl", ""),
                    "ilink_user_id": data.get("ilink_user_id", ""),
                }
            )

        return result


async def poll_qrcode_until_confirmed(
    qrcode: str,
    overall_timeout: float = 300.0,  # 5 minutes
    poll_interval: float = 2.0,
) -> Credentials:
    """
    Continuously poll QR code status until confirmed, expired, or timeout.

    Args:
        qrcode: The QR code ID
        overall_timeout: Maximum total time to wait for login
        poll_interval: Time between polls when status is "wait"

    Returns:
        Credentials object on successful login

    Raises:
        TimeoutError: If login times out
        RuntimeError: If QR code expires or other error occurs
    """
    start_time = time.monotonic()

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= overall_timeout:
            raise TimeoutError(f"Login timed out after {overall_timeout} seconds")

        try:
            result = await poll_qrcode_status(qrcode, timeout=40.0)
            status = result.get("status", "")

            if status == "confirmed":
                creds_data = {
                    "bot_token": result.get("bot_token", ""),
                    "ilink_bot_id": result.get("ilink_bot_id", ""),
                    "baseurl": result.get("baseurl", ""),
                    "ilink_user_id": result.get("ilink_user_id", ""),
                }
                if not creds_data["bot_token"] or not creds_data["ilink_bot_id"]:
                    raise RuntimeError("Login confirmed but no credentials received")

                credentials = Credentials.from_dict(creds_data)
                if not credentials.bot_token or not credentials.ilink_bot_id:
                    raise RuntimeError("Login confirmed but credentials are incomplete")

                return credentials

            elif status == "expired":
                raise RuntimeError("QR code has expired")

            elif status in ("wait", "scaned"):
                # Continue polling
                await asyncio.sleep(poll_interval)
                continue

            else:
                logger.warning(f"Unknown QR code status: {status}")
                await asyncio.sleep(poll_interval)

        except asyncio.TimeoutError:
            # Long poll timeout, continue polling
            continue
        except httpx.HTTPError as e:
            logger.warning(f"Network error during QR code polling: {e}")
            await asyncio.sleep(poll_interval)


@dataclass
class LoginResult:
    """Result of a login attempt."""

    success: bool
    credentials: Credentials | None = None
    error: str | None = None
    qrcode: str | None = None


def save_credentials(credentials: Credentials, accounts_dir: str | Path | None = None) -> Path:
    """
    Save credentials to disk.

    Args:
        credentials: The credentials to save
        accounts_dir: Target directory (defaults to new aeloon directory)

    Returns:
        Path to the saved credential file
    """
    directory = Path(accounts_dir).expanduser() if accounts_dir else new_accounts_dir()
    directory.mkdir(parents=True, exist_ok=True)

    # Use ilink_bot_id as filename
    filename = f"{credentials.ilink_bot_id}.json"
    file_path = directory / filename

    # Prepare data for saving
    data = {
        "bot_token": credentials.bot_token,
        "ilink_bot_id": credentials.ilink_bot_id,
        "baseurl": credentials.base_url,
        "ilink_user_id": credentials.ilink_user_id,
    }

    # Write with restricted permissions
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(file_path, 0o600)

    logger.info("Saved credentials to {}", file_path)
    return file_path


def remove_credentials(ilink_bot_id: str, accounts_dir: str | Path | None = None) -> bool:
    """
    Remove credentials for a bot.

    Args:
        ilink_bot_id: The bot ID to remove
        accounts_dir: Directory to search (defaults to both new and legacy directories)

    Returns:
        True if credentials were found and removed, False otherwise
    """
    if accounts_dir:
        directories = [Path(accounts_dir).expanduser()]
    else:
        # Search both new and legacy directories
        directories = [new_accounts_dir(), default_accounts_dir()]

    filename = f"{ilink_bot_id}.json"
    removed = False

    for directory in directories:
        file_path = directory / filename
        if file_path.exists():
            file_path.unlink()
            logger.info("Removed credentials at {}", file_path)
            removed = True

    return removed


def remove_all_credentials(accounts_dir: str | Path | None = None) -> int:
    """
    Remove all credentials.

    Returns:
        Number of credential files removed
    """
    if accounts_dir:
        directories = [Path(accounts_dir).expanduser()]
    else:
        directories = [new_accounts_dir(), default_accounts_dir()]

    count = 0
    for directory in directories:
        if directory.exists():
            for file_path in directory.glob("*.json"):
                file_path.unlink()
                logger.info("Removed credentials at {}", file_path)
                count += 1

    return count
