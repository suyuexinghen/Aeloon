"""Channel authentication utilities for WeChat and Feishu."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from aeloon.core.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from aeloon.channels.manager import ChannelManager


class WeChatAuthManager:
    """Manages WeChat QR code authentication flow."""

    def __init__(self) -> None:
        self._login_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._login_status: dict[tuple[str, str], dict] = {}

    @staticmethod
    def set_enabled(enabled: bool) -> None:
        """Persist ``channels.wechat.enabled`` in config.json."""
        from aeloon.core.config.loader import get_config_path

        config_path = get_config_path()
        if not config_path.exists():
            return

        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            channels = data.setdefault("channels", {})
            wechat = channels.setdefault("wechat", {})
            wechat["enabled"] = enabled
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to persist wechat.enabled={}: {}", enabled, exc)

    def has_pending_login(self, request_channel: str, request_chat_id: str) -> bool:
        """Check if there's a pending WeChat login for the given channel/chat."""
        key = (request_channel, request_chat_id)
        return key in self._login_tasks and not self._login_tasks[key].done()

    def get_login_status(self, request_channel: str, request_chat_id: str) -> dict | None:
        """Get the status of a pending WeChat login."""
        key = (request_channel, request_chat_id)
        return self._login_status.get(key)

    def update_login_status(
        self,
        request_channel: str,
        request_chat_id: str,
        updates: dict,
    ) -> None:
        """Merge updates into the stored WeChat login status."""
        key = (request_channel, request_chat_id)
        if key not in self._login_status:
            self._login_status[key] = {}
        self._login_status[key].update(updates)

    def clear_login_status(self, request_channel: str, request_chat_id: str) -> None:
        """Remove stored WeChat login status for one request target."""
        self._login_status.pop((request_channel, request_chat_id), None)

    def register_login_task(
        self,
        request_channel: str,
        request_chat_id: str,
        task: asyncio.Task,
        status: dict,
    ) -> None:
        """Register a WeChat login task and its status."""
        key = (request_channel, request_chat_id)
        self._login_tasks[key] = task
        self._login_status[key] = status

        # Clean up when task completes
        def _cleanup(t: asyncio.Task) -> None:
            if key in self._login_tasks and self._login_tasks[key] is t:
                del self._login_tasks[key]
            # Keep status for a while to allow status queries

        task.add_done_callback(_cleanup)

    def cancel_login(self, request_channel: str, request_chat_id: str) -> bool:
        """Cancel a pending WeChat login task."""
        key = (request_channel, request_chat_id)
        task = self._login_tasks.get(key)
        if task and not task.done():
            task.cancel()
            self.clear_login_status(request_channel, request_chat_id)
            return True
        return False

    async def cancel_all_logins(self) -> int:
        """Cancel all pending WeChat login tasks. Returns count cancelled."""
        count = 0
        for key, task in list(self._login_tasks.items()):
            if not task.done():
                task.cancel()
                count += 1
        self._login_tasks.clear()
        self._login_status.clear()
        return count

    @staticmethod
    def render_ascii_qrcode(data: str) -> str | None:
        """Render QR payload as ASCII art using the ``qrcode`` library."""
        if not data:
            return None
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(data)
            qr.make(fit=True)
            from io import StringIO

            buf = StringIO()
            qr.print_ascii(out=buf, invert=True)
            return buf.getvalue().rstrip()
        except ImportError:
            return None


class FeishuAuthManager:
    """Manages Feishu app credential authentication."""

    def __init__(self, channel_manager: ChannelManager | None = None) -> None:
        self._channel_manager = channel_manager

    def set_channel_manager(self, channel_manager: ChannelManager | None) -> None:
        """Set the channel manager reference."""
        self._channel_manager = channel_manager

    def has_credentials(self) -> bool:
        """Check if Feishu credentials are configured."""
        config = self.get_config()
        if isinstance(config, dict):
            return bool(config.get("app_id", ""))
        if config:
            return bool(getattr(config, "app_id", ""))
        return False

    def get_config(self) -> Any:
        """Get Feishu configuration from channel manager."""
        if self._channel_manager is None:
            return None
        return self._channel_manager._get_channel_config("feishu")

    @staticmethod
    async def validate_credentials(app_id: str, app_secret: str) -> bool:
        """Validate Feishu credentials by calling the bot info API."""
        try:
            import lark_oapi as lark
            from lark_oapi.api.bot.v3 import GetBotInfoRequest

            client = (
                lark.Client.builder()
                .app_id(app_id)
                .app_secret(app_secret)
                .log_level(lark.LogLevel.ERROR)
                .build()
            )

            request = GetBotInfoRequest.builder().build()
            response = client.bot.v3.info.get(request)

            return response.success()
        except ImportError:
            # lark_oapi not installed, skip validation
            logger.warning("lark_oapi not installed, skipping Feishu credential validation")
            return True
        except Exception as exc:
            logger.warning("Feishu credential validation failed: {}", exc)
            return False

    @staticmethod
    def set_credentials(app_id: str, app_secret: str, enabled: bool = True) -> None:
        """Persist Feishu credentials in config.json."""
        from aeloon.core.config.loader import get_config_path

        config_path = get_config_path()
        if not config_path.exists():
            return

        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            channels = data.setdefault("channels", {})
            feishu = channels.setdefault("feishu", {})
            feishu["enabled"] = enabled
            feishu["app_id"] = app_id
            feishu["app_secret"] = app_secret
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to persist Feishu credentials: {}", exc)


class GatewayManager:
    """Manages the aeloon gateway background process."""

    @staticmethod
    def is_running() -> bool:
        """Check if an ``aeloon gateway`` process is currently running."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "aeloon gateway"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    @staticmethod
    def stop() -> bool:
        """Kill any running ``aeloon gateway`` process. Returns True if killed."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "aeloon gateway"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
        except Exception:
            return False

        killed = False
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                killed = True
            except (ProcessLookupError, PermissionError):
                pass
        return killed

    @staticmethod
    def start_background() -> bool:
        """Start ``aeloon gateway`` as a detached background process."""
        try:
            cmd = [sys.executable, "-m", "aeloon", "gateway"]
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to start gateway: {}", exc)
            return False


class ChannelAuthHelper:
    """Helper class that combines all channel auth managers."""

    def __init__(self, channel_manager: ChannelManager | None = None) -> None:
        self.wechat = WeChatAuthManager()
        self.feishu = FeishuAuthManager(channel_manager)
        self.gateway = GatewayManager()
        self._channel_manager = channel_manager

    def set_channel_manager(self, channel_manager: ChannelManager | None) -> None:
        """Update channel manager reference in all managers."""
        self._channel_manager = channel_manager
        self.feishu.set_channel_manager(channel_manager)

    def _wechat_accounts_dir(self) -> str | None:
        """Resolve the configured accounts directory for the WeChat channel, if any."""
        if self._channel_manager is None:
            return None
        channel = self._channel_manager.get_channel("wechat")
        if channel is not None:
            return getattr(channel.config, "accounts_dir", None)
        config = self._channel_manager._get_channel_config("wechat")
        if config is not None:
            if isinstance(config, dict):
                return config.get("accounts_dir")
            return getattr(config, "accounts_dir", None)
        return None

    async def handle_wechat_command(
        self,
        msg: InboundMessage,
        args: list[str],
        agent_loop: Any,
    ) -> OutboundMessage:
        """Handle /wechat slash command."""
        if not args:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Usage: /wechat login|logout|status",
            )

        subcommand = args[0].lower()

        if subcommand == "login":
            return await self._handle_wechat_login(msg, agent_loop)
        elif subcommand == "logout":
            return await self._handle_wechat_logout(msg)
        elif subcommand == "status":
            return await self._handle_wechat_status(msg)
        else:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Unknown subcommand: {subcommand}. Use: /wechat login|logout|status",
            )

    async def _handle_wechat_login(self, msg: InboundMessage, agent_loop: Any) -> OutboundMessage:
        """Handle /wechat login command."""
        from aeloon.channels.wechat_ilink.auth import (
            download_qr_image,
            fetch_qrcode,
            get_qr_code_dir,
            has_valid_credentials,
            poll_qrcode_until_confirmed,
            save_credentials,
        )

        accounts_dir = self._wechat_accounts_dir()

        # Check if already logged in
        if has_valid_credentials(accounts_dir):
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Already logged in to WeChat. Use /wechat logout first if you want to switch accounts.",
            )

        # Check if there's already a pending login for this channel/chat
        if self.wechat.has_pending_login(msg.channel, msg.chat_id):
            status = self.wechat.get_login_status(msg.channel, msg.chat_id)
            qr_status = status.get("status", "pending") if status else "pending"
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Login already in progress. Current status: {qr_status}. Please scan the previously sent QR code.",
            )

        try:
            # Step 1: Fetch QR code
            qr_data = await fetch_qrcode()
            qrcode_id = qr_data["qrcode"]
            qrcode_img_content = qr_data["qrcode_img_content"]

            # Step 2: Download QR code image for channels that support media
            qr_image_path: Path | None = None
            try:
                qr_dir = get_qr_code_dir()
                qr_image_path = qr_dir / "wechat_login.png"
                await download_qr_image(qrcode_img_content, qr_image_path)
            except Exception:
                logger.warning("Failed to download QR image, ASCII fallback will be used")
                qr_image_path = None

            # Step 3: Start background polling task
            async def _login_task():
                try:
                    self.wechat.update_login_status(
                        msg.channel,
                        msg.chat_id,
                        {"status": "waiting"},
                    )

                    # Poll until confirmed or error
                    credentials = await poll_qrcode_until_confirmed(qrcode_id)

                    # Save credentials
                    save_credentials(credentials, accounts_dir)

                    # Update status
                    self.wechat.update_login_status(
                        msg.channel,
                        msg.chat_id,
                        {
                            "status": "confirmed",
                            "ilink_bot_id": credentials.ilink_bot_id,
                        },
                    )

                    # Notify user of success
                    await agent_loop.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"✅ WeChat login successful! Bot ID: {credentials.ilink_bot_id}",
                        )
                    )

                    # Persist wechat.enabled = true in config
                    self.wechat.set_enabled(True)

                    # Start or reload wechat channel
                    if self._channel_manager:
                        reloaded = await self._channel_manager.reload_channel("wechat")
                        await agent_loop.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content=(
                                    "WeChat channel is starting with the new credentials."
                                    if reloaded
                                    else "WeChat credentials saved. The wechat channel is not enabled in config."
                                ),
                            )
                        )

                    # Ensure gateway is running
                    if not self.gateway.is_running():
                        self.gateway.start_background()

                except asyncio.CancelledError:
                    await agent_loop.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="WeChat login was cancelled.",
                        )
                    )
                    raise
                except TimeoutError:
                    self.wechat.update_login_status(
                        msg.channel, msg.chat_id, {"status": "timed_out"}
                    )
                    await agent_loop.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="❌ WeChat login timed out. Please try again with /wechat login",
                        )
                    )
                except Exception as e:
                    logger.exception("WeChat login failed")
                    self.wechat.update_login_status(
                        msg.channel,
                        msg.chat_id,
                        {"status": "failed", "error": str(e)},
                    )
                    await agent_loop.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"❌ WeChat login failed: {e}",
                        )
                    )

            # Create and register the task
            task = asyncio.create_task(_login_task())
            self.wechat.register_login_task(
                msg.channel,
                msg.chat_id,
                task,
                {
                    "qrcode": qrcode_id,
                    "status": "pending",
                    "started_at": asyncio.get_running_loop().time(),
                },
            )

            # Build response — PNG image (primary) + ASCII QR (fallback)
            qr_ascii = self.wechat.render_ascii_qrcode(qrcode_img_content)
            content_parts = ["Please scan this QR code with WeChat within 5 minutes."]
            if msg.channel == "cli" and qr_image_path and qr_image_path.exists():
                content_parts.append(f"\nImage saved to: {qr_image_path}")
            if qr_ascii:
                content_parts.append("")
                content_parts.append(f"```\n{qr_ascii}\n```")

            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="\n".join(content_parts),
                media=[str(qr_image_path)] if qr_image_path and qr_image_path.exists() else [],
            )

        except Exception as e:
            logger.exception("Failed to initiate WeChat login")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Failed to initiate WeChat login: {e}",
            )

    async def _handle_wechat_logout(self, msg: InboundMessage) -> OutboundMessage:
        """Handle /wechat logout command."""
        from aeloon.channels.wechat_ilink.auth import (
            has_valid_credentials,
            remove_all_credentials,
        )

        accounts_dir = self._wechat_accounts_dir()

        # Cancel any pending login
        self.wechat.cancel_login(msg.channel, msg.chat_id)
        self.wechat.clear_login_status(msg.channel, msg.chat_id)

        # Check if logged in
        if not has_valid_credentials(accounts_dir):
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Not currently logged in to WeChat.",
            )

        # Remove credentials
        count = remove_all_credentials(accounts_dir)

        # Persist wechat.enabled = false in config
        self.wechat.set_enabled(False)

        # Stop wechat channel
        if self._channel_manager:
            await self._channel_manager.stop_channel("wechat")

        # Stop the gateway process
        gateway_killed = self.gateway.stop()

        content = f"✅ WeChat logged out. Removed {count} credential file(s)."
        if gateway_killed:
            content += "\nGateway process stopped."

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
        )

    async def _handle_wechat_status(self, msg: InboundMessage) -> OutboundMessage:
        """Handle /wechat status command."""
        from aeloon.channels.wechat_ilink.auth import (
            get_first_credential,
            has_valid_credentials,
            load_all_credentials,
        )

        accounts_dir = self._wechat_accounts_dir()

        lines = ["📱 WeChat Status:"]

        # Check for login task status
        pending_status = self.wechat.get_login_status(msg.channel, msg.chat_id)
        if pending_status and self.wechat.has_pending_login(msg.channel, msg.chat_id):
            lines.append("\n🔄 Login in progress:")
            lines.append(f"  QR Code: {pending_status.get('qrcode', 'N/A')}")
            lines.append(f"  Status: {pending_status.get('status', 'unknown')}")
        elif pending_status:
            lines.append("\n📋 Last login attempt:")
            lines.append(f"  Status: {pending_status.get('status', 'unknown')}")
            if "error" in pending_status:
                lines.append(f"  Error: {pending_status['error']}")

        # Check credentials
        if has_valid_credentials(accounts_dir):
            creds = get_first_credential(accounts_dir)
            all_creds = load_all_credentials(accounts_dir)
            lines.append("\n✅ Logged in:")
            lines.append(f"  Bot ID: {creds.ilink_bot_id if creds else 'N/A'}")
            lines.append(f"  Accounts: {len(all_creds)}")

            # Check channel status
            if self._channel_manager:
                channel = self._channel_manager.get_channel("wechat")
                if channel:
                    lines.append(f"  Channel: {'Running' if channel.is_running else 'Stopped'}")
                else:
                    lines.append("  Channel: Not configured")
        else:
            lines.append("\n❌ Not logged in")

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="\n".join(lines),
        )

    async def handle_feishu_command(
        self,
        msg: InboundMessage,
        args: list[str],
    ) -> OutboundMessage:
        """Handle /feishu slash command."""
        if not args:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Usage: /feishu login <app_id> <app_secret>|logout|status",
            )

        subcommand = args[0].lower()

        if subcommand == "login":
            return await self._handle_feishu_login(msg, args[1:])
        elif subcommand == "logout":
            return await self._handle_feishu_logout(msg)
        elif subcommand == "status":
            return await self._handle_feishu_status(msg)
        else:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Unknown subcommand: {subcommand}. Use: /feishu login <app_id> <app_secret>|logout|status",
            )

    async def _handle_feishu_login(
        self,
        msg: InboundMessage,
        args: list[str],
    ) -> OutboundMessage:
        """Handle /feishu login command with app_id and app_secret."""
        # Check if already logged in
        if self.feishu.has_credentials():
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Already logged in to Feishu. Use `/feishu logout` first if you want to switch accounts.",
            )

        # Parse arguments
        if len(args) < 2:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Usage: /feishu login <app_id> <app_secret>\n\nGet these from https://open.feishu.cn/app",
            )

        app_id = args[0].strip()
        app_secret = args[1].strip()

        if not app_id or not app_secret:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="App ID and App Secret cannot be empty.",
            )

        # Validate credentials by attempting to connect
        try:
            is_valid = await self.feishu.validate_credentials(app_id, app_secret)
            if not is_valid:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="❌ Failed to validate Feishu credentials. Please check your App ID and App Secret.",
                )
        except Exception as e:
            logger.exception("Feishu credential validation failed")
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"❌ Error validating credentials: {e}",
            )

        # Save credentials to config
        self.feishu.set_credentials(app_id, app_secret, enabled=True)

        # Start or reload Feishu channel
        if self._channel_manager:
            reloaded = await self._channel_manager.reload_channel("feishu")
            if reloaded:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"✅ Feishu login successful!\nApp ID: {app_id}\nChannel is now starting.",
                )
            else:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"✅ Feishu credentials saved!\nApp ID: {app_id}\n\nNote: Channel could not be started automatically. Please check config.",
                )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"✅ Feishu credentials saved!\nApp ID: {app_id}",
        )

    async def _handle_feishu_logout(self, msg: InboundMessage) -> OutboundMessage:
        """Handle /feishu logout command."""
        # Check if logged in
        if not self.feishu.has_credentials():
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Not currently logged in to Feishu.",
            )

        # Get app_id before clearing
        config = self.feishu.get_config()
        app_id = config.get("app_id", "N/A") if isinstance(config, dict) else "N/A"

        # Clear credentials
        self.feishu.set_credentials("", "", enabled=False)

        # Stop Feishu channel
        if self._channel_manager:
            await self._channel_manager.stop_channel("feishu")

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"✅ Feishu logged out.\nApp ID: {app_id} has been removed.",
        )

    async def _handle_feishu_status(self, msg: InboundMessage) -> OutboundMessage:
        """Handle /feishu status command."""
        lines = ["🚀 Feishu Status:"]

        config = self.feishu.get_config()
        if isinstance(config, dict):
            enabled = config.get("enabled", False)
            app_id = config.get("app_id", "")
            has_credentials = bool(app_id)
        else:
            enabled = getattr(config, "enabled", False) if config else False
            app_id = getattr(config, "app_id", "") if config else ""
            has_credentials = bool(app_id)

        if has_credentials:
            lines.append("\n✅ Configured:")
            lines.append(f"  App ID: {app_id}")
            lines.append(f"  Enabled: {enabled}")

            # Check channel status
            if self._channel_manager:
                channel = self._channel_manager.get_channel("feishu")
                if channel:
                    lines.append(f"  Channel: {'Running' if channel.is_running else 'Stopped'}")
                else:
                    lines.append("  Channel: Not loaded (check config)")
        else:
            lines.append("\n❌ Not configured")
            lines.append("\nTo configure, use:")
            lines.append("  `/feishu login <app_id> <app_secret>`")
            lines.append("\nGet credentials from: https://open.feishu.cn/app")

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="\n".join(lines),
        )
