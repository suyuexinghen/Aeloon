"""/acp command handlers for the ACP Bridge plugin."""

from __future__ import annotations

import logging
import platform
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aeloon.plugins._sdk.acp.types import BackendProfile
from aeloon.plugins._sdk.types import CommandContext

if TYPE_CHECKING:
    from aeloon.plugins.acp_bridge.service import ACPConnectionService

logger = logging.getLogger(__name__)

_REGISTRY_URL = "https://cdn.agentclientprotocol.com/registry/v1/latest/registry.json"
_REGISTRY_CACHE_PATH = Path.home() / ".aeloon" / "cache" / "acp_registry.json"
_REGISTRY_TTL_SECONDS = 3600  # 1 hour


def _get_merged_plugin_config(ctx: CommandContext) -> dict[str, Any]:
    """Get plugin config merged with external acp.json if available.

    External config takes precedence over main config.
    """
    from .config import load_acp_config

    main_config = dict(ctx.plugin_config)
    external_config = load_acp_config()

    if external_config:
        # External values take precedence
        main_config.update(external_config)

    return main_config


def _current_platform_key() -> str | None:
    """Return the platform key matching the registry schema (e.g. ``linux-x86_64``)."""
    os_name = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    if not os_name:
        return None
    machine = platform.machine()
    if machine not in ("x86_64", "aarch64", "arm64", "AMD64"):
        return None
    arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    return f"{os_name}-{arch}"


async def _fetch_registry() -> dict[str, Any]:
    """Fetch the ACP registry index, using a local cache when fresh."""
    import json
    import time

    import httpx

    cache = _REGISTRY_CACHE_PATH
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            fetched_at = data.get("_fetched_at", 0)
            if time.time() - fetched_at < _REGISTRY_TTL_SECONDS:
                return data
        except (json.JSONDecodeError, OSError):
            pass

    async with httpx.AsyncClient() as client:
        resp = await client.get(_REGISTRY_URL, timeout=15)
        resp.raise_for_status()

    data = resp.json()
    data["_fetched_at"] = time.time()

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.debug("Could not cache ACP registry")

    return data


def _resolve_registry_agent(registry: dict[str, Any], agent_id: str) -> dict[str, Any] | None:
    """Find an agent by exact id or case-insensitive name match."""
    agents = registry.get("agents", [])
    for a in agents:
        if a.get("id") == agent_id:
            return a
    lower = agent_id.lower()
    for a in agents:
        if a.get("name", "").lower() == lower:
            return a
    return None


def _format_agent_entry(a: dict[str, Any]) -> str:
    """Format a single registry agent as a markdown table row."""
    dist_tags: list[str] = []
    dist = a.get("distribution", {})
    if dist.get("binary"):
        dist_tags.append("binary")
    if dist.get("npx"):
        dist_tags.append("npx")
    tag_str = " ".join(f"`{t}`" for t in dist_tags) if dist_tags else "`other`"
    desc = a.get("description", "").replace("|", "\\|")
    repo = a.get("repository")
    ver = f"`v{a['version']}`"
    if repo:
        ver = f"[{ver}]({repo})"

    return f"| `{a['id']}` | **{a['name']}** | {ver} | {tag_str} | {desc} |"


_MARKET_TABLE_HEADER = "| ID | Name | Version | Type | Description |\n|---|---|---|---|---|"


HELP_TEXT = """\
ACP Bridge — connect to external ACP agent servers

Usage:
  /acp connect [profile]   Connect to an ACP backend (default: claude_code)
  /acp list                List available ACP backend profiles
  /acp market [filter]     Browse the ACP registry for installable agents
  /acp install <agent>     Install an agent from the ACP registry
  /acp chat <message>      Send a message to the connected agent
  /acp disconnect          Disconnect from the current backend
  /acp status              Show connection state, profile, and sessions
  /acp help                Show this help message

Prerequisites:
  1. Install the Python ACP SDK: pip install agent-client-protocol
     (this provides the `acp` Python module used by Aeloon)
  2. Install the agent: npm install -g @agentclientprotocol/claude-agent-acp
     (or just use npx — it auto-downloads)
  3. Authenticate: run 'claude login' or set ANTHROPIC_API_KEY

Configuration (in aeloon config):
  [plugins.aeloon_acp_bridge]
  enabled = true
  default_profile = "claude_code"

  [plugins.aeloon_acp_bridge.profiles.claude_code]
  command = ["npx", "@agentclientprotocol/claude-agent-acp"]
  env = {ACP_PERMISSION_MODE = "acceptEdits"}
"""


async def handle_acp_command(
    ctx: CommandContext,
    args: str,
) -> str | None:
    """Main dispatcher for the ``/acp`` command namespace."""
    # Get the service from plugin — passed through closure
    # This function is wrapped by the plugin to inject the service reference.
    raise NotImplementedError("Must be called via make_command_handler")


def make_command_handler(service: ACPConnectionService) -> Any:
    """Create a bound command handler with access to the service."""

    async def _handler(ctx: CommandContext, args: str) -> str | None:
        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0] if parts else "help"
        subcmd_args = parts[1] if len(parts) > 1 else ""

        if subcmd == "connect":
            return await _cmd_connect(ctx, subcmd_args, service)
        elif subcmd == "list":
            return await _cmd_list(ctx)
        elif subcmd == "market":
            return await _cmd_market(ctx, subcmd_args)
        elif subcmd == "install":
            return await _cmd_install(ctx, subcmd_args)
        elif subcmd in ("chat", "ask", "delegate"):
            return await _cmd_chat(ctx, subcmd_args, service)
        elif subcmd == "disconnect":
            return await _cmd_disconnect(ctx, service)
        elif subcmd == "status":
            return await _cmd_status(ctx, service)
        elif subcmd == "help":
            return HELP_TEXT
        else:
            return f"Unknown /acp subcommand: {subcmd}\n\n{HELP_TEXT}"

    return _handler


async def _cmd_connect(
    ctx: CommandContext,
    args: str,
    service: ACPConnectionService,
) -> str:
    """Handle ``/acp connect [profile]``."""
    # Merge external acp.json with main config
    config = _get_merged_plugin_config(ctx)

    profile_name = args.strip() or config.get("default_profile", "claude_code")

    # Resolve profile from merged config (includes external acp.json)
    profiles = config.get("profiles", {})
    profile_data = profiles.get(profile_name, {})

    if not profile_data and profile_name == "claude_code":
        # Use built-in default
        profile = BackendProfile(
            name="claude_code",
            command=["npx", "@agentclientprotocol/claude-agent-acp"],
            cwd="~",
            timeout_seconds=30,
            env={"ACP_PERMISSION_MODE": "acceptEdits"},
        )
    elif profile_data:
        profile = BackendProfile(
            name=profile_name,
            command=profile_data.get("command", ["npx", "@zed-industries/claude-agent-acp"]),
            cwd=profile_data.get("cwd", "~"),
            timeout_seconds=profile_data.get("timeout_seconds", 30),
            env=profile_data.get("env", {}),
        )
    else:
        return f"Profile '{profile_name}' not found in configuration."

    await ctx.send_progress(f"Connecting to ACP backend '{profile_name}'...")

    try:
        await service.connect(profile)
        return f"Connected to ACP backend '{profile_name}'."
    except ModuleNotFoundError as exc:
        if exc.name == "acp":
            return (
                "Failed to connect: missing Python module `acp`. "
                "Install the ACP Python SDK with `pip install agent-client-protocol`, "
                "then retry `/acp connect`."
            )
        return f"Failed to connect: {exc}"
    except Exception as exc:
        return f"Failed to connect: {exc}"


async def _cmd_list(ctx: CommandContext) -> str:
    """Handle ``/acp list``."""
    config = _get_merged_plugin_config(ctx)
    default_profile = config.get("default_profile", "claude_code")
    profiles = dict(config.get("profiles", {}))

    if "claude_code" not in profiles:
        profiles["claude_code"] = {
            "command": ["npx", "@agentclientprotocol/claude-agent-acp"],
            "cwd": "~",
            "timeout_seconds": 30,
        }

    lines = ["Available ACP backends:"]
    for name in sorted(profiles):
        profile = profiles[name] or {}
        command = profile.get("command", [])
        command_text = " ".join(command) if command else "(no command)"
        suffix = " (default)" if name == default_profile else ""
        lines.append(f"- {name}{suffix} — {command_text}")

    return "\n".join(lines)


async def _cmd_chat(
    ctx: CommandContext,
    args: str,
    service: ACPConnectionService,
) -> str:
    """Handle ``/acp chat <message>``."""
    task_text = args.strip()
    if not task_text:
        return "Usage: /acp chat <message>"

    if not service.client.is_connected:
        return "Not connected. Use /acp connect first."

    await ctx.send_progress("Delegating to ACP agent...")

    try:
        result = await service.client.prompt(
            aeloon_session_key=ctx.session_key,
            text=task_text,
        )

        if result.content:
            await ctx.reply(result.content)
        else:
            stop_reason = result.execution_meta.get("stop_reason") or "unknown"
            update_types = result.execution_meta.get("update_types") or []
            unknown_update_types = result.execution_meta.get("unknown_update_types") or []
            await ctx.reply(
                "(agent returned empty response)\n"
                f"stop_reason: {stop_reason}\n"
                f"update_types: {', '.join(update_types) if update_types else 'none'}\n"
                f"unparsed_update_types: {', '.join(unknown_update_types) if unknown_update_types else 'none'}"
            )
        return None  # response already sent via ctx.reply

    except RuntimeError as exc:
        return f"Delegation failed: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"
    finally:
        service.client.set_update_callback(None)


async def _cmd_disconnect(
    ctx: CommandContext,
    service: ACPConnectionService,
) -> str:
    """Handle ``/acp disconnect``."""
    if not service.client.is_connected:
        return "Not connected."

    try:
        profile = service.active_profile
        await service.disconnect()
        name = profile.name if profile else "unknown"
        return f"Disconnected from ACP backend '{name}'."
    except Exception as exc:
        return f"Error during disconnect: {exc}"


async def _cmd_status(
    ctx: CommandContext,
    service: ACPConnectionService,
) -> str:
    """Handle ``/acp status``."""
    health = service.health_check()
    state = health.get("state", "disconnected")

    lines = [f"State: {state}"]

    if "profile" in health:
        lines.append(f"Profile: {health['profile']}")

    sessions = health.get("sessions", 0)
    if sessions > 0:
        lines.append(f"Sessions: {sessions}")

    last_error = health.get("last_error")
    if last_error:
        lines.append(f"Last error: {last_error}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /acp market
# ---------------------------------------------------------------------------


async def _cmd_market(ctx: CommandContext, args: str) -> str:
    """Handle ``/acp market [filter]`` — browse the ACP registry."""
    await ctx.send_progress("Fetching ACP registry...")

    try:
        registry = await _fetch_registry()
    except Exception as exc:
        return f"Failed to fetch ACP registry: {exc}"

    agents: list[dict[str, Any]] = registry.get("agents", [])

    keyword = args.strip().lower()
    if keyword:
        agents = [
            a
            for a in agents
            if keyword in a.get("id", "").lower()
            or keyword in a.get("name", "").lower()
            or keyword in a.get("description", "").lower()
        ]

    if not agents:
        return "No agents found."

    lines = [f"## ACP Registry — {len(agents)} agent(s)\n"]
    lines.append(_MARKET_TABLE_HEADER)
    for a in agents:
        lines.append(_format_agent_entry(a))

    lines.append("")
    lines.append("Use `/acp install <id>` to install an agent.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /acp install
# ---------------------------------------------------------------------------


def _build_npx_profile(agent: dict[str, Any]) -> dict[str, Any]:
    """Build a profile dict from an npx distribution."""
    npx = agent["distribution"]["npx"]
    return {
        "command": ["npx", npx["package"]] + npx.get("args", []),
        "cwd": "~",
        "timeoutSeconds": 60,
        "env": npx.get("env", {}),
    }


async def _download_binary_agent(ctx: CommandContext, agent: dict[str, Any]) -> dict[str, Any]:
    """Download and extract a binary agent, returning a profile dict."""
    import httpx

    platform_key = _current_platform_key()
    if not platform_key:
        raise ValueError("Unsupported platform for binary install")

    targets = agent["distribution"]["binary"]
    target = targets.get(platform_key)
    if not target:
        raise ValueError(f"No binary for {platform_key}. Available: {', '.join(targets.keys())}")

    await ctx.send_progress(f"Downloading {agent['name']}...")

    async with httpx.AsyncClient() as client:
        resp = await client.get(target["archive"], timeout=120)
        resp.raise_for_status()

    install_dir = Path.home() / ".aeloon" / "agents" / agent["id"]
    install_dir.mkdir(parents=True, exist_ok=True)

    archive_bytes = BytesIO(resp.content)
    archive_url = target["archive"]

    if archive_url.endswith(".tar.gz") or archive_url.endswith(".tgz"):
        with tarfile.open(fileobj=archive_bytes, mode="r:gz") as tar:
            tar.extractall(install_dir, filter="data")
    elif archive_url.endswith(".zip"):
        with zipfile.ZipFile(archive_bytes) as zf:
            zf.extractall(install_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_url}")

    cmd = str(install_dir / target["cmd"])
    return {
        "command": [cmd] + target.get("args", []),
        "cwd": "~",
        "timeoutSeconds": 60,
        "env": target.get("env", {}),
    }


async def _cmd_install(ctx: CommandContext, args: str) -> str:
    """Handle ``/acp install <agent-id>``."""
    agent_id = args.strip()
    if not agent_id:
        return "Usage: /acp install <agent-id>\nUse /acp market to browse available agents."

    await ctx.send_progress(f"Looking up '{agent_id}'...")

    try:
        registry = await _fetch_registry()
    except Exception as exc:
        return f"Failed to fetch ACP registry: {exc}"

    agent = _resolve_registry_agent(registry, agent_id)
    if not agent:
        return f"Agent '{agent_id}' not found in ACP registry.\nUse /acp market to browse."

    dist = agent.get("distribution", {})
    has_npx = bool(dist.get("npx"))
    has_binary = bool(dist.get("binary"))

    if not has_npx and not has_binary:
        return f"Agent '{agent['name']}' has no compatible distribution."

    try:
        if has_npx:
            profile = _build_npx_profile(agent)
        else:
            profile = await _download_binary_agent(ctx, agent)
    except Exception as exc:
        return f"Install failed: {exc}"

    # Persist to ~/.aeloon/acp.json
    from .config import save_profile_to_acp_config

    try:
        save_profile_to_acp_config(agent["id"], profile)
    except OSError as exc:
        return f"Installed but failed to save config: {exc}\nProfile: {profile}"

    name = agent["name"]
    version = agent["version"]
    method = "npx" if has_npx else "binary"
    return f"Installed {name} (v{version}) via {method}.\nUse /acp connect {agent['id']} to start."
