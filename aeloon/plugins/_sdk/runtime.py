"""Plugin runtime — platform capabilities available to plugins.

:class:`PluginRuntime` provides access to the agent loop, LLM proxy,
namespaced storage, configuration, and logging.
:class:`LegacyRuntimeAdapter` wraps the same interface for backward-
compatible Science Agent code during migration.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

if TYPE_CHECKING:
    from aeloon.core.agent.loop import AgentLoop

# Well-known prefix for plugin-internal session keys.
# Core code checks this prefix to skip memory consolidation and other
# user-scoped behaviours for plugin-owned sessions.
PLUGIN_SESSION_PREFIX = "_plugin:"


class PluginLLMProxy:
    """Thin wrapper around the agent's LLM provider for plugin use."""

    def __init__(self, provider: Any, default_model: str) -> None:
        self._provider = provider
        self._default_model = default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """Send a chat completion and return the text content."""
        response = await self._provider.chat(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content or ""

    async def structured_output(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Request structured JSON output from the LLM.

        Appends a JSON-schema instruction to the system message and
        attempts to parse the response as JSON.
        """
        schema_instruction = (
            "Respond with a valid JSON object conforming to this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )
        augmented = list(messages)
        if augmented and augmented[0].get("role") == "system":
            augmented[0] = {
                **augmented[0],
                "content": augmented[0]["content"] + "\n\n" + schema_instruction,
            }
        else:
            augmented.insert(0, {"role": "system", "content": schema_instruction})

        raw = await self.chat(augmented, model=model)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise

    async def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Chat completion returning parsed JSON dict with repair logic."""
        raw = await self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        if not raw or not raw.strip():
            raise RuntimeError("LLM returned empty content")
        # Extract JSON from possible markdown fences
        content = raw.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end >= 0:
            content = content[start : end + 1]
        return self._parse_json_payload(content)

    # ------------------------------------------------------------------
    # JSON repair helpers
    # ------------------------------------------------------------------

    def _parse_json_payload(self, raw_json: str) -> dict[str, Any]:
        last_error: Exception | None = None
        candidates = [
            raw_json,
            self._normalize_json_text(raw_json),
        ]
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                last_error = exc
        repaired = self._repair_js_like_object(raw_json)
        for candidate in (repaired, self._normalize_json_text(repaired)):
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                last_error = exc
        try:
            python_literal = self._to_python_literal(repaired)
            payload = ast.literal_eval(python_literal)
            if isinstance(payload, dict):
                return payload
            raise RuntimeError("parsed payload is not an object")
        except Exception as exc:
            last_error = exc
        raise RuntimeError(
            f"llm json parse failed: {last_error}; snippet={raw_json[:400]}"
        ) from last_error

    def _normalize_json_text(self, text: str) -> str:
        value = text.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*", "", value)
            value = re.sub(r"\s*```$", "", value)
        value = (
            value.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        value = re.sub(r",(\s*[}\]])", r"\1", value)
        return value.strip()

    def _repair_js_like_object(self, text: str) -> str:
        value = self._normalize_json_text(text)
        value = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', value)
        value = re.sub(
            r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'",
            lambda m: ': "' + self._escape_json_string(m.group(1)) + '"',
            value,
        )
        value = re.sub(
            r"([{,]\s*)\'([^\'\\]*(?:\\.[^\'\\]*)*)\'(\s*:)",
            lambda m: f'{m.group(1)}"{self._escape_json_string(m.group(2))}"{m.group(3)}',
            value,
        )
        value = re.sub(r"\bTrue\b", "true", value)
        value = re.sub(r"\bFalse\b", "false", value)
        value = re.sub(r"\bNone\b", "null", value)
        return value

    def _to_python_literal(self, text: str) -> str:
        value = self._repair_js_like_object(text)
        value = re.sub(r"\btrue\b", "True", value)
        value = re.sub(r"\bfalse\b", "False", value)
        value = re.sub(r"\bnull\b", "None", value)
        return value

    def _escape_json_string(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')


class PluginRuntime:
    """Platform capabilities available to a plugin during ``activate()`` and runtime."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        plugin_id: str,
        config: Mapping[str, Any],
        storage_base: Path,
    ) -> None:
        self._agent_loop = agent_loop
        self._plugin_id = plugin_id
        self._config = config
        self._storage_path = storage_base / plugin_id.replace(".", "/")
        self._logger = logging.getLogger(f"aeloon.plugin.{plugin_id}")
        self._llm = PluginLLMProxy(
            provider=agent_loop.provider,
            default_model=getattr(agent_loop, "model", ""),
        )

    @property
    def agent_loop(self) -> AgentLoop:
        return self._agent_loop

    def internal_session_key(self, *parts: str) -> str:
        """Build a session key scoped to this plugin's internal use.

        Core code recognises the ``_plugin:`` prefix and skips user-scoped
        behaviours (memory consolidation, etc.) for these sessions.

        Example::

            runtime.internal_session_key("task_123", "step1")
            # → "_plugin:aeloon.science:task_123:step1"
        """
        return f"{PLUGIN_SESSION_PREFIX}{self._plugin_id}:{':'.join(parts)}"

    async def process_direct(self, content: str, **kwargs: Any) -> str:
        """Delegate processing to the main agent loop (for Task plugins).

        Falls back to an LLM chat call if the agent loop does not expose
        a direct processing method.
        """
        # Try the agent loop's own direct-processing method first
        if hasattr(self._agent_loop, "process_direct"):
            return await self._agent_loop.process_direct(content, **kwargs)
        # Fallback: simple LLM chat
        return await self._llm.chat([{"role": "user", "content": content}])

    @property
    def llm(self) -> PluginLLMProxy:
        return self._llm

    @property
    def storage_path(self) -> Path:
        """Plugin-private storage directory.  Created on first access."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        return self._storage_path

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def add_deep_profile_section(self, title: str, lines: list[str]) -> None:
        """Contribute a section to the agent deep-profile report.

        No-ops if profiling is disabled or no profiler is attached.
        """
        profiler = getattr(self._agent_loop, "profiler", None)
        if profiler is None or not getattr(profiler, "enabled", False):
            return
        profiler.add_deep_profile_section(title, lines)

    async def tool_execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """Execute an agent tool by name.  Returns the tool result as a string."""
        return await self._agent_loop.tools.execute(tool_name, params)

    @property
    def supports_async_tool_execute(self) -> bool:
        """Whether the runtime supports async tool execution."""
        tools = getattr(self._agent_loop, "tools", None)
        execute = getattr(tools, "execute", None)
        return isinstance(execute, AsyncMock) or asyncio.iscoroutinefunction(execute)


class LegacyRuntimeAdapter(PluginRuntime):
    """Compatibility adapter for the Science Agent migration.

    Wraps :class:`PluginRuntime` so that code originally written as
    ``SciencePipeline(agent_loop=...)`` can work through the plugin
    runtime layer without a full rewrite.
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        config: Mapping[str, Any],
        storage_path: Path,
        plugin_id: str = "legacy",
    ) -> None:
        super().__init__(
            agent_loop=agent_loop,
            plugin_id=plugin_id,
            config=config,
            storage_base=storage_path.parent,
        )
        # Override storage path to use the explicit path (backward compat).
        self._storage_path = storage_path
