"""Implicit query-grounding middleware for the wiki plugin."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from aeloon.core.agent.middleware import BaseAgentMiddleware

from ..services.query_service import QueryService
from ..services.usage_mode import UsageModeStore

_EVIDENCE_SENTINEL = "## Wiki Evidence"
_GAP_SENTINEL = "## Wiki Coverage Gap"
_CURRENT_MESSAGE_CONTEXT: ContextVar["WikiMessageContext | None"] = ContextVar(
    "wiki_message_context",
    default=None,
)


@dataclass(slots=True)
class WikiMessageContext:
    """Per-message context captured from the MESSAGE_RECEIVED hook."""

    session_key: str
    channel: str
    chat_id: str
    content: str = ""
    media: list[str] = field(default_factory=list)


class WikiQueryMiddleware(BaseAgentMiddleware):
    """Inject wiki evidence and related-entry guidance."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._query_service: QueryService | None = None
        self._usage_mode_store: UsageModeStore | None = None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable grounding behavior."""
        self._enabled = enabled

    def set_query_service(self, query_service: QueryService | None) -> None:
        """Attach the query service used for evidence lookup."""
        self._query_service = query_service

    def set_usage_mode_store(self, usage_mode_store: UsageModeStore | None) -> None:
        """Attach session-scoped wiki usage mode state."""
        self._usage_mode_store = usage_mode_store

    async def capture_message_context(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        content: str = "",
        media: list[str] | None = None,
        **_: Any,
    ) -> None:
        """Capture the current message context without requiring core changes."""
        _CURRENT_MESSAGE_CONTEXT.set(
            WikiMessageContext(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                content=content,
                media=list(media or []),
            )
        )

    async def around_llm(
        self,
        messages: list[dict],
        tool_defs: list[dict],
        call_llm: Callable[[list[dict], list[dict]], Awaitable[Any]],
    ) -> Any:
        if not self._enabled or self._query_service is None:
            return await call_llm(messages, tool_defs)
        if any(message.get("role") == "tool" for message in messages):
            return await call_llm(messages, tool_defs)

        query = self._latest_user_text(messages)
        if not query or query.startswith("/"):
            return await call_llm(messages, tool_defs)

        mode = self._usage_mode()
        if mode == "off":
            return await call_llm(messages, tool_defs)

        result = await self._query_service.search(query)
        if result.primary_evidence:
            return await call_llm(
                self._inject_block(
                    messages,
                    self._query_service.format_evidence_block(
                        query,
                        result.primary_evidence,
                        result.related_entries,
                    ),
                ),
                tool_defs,
            )

        if mode == "local-only" and self._looks_like_knowledge_query(query):
            return await call_llm(
                self._inject_block(messages, self._query_service.format_gap_block(query)),
                tool_defs,
            )
        return await call_llm(messages, tool_defs)

    def _usage_mode(self) -> str:
        if self._usage_mode_store is None:
            return "prefer-local"
        current = _CURRENT_MESSAGE_CONTEXT.get()
        if current is None:
            return "prefer-local"
        return self._usage_mode_store.get_mode(current.session_key)

    def _latest_user_text(self, messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                return "\n".join(part for part in text_parts if part).strip()
        return ""

    def _inject_block(self, messages: list[dict], block: str) -> list[dict]:
        if not messages:
            return messages
        system_message = messages[0]
        content = system_message.get("content", "")
        if isinstance(content, str) and (
            _EVIDENCE_SENTINEL in content
            or _GAP_SENTINEL in content
        ):
            return messages
        mutated = list(messages)
        mutated[0] = {
            **system_message,
            "content": f"{content}\n\n{block}" if content else block,
        }
        return mutated

    def _looks_like_knowledge_query(self, query: str) -> bool:
        lowered = query.lower()
        if re.fullmatch(r"[1-9][0-9]*", query.strip()):
            return False
        if "?" in query:
            return True
        prefixes = (
            "what",
            "why",
            "how",
            "compare",
            "explain",
            "summarize",
            "tell me",
        )
        return lowered.startswith(prefixes)
