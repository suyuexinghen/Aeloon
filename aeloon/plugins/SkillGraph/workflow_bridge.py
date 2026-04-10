"""Bridge compiled workflows to the Aeloon provider interface."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from aeloon.providers.base import LLMProvider


def make_llm_callable(
    provider: LLMProvider,
    model: str,
) -> Callable[[str, str], Awaitable[str]]:
    """Adapt an Aeloon provider to the simple `(system_prompt, user_prompt) -> text` shape."""

    async def _call(system_prompt: str, user_prompt: str) -> str:
        response = await provider.chat_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
        )
        return response.content or ""

    return _call
