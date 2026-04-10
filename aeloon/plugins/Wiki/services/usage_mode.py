"""Session-scoped wiki usage mode storage."""

from __future__ import annotations


class UsageModeStore:
    """Track how strongly each session should rely on the local wiki."""

    def __init__(self, *, default_mode: str = "prefer-local") -> None:
        self._default_mode = default_mode
        self._state: dict[str, str] = {}

    def get_mode(self, session_key: str) -> str:
        """Return the effective usage mode for one session."""
        if not session_key:
            return self._default_mode
        return self._state.get(session_key, self._default_mode)

    def set_mode(self, session_key: str, mode: str) -> None:
        """Persist one session-scoped usage mode."""
        self._state[session_key] = mode


class SessionToggleStore:
    """Track one boolean session-scoped feature flag."""

    def __init__(self, *, default_enabled: bool = False) -> None:
        self._default_enabled = default_enabled
        self._state: dict[str, bool] = {}

    def is_enabled(self, session_key: str) -> bool:
        """Return whether the flag is enabled for one session."""
        if not session_key:
            return self._default_enabled
        return self._state.get(session_key, self._default_enabled)

    def set_enabled(self, session_key: str, enabled: bool) -> None:
        """Persist one session-scoped boolean flag."""
        self._state[session_key] = enabled
