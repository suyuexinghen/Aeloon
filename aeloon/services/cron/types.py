"""Cron data types."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CronSchedule:
    """Cron schedule."""

    kind: Literal["at", "every", "cron"]
    # Used when kind == "at".
    at_ms: int | None = None
    # Used when kind == "every".
    every_ms: int | None = None
    # Used when kind == "cron".
    expr: str | None = None
    # Timezone for cron expressions.
    tz: str | None = None


@dataclass
class CronPayload:
    """Cron job payload."""

    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    # Deliver the result to a channel.
    deliver: bool = False
    channel: str | None = None  # For example: "whatsapp".
    to: str | None = None  # For example: a phone number.


@dataclass
class CronJobState:
    """Cron job state."""

    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None


@dataclass
class CronJob:
    """Cron job."""

    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False


@dataclass
class CronStore:
    """Stored cron jobs."""

    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
