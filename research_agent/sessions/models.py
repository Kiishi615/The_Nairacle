"""
Data models for sessions and messages.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Session:
    """A single research conversation belonging to one user."""

    id: str                                        # UUID
    user_id: int                                   # Telegram user ID
    title: str = "Untitled"                        # Auto-generated or user-set
    summary: str = ""                              # Rolling LLM summary
    is_active: bool = False
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass
class Message:
    """A single message within a session."""

    id: int = 0                                    # Auto-incremented by SQLite
    session_id: str = ""                           # FK → Session.id
    role: str = ""                                 # "user" | "assistant"
    content: str = ""
    sources: str = "[]"                            # JSON string of source dicts
    created_at: datetime = field(default_factory=_utcnow)
