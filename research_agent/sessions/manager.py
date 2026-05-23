"""
Session Manager — CRUD, context loading, rolling summaries, auto-titles.

Every public method is async.  The webhook handler and Mini App API
both call into this module; it is the single source of truth for
session state.
"""

import json
import uuid
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from research_agent.config import CHAT_MODEL, CONTEXT_WINDOW_SIZE, SUMMARY_TRIGGER
from research_agent.sessions import database as db
from research_agent.sessions.models import Message, Session


# ---------------------------------------------------------------------------
# LLM chains for summaries and titles
# ---------------------------------------------------------------------------

_model = init_chat_model(CHAT_MODEL)

_summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Summarise this conversation concisely. Capture key topics, facts, "
     "numbers, and decisions. Keep it under 200 words."),
    ("human",
     "Previous summary:\n{old_summary}\n\n"
     "New messages:\n{new_messages}\n\n"
     "Updated summary:"),
])
_summary_chain = _summary_prompt | _model | StrOutputParser()

_title_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Generate a short title (3–5 words) for a research conversation "
     "based on the first exchange. Return ONLY the title, nothing else."),
    ("human",
     "User asked: {user_msg}\n\nAssistant replied: {bot_reply}\n\nTitle:"),
])
_title_chain = _title_prompt | _model | StrOutputParser()


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages multi-session state backed by SQLite."""

    # ── Read operations ──────────────────────────────────────────────────

    async def get_active_session(self, user_id: int) -> Session | None:
        """Return the user's currently active session, or None."""
        row = await db.fetch_one(
            "SELECT * FROM sessions WHERE user_id = ? AND is_active = TRUE",
            (user_id,),
        )
        if row is None:
            return None
        return self._row_to_session(row)

    async def get_or_create_active(self, user_id: int) -> Session:
        """Load the active session or create a fresh one."""
        session = await self.get_active_session(user_id)
        if session is not None:
            return session
        return await self.create_session(user_id)

    async def load_session(self, session_id: str) -> Session | None:
        """Load a session by ID."""
        row = await db.fetch_one(
            "SELECT * FROM sessions WHERE id = ?", (session_id,),
        )
        if row is None:
            return None
        return self._row_to_session(row)

    async def list_sessions(self, user_id: int,
                            page: int = 0, per_page: int = 10) -> list[dict]:
        """Paginated session list for the Mini App, newest first."""
        rows = await db.fetch_all(
            "SELECT s.*, "
            "  (SELECT COUNT(*) FROM messages WHERE session_id = s.id) AS msg_count, "
            "  (SELECT content FROM messages WHERE session_id = s.id "
            "   ORDER BY created_at DESC LIMIT 1) AS last_message "
            "FROM sessions s "
            "WHERE s.user_id = ? "
            "ORDER BY s.updated_at DESC "
            "LIMIT ? OFFSET ?",
            (user_id, int(per_page), int(page * per_page)),
        )
        return rows

    # ── Write operations ─────────────────────────────────────────────────

    async def create_session(self, user_id: int) -> Session:
        """Deactivate old session, create and activate a new one."""
        await self._deactivate_all(user_id)

        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        await db.execute(
            "INSERT INTO sessions (id, user_id, is_active, created_at, updated_at) "
            "VALUES (?, ?, TRUE, ?, ?)",
            (session_id, user_id, now, now),
        )

        return Session(
            id=session_id,
            user_id=user_id,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    async def switch_session(self, user_id: int, session_id: str):
        """Deactivate current, activate *session_id*."""
        await self._deactivate_all(user_id)
        await db.execute(
            "UPDATE sessions SET is_active = TRUE, updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc), session_id),
        )

    async def rename_session(self, session_id: str, new_title: str):
        """Set a custom title."""
        await db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, datetime.now(timezone.utc), session_id),
        )

    async def clear_session(self, session_id: str):
        """Wipe messages and summary but keep the session shell."""
        await db.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,),
        )
        await db.execute(
            "UPDATE sessions SET summary = '', updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc), session_id),
        )

    async def delete_session(self, session_id: str):
        """Fully remove a session and its messages."""
        await db.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,),
        )
        await db.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,),
        )

    # ── Message operations ───────────────────────────────────────────────

    async def add_message(self, session_id: str, role: str, content: str,
                          sources: list[dict] | None = None):
        """Insert a message and touch the session's updated_at."""
        sources_json = json.dumps(sources or [])
        await db.execute(
            "INSERT INTO messages (session_id, role, content, sources) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, sources_json),
        )
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc), session_id),
        )

    async def get_messages(self, session_id: str) -> list[dict]:
        """All messages in a session, oldest first."""
        return await db.fetch_all(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )

    async def get_context_window(self, session_id: str,
                                 limit: int | None = None) -> list[dict]:
        """Last N messages formatted for the LLM context."""
        limit = limit or CONTEXT_WINDOW_SIZE
        rows = await db.fetch_all(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        # Reverse so oldest is first (chronological order)
        rows.reverse()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    async def get_message_count(self, session_id: str) -> int:
        """How many messages are in this session."""
        count = await db.fetch_scalar(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        )
        return count or 0

    # ── Summary management ───────────────────────────────────────────────

    async def get_summary(self, session_id: str) -> str:
        """Return the rolling summary for a session."""
        row = await db.fetch_one(
            "SELECT summary FROM sessions WHERE id = ?", (session_id,),
        )
        return row["summary"] if row else ""

    async def maybe_update_summary(self, session_id: str):
        """If the message count crosses a SUMMARY_TRIGGER boundary,
        generate a new rolling summary from the recent messages."""
        count = await self.get_message_count(session_id)
        if count < SUMMARY_TRIGGER or count % (SUMMARY_TRIGGER // 2) != 0:
            return

        old_summary = await self.get_summary(session_id)

        # Grab the most recent batch of messages for summarisation
        recent = await db.fetch_all(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, SUMMARY_TRIGGER),
        )
        recent.reverse()

        new_messages = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: "
            f"{m['content'][:300]}"
            for m in recent
        )

        summary = await _summary_chain.ainvoke({
            "old_summary": old_summary or "(none)",
            "new_messages": new_messages,
        })

        await db.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            (summary.strip(), session_id),
        )

    async def auto_title(self, session_id: str,
                         user_msg: str, bot_reply: str):
        """Generate a 3–5 word title from the first exchange."""
        session = await self.load_session(session_id)
        if session is None or session.title != "Untitled":
            return  # Already titled

        title = await _title_chain.ainvoke({
            "user_msg": user_msg[:300],
            "bot_reply": bot_reply[:300],
        })

        clean_title = title.strip().strip('"').strip("'")[:60]
        await self.rename_session(session_id, clean_title)

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _deactivate_all(self, user_id: int):
        """Set is_active = 0 for all sessions belonging to *user_id*."""
        await db.execute(
            "UPDATE sessions SET is_active = FALSE WHERE user_id = ?",
            (user_id,),
        )

    @staticmethod
    def _row_to_session(row: dict) -> Session:
        """Convert a database row dict into a Session dataclass."""
        return Session(
            id=row["id"],
            user_id=row["user_id"],
            title=row.get("title", "Untitled"),
            summary=row.get("summary", ""),
            is_active=bool(row.get("is_active", False)),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
        )
