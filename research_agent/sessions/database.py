"""
SQLite database setup and async helpers.

Creates the sessions and messages tables on first run.
All queries go through the helpers here so the rest of the code
never touches raw SQL.
"""

import aiosqlite

from research_agent.config import DB_PATH


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    title      TEXT    DEFAULT 'Untitled',
    summary    TEXT    DEFAULT '',
    is_active  BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    sources    TEXT    DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON sessions(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);
"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def execute(sql: str, params: tuple = ()):
    """Run a write query (INSERT / UPDATE / DELETE)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute(sql, params)
        await db.commit()


async def fetch_one(sql: str, params: tuple = ()) -> dict | None:
    """Return a single row as a dict, or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)


async def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
    """Return all matching rows as a list of dicts."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def fetch_scalar(sql: str, params: tuple = ()):
    """Return a single scalar value (first column of first row)."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(sql, params)
        row = await cursor.fetchone()
        return row[0] if row else None
