"""
Configuration — all env vars, paths, and constants in one place.

Reads settings from configs/{APP_ENV}.cfg (dev.cfg / prod.cfg).
API keys come from environment variables (or .env for local dev).
Env vars can override any .cfg value using APP_{SECTION}_{KEY} naming.
"""

import configparser
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths — resolve relative to THIS file, no assumptions about parent layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # research_agent/
CONFIG_DIR = SCRIPT_DIR / "configs"

# .env loading — optional, for local development only.
# On Railway / production, secrets are injected as real env vars.
# We check multiple possible locations for the .env file.
_env_candidates = [
    Path(os.getenv("DOTENV_PATH", ""))  if os.getenv("DOTENV_PATH") else None,
    SCRIPT_DIR / ".env",                # research_agent/.env
    SCRIPT_DIR.parent / ".env",         # day04-Final_Project/.env
    SCRIPT_DIR.parent.parent / ".env",  # 7-day bootcamp/.env (original location)
]
for _candidate in _env_candidates:
    if _candidate and _candidate.exists():
        load_dotenv(_candidate)
        break

# nigerian_stocks.json — check inside package first, then parent
NIGERIAN_STOCKS_PATH = SCRIPT_DIR / "nigerian_stocks.json"
if not NIGERIAN_STOCKS_PATH.exists():
    _parent_path = SCRIPT_DIR.parent / "nigerian_stocks.json"
    if _parent_path.exists():
        NIGERIAN_STOCKS_PATH = _parent_path

MINI_APP_DIR = SCRIPT_DIR / "app"

# ---------------------------------------------------------------------------
# LangSmith tracing — must be set BEFORE any LangChain import
# ---------------------------------------------------------------------------
# LANGCHAIN_API_KEY is what langchain/langsmith actually reads.
# We map from LANGSMITH_API_KEY (the .env convention) for convenience.
_langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
if _langsmith_key:
    os.environ.setdefault("LANGCHAIN_API_KEY", _langsmith_key)
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "default"))

# ---------------------------------------------------------------------------
# Load .cfg file based on APP_ENV
# ---------------------------------------------------------------------------
APP_ENV = os.getenv("APP_ENV", "dev").lower()

_config_path = CONFIG_DIR / f"{APP_ENV}.cfg"
if not _config_path.exists():
    # Fallback: try dev.cfg if the requested env doesn't exist
    _fallback = CONFIG_DIR / "dev.cfg"
    if _fallback.exists():
        _config_path = _fallback
    else:
        raise FileNotFoundError(
            f"Config file not found: {_config_path.resolve()}\n"
            f"Available: {list(CONFIG_DIR.glob('*.cfg'))}"
        )

_parser = configparser.ConfigParser()
_read_ok = _parser.read(_config_path)
if not _read_ok:
    raise RuntimeError(f"Failed to parse: {_config_path}")


def _get(section: str, key: str, env_prefix: str = "APP") -> str:
    """Return a config value — env var override takes priority."""
    env_var = f"{env_prefix}_{section}_{key}".upper()
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value
    return _parser.get(section, key)


def _get_int(section: str, key: str) -> int:
    return int(_get(section, key))


def _get_bool(section: str, key: str) -> bool:
    """Return a config value as a boolean (accepts true/false/yes/no/1/0)."""
    val = _get(section, key).strip().lower()
    return val in ("true", "yes", "1")


# ---------------------------------------------------------------------------
# Logging — read from .cfg, apply via logging_setup (now inside package)
# ---------------------------------------------------------------------------
LOG_DIR = SCRIPT_DIR / _get("logging", "log_dir")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Import from within the package (no sys.path manipulation needed)
from research_agent.logging_setup import setup_logging  # noqa: E402

_is_cloud = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("KOYEB_APP_NAME") or os.getenv("DYNO") or os.getenv("RENDER")

setup_logging(
    level=_get_int("logging", "level"),
    log_dir=str(LOG_DIR),
    enable_console=bool(_is_cloud),  # Always enable console on cloud platforms
)

# ---------------------------------------------------------------------------
# API Keys (secrets — always from env vars, never in .cfg)
# ---------------------------------------------------------------------------

def _require_env(key: str, friendly_name: str = "") -> str:
    """Get a required env var with a clear error message."""
    val = os.getenv(key, "")
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {key}"
            + (f" ({friendly_name})" if friendly_name else "")
            + "\nSet it in your .env file (local) or Railway dashboard (production)."
        )
    return val


ANTHROPIC_API_KEY = _require_env("ANTHROPIC_API_KEY", "Anthropic API key")
OPENAI_API_KEY = _require_env("OPENAI_API_KEY", "OpenAI API key for embeddings")
PINECONE_API_KEY = _require_env("PINECONE_API_KEY", "Pinecone API key")
PINECONE_INDEX_NAME = _require_env("PINECONE_INDEX_NAME", "Pinecone index name")
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_API_KEY", "Telegram Bot API token")

TAVILY_API_KEYS: list[str] = []
# Collect all TAVILY_API_KEY, TAVILY_API_KEY_2, ..., TAVILY_API_KEY_6
_first = os.environ.get("TAVILY_API_KEY", "")
if _first:
    TAVILY_API_KEYS.append(_first)
for _i in range(2, 7):
    _key = os.environ.get(f"TAVILY_API_KEY_{_i}", "")
    if _key:
        TAVILY_API_KEYS.append(_key)
if not TAVILY_API_KEYS:
    raise RuntimeError(
        "No TAVILY_API_KEY found — need at least one.\n"
        "Set TAVILY_API_KEY in your .env file or Railway dashboard."
    )

# Random string — set this in .env or we generate a default
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
if not WEBHOOK_SECRET:
    import secrets
    WEBHOOK_SECRET = secrets.token_hex(32)
    # Log warning so operator knows to set it explicitly
    import logging
    logging.getLogger("research_agent.config").warning(
        "WEBHOOK_SECRET not set — generated a random one. "
        "Set WEBHOOK_SECRET env var for stable webhook auth."
    )

# ---------------------------------------------------------------------------
# Model & Embedding
# ---------------------------------------------------------------------------
CHAT_MODEL = _get("model", "chat_model")
EMBEDDING_MODEL = _get("model", "embedding_model")
OPUS_MODEL = _get("model", "opus_model")
OPUS_THINKING_BUDGET = _get_int("model", "opus_thinking_budget")

# ---------------------------------------------------------------------------
# Agent defaults
# ---------------------------------------------------------------------------
USE_RERANKER = _get_bool("agent", "use_reranker")
PINECONE_TOP_K = _get_int("agent", "pinecone_top_k")
RERANKER_TOP_K = _get_int("agent", "reranker_top_k")
TAVILY_MAX_RESULTS = _get_int("agent", "tavily_max_results")
TAVILY_RESEARCH_POLL_INTERVAL = _get_int("agent", "tavily_research_poll_interval")
TAVILY_RESEARCH_MAX_WAIT = _get_int("agent", "tavily_research_max_wait")
CONTEXT_WINDOW_SIZE = _get_int("agent", "context_window_size")
SUMMARY_TRIGGER = _get_int("agent", "summary_trigger")
RETRIEVAL_STRATEGY = _get("agent", "retrieval_strategy").strip().lower()
DEFAULT_CUTOFF_MONTHS = _get_int("agent", "default_cutoff_months")

# ---------------------------------------------------------------------------
# Session DB
# ---------------------------------------------------------------------------
DB_PATH = SCRIPT_DIR / "sessions.db"

# ---------------------------------------------------------------------------
# Stock Price DB (PostgreSQL — Supabase / Neon / any Postgres)
# ---------------------------------------------------------------------------
# If set, agent reads historic stock prices from this remote database.
# If not set, falls back to local SQLite files in PROJECT_DIR/db/.
STOCK_DB_URL = os.getenv("DATABASE_URL", "")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
# Railway injects PORT; default to 8000 for local dev
PORT = int(os.getenv("PORT", "8000"))

# ---------------------------------------------------------------------------
# LangSmith (read-only exports for agent.py)
# ---------------------------------------------------------------------------
LANGSMITH_TRACING_ENABLED = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "default")
