"""
Nairametrics Research Agent — built on LangChain.

Tools:
    1. search_knowledge_base       — Pinecone retriever (4,792 articles)
    2. search_the_internet         — Tavily web search (fallback)
    3. deep_market_research        — Tavily Research API (comprehensive)
    4. lookup_nigerian_ticker      — local JSON lookup
    5. search_stock_price          — TradingView Screener (live prices)
    6. search_historic_stock_price — local SQLite DB (historical prices)
    7. compute_ema                 — 20 & 50 EMA from local SQLite DB
    8. escalate_to_opus            — Claude Opus 4.7 extended thinking
    9. send_telegram_message       — send text to a Telegram chat

Pattern: mirrors robust_agent.py — same retry decorator, same @tool
         style, same create_agent() with prompt-based tool ordering.
"""

import anthropic
import functools
import json
import logging
import sqlite3
import threading
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from dateutil.relativedelta import relativedelta
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable
from openai import OpenAI
from pinecone import Pinecone
from tavily import TavilyClient
from tradingview_screener import Query

from research_agent.config import (
    ANTHROPIC_API_KEY,
    CHAT_MODEL,
    DEFAULT_CUTOFF_MONTHS,
    EMBEDDING_MODEL,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING_ENABLED,
    NIGERIAN_STOCKS_PATH,
    OPENAI_API_KEY,
    OPUS_MODEL,
    OPUS_THINKING_BUDGET,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_TOP_K,
    RERANKER_TOP_K,
    RETRIEVAL_STRATEGY,
    SCRIPT_DIR,
    STOCK_DB_URL,
    TAVILY_API_KEYS,
    TAVILY_MAX_RESULTS,
    TAVILY_RESEARCH_MAX_WAIT,
    TAVILY_RESEARCH_POLL_INTERVAL,
    TELEGRAM_BOT_TOKEN,
    USE_RERANKER,
)
if RETRIEVAL_STRATEGY in ("reranker", "hybrid") or USE_RERANKER:
    from research_agent.reranker import rerank

# ---------------------------------------------------------------------------
# Logging — configured by logging_{dev,prod}.cfg via config.py import
# ---------------------------------------------------------------------------
logger = logging.getLogger("research_agent")
logger.info(
    "LangSmith tracing: %s (project: %s)",
    "ENABLED" if LANGSMITH_TRACING_ENABLED else "DISABLED",
    LANGSMITH_PROJECT,
)


# ---------------------------------------------------------------------------
# Retry Decorator
# ---------------------------------------------------------------------------

def retry(max_retries: int = 3, backoff_base: int = 2, fallback=None):
    """Retry a function up to *max_retries* times with exponential back-off.

    Args:
        max_retries:  Number of retry attempts after the first failure.
        backoff_base: Base for the exponential delay (seconds).
        fallback:     Optional callable(exception) → str invoked when every
                      retry is exhausted. If None, a generic error string is
                      returned.

    Usage::

        @retry(max_retries=3, fallback=lambda e: "service down")
        def my_tool(arg):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 2):  # attempt 1 = original call
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    if attempt <= max_retries:
                        delay = backoff_base ** attempt
                        logger.warning(
                            "Tool '%s' failed (attempt %d/%d): %s — "
                            "retrying in %ds…",
                            func.__name__, attempt, max_retries + 1,
                            exc, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "Tool '%s' exhausted all %d attempts. "
                            "Last error: %s",
                            func.__name__, max_retries + 1, exc,
                        )

            # All retries exhausted — use fallback or generic message
            if fallback is not None:
                return fallback(last_exception)
            return (
                f"Error: tool '{func.__name__}' failed after "
                f"{max_retries + 1} attempts. Last error: {last_exception}"
            )

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

model = init_chat_model(CHAT_MODEL)

# OpenAI client — used ONLY for embedding queries (Pinecone search)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=PINECONE_INDEX_NAME)

# Anthropic client — used for Opus escalation (extended thinking)
opus_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Nigerian stocks lookup table
try:
    with open(NIGERIAN_STOCKS_PATH, "r") as f:
        NIGERIAN_STOCKS = json.load(f)
    logger.info("Loaded %d Nigerian stocks from %s", len(NIGERIAN_STOCKS), NIGERIAN_STOCKS_PATH)
except FileNotFoundError:
    logger.warning("nigerian_stocks.json not found at %s — ticker lookup disabled", NIGERIAN_STOCKS_PATH)
    NIGERIAN_STOCKS = []

# Historic price databases — PostgreSQL (remote) or local SQLite fallback
_USE_POSTGRES = bool(STOCK_DB_URL)

if _USE_POSTGRES:
    import psycopg2
    # Fetch available stock names from PostgreSQL
    try:
        _pg = psycopg2.connect(STOCK_DB_URL)
        _cur = _pg.cursor()
        _cur.execute("SELECT DISTINCT stock_name FROM stock_prices ORDER BY stock_name")
        _AVAILABLE_STOCKS = [row[0] for row in _cur.fetchall()]
        _cur.close()
        _pg.close()
    except Exception as _e:
        logger.error("Failed to connect to PostgreSQL for stock list: %s", _e)
        _AVAILABLE_STOCKS = []
    _AVAILABLE_DBS = {}  # not used in Postgres mode
    logger.info("Stock prices: PostgreSQL mode (%d stocks)", len(_AVAILABLE_STOCKS))
else:
    # Try multiple possible locations for the db/ directory
    _db_candidates = [
        SCRIPT_DIR / "db",              # research_agent/db/
        SCRIPT_DIR.parent / "db",       # day04-Final_Project/db/
    ]
    DB_DIR = None
    for _candidate in _db_candidates:
        if _candidate.exists():
            DB_DIR = _candidate
            break

    if DB_DIR:
        _AVAILABLE_DBS = {
            f.stem: DB_DIR / f.name
            for f in sorted(DB_DIR.glob("*.db"))
        }
    else:
        logger.warning("No db/ directory found — historic stock prices unavailable")
        _AVAILABLE_DBS = {}
    _AVAILABLE_STOCKS = list(_AVAILABLE_DBS.keys())
    logger.info("Stock prices: local SQLite mode (%d stocks)", len(_AVAILABLE_STOCKS))

_STOCK_LIST_STR = ", ".join(_AVAILABLE_STOCKS)


def _query_stock_db(stock_name: str, sql_sqlite: str, params_sqlite: tuple,
                    sql_pg: str, params_pg: tuple):
    """Run a stock price query against PostgreSQL or local SQLite.

    Args:
        stock_name:    Exact stock name.
        sql_sqlite:    SQL for local SQLite (uses 'prices' table).
        params_sqlite: Parameters for SQLite query.
        sql_pg:        SQL for PostgreSQL (uses 'stock_prices' table).
        params_pg:     Parameters for PostgreSQL query.

    Returns:
        List of tuples, or None if stock not found.
    """
    if _USE_POSTGRES:
        conn = psycopg2.connect(STOCK_DB_URL)
        cur = conn.cursor()
        try:
            cur.execute(sql_pg, params_pg)
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()
    else:
        db_path = _AVAILABLE_DBS.get(stock_name)
        if db_path is None:
            return None
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        try:
            cur.execute(sql_sqlite, params_sqlite)
            return cur.fetchall()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Traceable wrappers — make direct API calls visible in LangSmith
# ---------------------------------------------------------------------------

@traceable(run_type="embedding", name="openai_embed_query")
def _embed_query(text: str) -> list[float]:
    """Embed a single text string using OpenAI. Traced by LangSmith."""
    response = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


@traceable(run_type="llm", name="anthropic_opus_thinking")
def _call_opus(prompt: str) -> anthropic.types.Message:
    """Call Anthropic Opus with extended thinking. Traced by LangSmith."""
    return opus_client.messages.create(
        model=OPUS_MODEL,
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": OPUS_THINKING_BUDGET,
        },
        messages=[{"role": "user", "content": prompt}],
    )


# ---------------------------------------------------------------------------
# Tavily Key Pool — auto-rotates across multiple API keys on quota errors
# ---------------------------------------------------------------------------

class TavilyKeyPool:
    """Round-robin pool of Tavily API keys with automatic failover.

    When a key hits a rate-limit or quota error, the pool marks it as
    exhausted and rotates to the next available key. All keys are tracked
    by index so the agent can keep working until *every* key is spent.
    """

    # Error substrings that signal we should rotate to the next key
    _ROTATE_SIGNALS = (
        "rate limit",
        "quota",
        "exceeded",
        "429",
        "too many requests",
        "insufficient credits",
    )

    def __init__(self, keys: list[str]):
        self._keys = list(keys)
        self._current_idx = 0
        self._exhausted: set[int] = set()
        self._lock = threading.Lock()
        logger.info(
            "TavilyKeyPool initialised with %d key(s). Active: key #1.",
            len(self._keys),
        )

    # -- public API --------------------------------------------------------

    def get_client(self) -> TavilyClient:
        """Return a TavilyClient using the current active key."""
        if len(self._exhausted) >= len(self._keys):
            raise RuntimeError(
                "All Tavily API keys are exhausted. "
                "Please add more keys to .env or wait for quota reset."
            )
        key = self._keys[self._current_idx]
        return TavilyClient(api_key=key)

    def rotate(self, error: Exception | None = None) -> bool:
        """Mark the current key as exhausted and switch to the next.

        Returns True if a new key is available, False if all are spent.
        """
        with self._lock:
            old_idx = self._current_idx
            self._exhausted.add(old_idx)
            logger.warning(
                "Tavily key #%d exhausted (error: %s). "
                "Rotating…",
                old_idx + 1, error,
            )

            # Find next non-exhausted key
            for offset in range(1, len(self._keys) + 1):
                candidate = (old_idx + offset) % len(self._keys)
                if candidate not in self._exhausted:
                    self._current_idx = candidate
                    logger.info(
                        "Switched to Tavily key #%d (%d/%d remaining).",
                        candidate + 1,
                        len(self._keys) - len(self._exhausted),
                        len(self._keys),
                    )
                    return True

            logger.error("All %d Tavily keys exhausted!", len(self._keys))
            return False

    def should_rotate(self, error: Exception) -> bool:
        """Check if an error indicates the current key's quota is spent."""
        msg = str(error).lower()
        return any(signal in msg for signal in self._ROTATE_SIGNALS)

    @property
    def active_key_number(self) -> int:
        """1-indexed number of the currently active key."""
        return self._current_idx + 1

    @property
    def keys_remaining(self) -> int:
        return len(self._keys) - len(self._exhausted)


tavily_pool = TavilyKeyPool(TAVILY_API_KEYS)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    "search_knowledge_base",
    description=(
        "Search the internal knowledge base of 4,792 Nigerian financial "
        "news articles from Nairametrics. Use this FIRST for any question "
        "about Nigerian finance, economy, banking, markets, or policy. "
        "Returns article summaries with titles, dates, and URLs.\n\n"
        "Args:\n"
        "  query: The search query.\n"
        "  cutoff_date: Optional earliest date (YYYY-MM-DD) to include. "
        "Only articles on or after this date are returned. Example: "
        "'2026-03-01' for articles from March 2026 onwards. If the user "
        "says 'last 2 months', calculate the date from today and pass it."
    ),
)
def search_knowledge_base(query: str, cutoff_date: str = "") -> str:
    """Embed *query* and retrieve the top-k articles from Pinecone.

    Args:
        query:       The user's search query.
        cutoff_date: Optional earliest date as YYYY-MM-DD. Articles before
                     this date are excluded. If empty, defaults to
                     DEFAULT_CUTOFF_MONTHS from config.
    """

    @retry(
        max_retries=2,
        fallback=lambda e: (
            "Knowledge base temporarily unavailable. "
            f"Error: {e}"
        ),
    )
    def _search(q):
        # Embed the query with the same model used at index time
        query_vector = _embed_query(q)

        # Query Pinecone
        results = index.query(
            vector=query_vector,
            top_k=PINECONE_TOP_K,
            include_metadata=True,
        )

        if not results.matches:
            return "No relevant articles found in the knowledge base."

        # ----- Client-side date filtering --------------------------------
        # Pinecone only supports $gte on numeric fields, but our dates are
        # stored as YYYY-MM-DD strings, which sort lexicographically.
        # So we filter here in Python after retrieval.
        if cutoff_date:
            min_date = cutoff_date
        else:
            # Default: go back DEFAULT_CUTOFF_MONTHS from today
            min_date = (
                date.today() - relativedelta(months=DEFAULT_CUTOFF_MONTHS)
            ).isoformat()

        filtered = [
            m for m in results.matches
            if m.metadata.get("date", "1970-01-01") >= min_date
        ]

        if not filtered:
            # Fallback: widen to all results if filter was too narrow
            logger.info(
                "Date filter (%s) removed all results — falling back to "
                "unfiltered.", min_date,
            )
            filtered = results.matches

        # ----- Apply retrieval strategy ---------------------------------
        output = []
        strategy = RETRIEVAL_STRATEGY

        if strategy == "chronological":
            # Sort by date descending (newest first)
            filtered.sort(
                key=lambda m: m.metadata.get("date", "1970-01-01"),
                reverse=True,
            )
            for i, match in enumerate(filtered[:RERANKER_TOP_K], 1):
                meta = match.metadata
                output.append(
                    f"[{i}] {meta.get('title', 'Untitled')} "
                    f"({meta.get('date', 'unknown date')})\n"
                    f"    Cosine score: {match.score:.4f}\n"
                    f"    Summary: {meta.get('summary', 'No summary')}\n"
                    f"    URL: {meta.get('url', '')}"
                )

        elif strategy == "reranker":
            # Cross-encoder rerank (ignores date ordering)
            ranked = rerank(q, filtered, top_k=RERANKER_TOP_K)
            for i, (match, ce_score) in enumerate(ranked, 1):
                meta = match.metadata
                output.append(
                    f"[{i}] {meta.get('title', 'Untitled')} "
                    f"({meta.get('date', 'unknown date')})\n"
                    f"    Relevance: {ce_score:.4f}\n"
                    f"    Summary: {meta.get('summary', 'No summary')}\n"
                    f"    URL: {meta.get('url', '')}"
                )

        else:  # hybrid — date-filtered + reranked
            ranked = rerank(q, filtered, top_k=RERANKER_TOP_K)
            for i, (match, ce_score) in enumerate(ranked, 1):
                meta = match.metadata
                output.append(
                    f"[{i}] {meta.get('title', 'Untitled')} "
                    f"({meta.get('date', 'unknown date')})\n"
                    f"    Relevance: {ce_score:.4f}\n"
                    f"    Summary: {meta.get('summary', 'No summary')}\n"
                    f"    URL: {meta.get('url', '')}"
                )

        return "\n\n".join(output) if output else (
            "No relevant articles found after filtering."
        )

    return _search(query)


@tool(
    "search_the_internet",
    description=(
        "Search the internet using Tavily for current events, "
        "recent news, or real-time information. Only use this "
        "when the knowledge base does not have relevant results "
        "or the user asks about very recent events."
    ),
)
def search_the_internet(query: str) -> str:
    """Web search via Tavily — fallback when Pinecone isn't enough."""

    max_attempts = tavily_pool.keys_remaining + 3  # retries + key rotations
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            tavily_client = tavily_pool.get_client()
            response = tavily_client.search(
                query=query,
                max_results=TAVILY_MAX_RESULTS,
                search_depth="advanced",
            )
            data = ""
            for i in response["results"]:
                data += "".join(i["content"])
            return data

        except Exception as exc:
            last_error = exc
            if tavily_pool.should_rotate(exc):
                if tavily_pool.rotate(exc):
                    logger.info(
                        "search_the_internet: rotated key, retrying "
                        "(attempt %d)…", attempt,
                    )
                    continue
                else:
                    return (
                        "All Tavily API keys exhausted. "
                        "Internet search unavailable."
                    )
            else:
                # Regular error — backoff and retry with same key
                delay = 2 ** attempt
                logger.warning(
                    "search_the_internet failed (attempt %d): %s "
                    "— retrying in %ds…", attempt, exc, delay,
                )
                time.sleep(delay)

    return (
        f"Internet search unavailable after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


@tool(
    "lookup_nigerian_ticker",
    description=(
        "Search for a Nigerian stock by company name or symbol. "
        "Returns the correct TradingView ticker. "
        "ALWAYS call this BEFORE search_stock_price."
    ),
)
def lookup_nigerian_ticker(search_term: str) -> str:
    """Local JSON lookup — no network needed, no retry required."""
    try:
        term = search_term.lower()
        for stock in NIGERIAN_STOCKS:
            if term in stock["name"].lower() or term in stock["description"].lower():
                return f"Ticker: {stock['ticker']}  |  Company: {stock['description']}"
        return f"No stock found matching '{search_term}'. Ask the user to clarify."
    except Exception as exc:
        logger.error("lookup_nigerian_ticker failed: %s", exc)
        return f"Error looking up ticker: {exc}"


@tool(
    "search_stock_price",
    description=(
        "Get the current price for an exact TradingView ticker "
        "(e.g. 'NSENG:UBA'). If you don't know the exact ticker, "
        "call lookup_nigerian_ticker first."
    ),
)
def search_stock_price(ticker: str) -> str:
    """Live Nigerian stock price via TradingView Screener."""

    @retry(
        max_retries=2,
        fallback=lambda e: (
            f"Stock data for '{ticker}' temporarily unavailable. "
            f"Error: {e}"
        ),
    )
    def _fetch(t):
        cols = ["name", "description", "close", "change", "high", "low", "volume"]
        _, df = (
            Query()
            .select(*cols)
            .set_markets("nigeria")
            .set_tickers(t)
            .get_scanner_data()
        )
        if df.empty:
            return f"No data for '{t}'. Use lookup_nigerian_ticker first."
        return df.to_string(index=False)

    return _fetch(ticker)


# Build a rich description that embeds every stock name for the LLM.
_HISTORIC_TOOL_DESC = (
    "Retrieve **historic** price data for a Nigerian stock from the local "
    "SQLite database. Each database stores daily rows with columns: "
    "date (YYYY-MM-DD), volume, and price.\n\n"
    "Use this tool when the user asks about past/historical prices, price "
    "on a specific date, or price changes over a date range. For *live* "
    "current-day prices, use search_stock_price instead.\n\n"
    "**Args:**\n"
    "  stock_name (required): The exact company name as stored in the DB. "
    "Must be one of the names listed below.\n"
    "  start_date (optional): Start date in YYYY-MM-DD format. If only "
    "this is provided, returns the price on that single date.\n"
    "  end_date (optional): End date in YYYY-MM-DD format. When provided "
    "together with start_date, returns all prices in that range.\n"
    "  latest_n (optional): Integer — return the N most recent trading "
    "days. Ignored if start_date is provided. Defaults to 1.\n\n"
    "**Available stocks (use these exact names for stock_name):**\n"
    f"{_STOCK_LIST_STR}"
)


@tool("search_historic_stock_price", description=_HISTORIC_TOOL_DESC)
def search_historic_stock_price(
    stock_name: str,
    start_date: str = "",
    end_date: str = "",
    latest_n: int = 1,
) -> str:
    """Query a local SQLite DB for historic price data.

    Args:
        stock_name: Exact company name matching a .db file in db/.
        start_date: Optional YYYY-MM-DD — single date or range start.
        end_date:   Optional YYYY-MM-DD — range end (requires start_date).
        latest_n:   If no dates given, return the N most recent rows.

    Returns:
        Formatted string of date | price | volume rows.
    """

    @retry(
        max_retries=1,
        fallback=lambda e: (
            f"Historic price lookup failed for '{stock_name}'. Error: {e}"
        ),
    )
    def _query():
        # ---- resolve stock name ---------------------------------------------
        resolved = stock_name
        if resolved not in _AVAILABLE_STOCKS:
            term = stock_name.lower()
            matches = [n for n in _AVAILABLE_STOCKS if term in n.lower()]
            if len(matches) == 1:
                resolved = matches[0]
            elif len(matches) > 1:
                options = ", ".join(matches[:10])
                return (
                    f"Multiple stocks match '{stock_name}': {options}. "
                    "Please use the exact name."
                )
            else:
                return (
                    f"No stock database found for '{stock_name}'. "
                    "Use one of the exact names from the tool description."
                )

        # ---- query -----------------------------------------------------------
        if start_date and end_date:
            rows = _query_stock_db(
                resolved,
                "SELECT date, price, volume FROM prices WHERE date BETWEEN ? AND ? ORDER BY date ASC",
                (start_date, end_date),
                "SELECT date, price, volume FROM stock_prices WHERE stock_name = %s AND date BETWEEN %s AND %s ORDER BY date ASC",
                (resolved, start_date, end_date),
            )
        elif start_date:
            rows = _query_stock_db(
                resolved,
                "SELECT date, price, volume FROM prices WHERE date = ?",
                (start_date,),
                "SELECT date, price, volume FROM stock_prices WHERE stock_name = %s AND date = %s",
                (resolved, start_date),
            )
        else:
            rows = _query_stock_db(
                resolved,
                "SELECT date, price, volume FROM prices ORDER BY date DESC LIMIT ?",
                (latest_n,),
                "SELECT date, price, volume FROM stock_prices WHERE stock_name = %s ORDER BY date DESC LIMIT %s",
                (resolved, latest_n),
            )

        if rows is None:
            return f"No stock database found for '{stock_name}'."

        if not rows:
            if start_date and end_date:
                return (
                    f"No price data for '{stock_name}' between "
                    f"{start_date} and {end_date}."
                )
            elif start_date:
                return (
                    f"No price data for '{stock_name}' on {start_date}. "
                    "This may be a weekend or public holiday."
                )
            return f"No price data found for '{stock_name}'."

        # ---- format output --------------------------------------------------
        header = f"Historic prices for {stock_name}:\n"
        header += f"{'Date':<12} {'Price':>12} {'Volume':>14}\n"
        header += "-" * 40 + "\n"
        lines = []
        for row_date, price, volume in rows:
            lines.append(f"{row_date:<12} {price:>12} {volume:>14}")

        # Add summary stats for ranges
        if len(rows) > 1:
            prices = [float(r[1]) for r in rows if r[1]]
            if prices:
                lines.append("-" * 40)
                lines.append(
                    f"  Rows: {len(rows)}  |  "
                    f"High: {max(prices):.4f}  |  "
                    f"Low: {min(prices):.4f}  |  "
                    f"Start: {prices[0]:.4f}  |  "
                    f"End: {prices[-1]:.4f}"
                )
                change = prices[-1] - prices[0]
                pct = (change / prices[0]) * 100 if prices[0] else 0
                direction = "▲" if change >= 0 else "▼"
                lines.append(
                    f"  Change: {direction} {abs(change):.4f} "
                    f"({pct:+.2f}%)"
                )

        return header + "\n".join(lines)

    return _query()


# Build a description that embeds every stock name for the EMA tool.
_EMA_TOOL_DESC = (
    "Compute the **20-day EMA** and **50-day EMA** for a Nigerian stock, "
    "using historical daily closing prices from the local SQLite database. "
    "This is a single, self-contained tool call — it retrieves the data "
    "and computes the EMAs internally.\n\n"
    "Returns: the latest closing price, 20-EMA, 50-EMA, and whether the "
    "stock is trading ABOVE or BELOW each EMA (bullish/bearish signal).\n\n"
    "**Args:**\n"
    "  stock_name (required): The exact company name as stored in the DB. "
    "Must be one of the names listed below.\n\n"
    "**Available stocks (use these exact names for stock_name):**\n"
    f"{_STOCK_LIST_STR}"
)


@tool("compute_ema", description=_EMA_TOOL_DESC)
def compute_ema(stock_name: str) -> str:
    """Retrieve recent prices and compute 20-day & 50-day EMAs.

    Args:
        stock_name: Exact company name matching a .db file in db/.

    Returns:
        Formatted string with latest price, both EMAs, and trend signals.
    """

    @retry(
        max_retries=1,
        fallback=lambda e: (
            f"EMA computation failed for '{stock_name}'. Error: {e}"
        ),
    )
    def _compute():
        # ---- resolve stock name ---------------------------------------------
        resolved = stock_name
        if resolved not in _AVAILABLE_STOCKS:
            term = stock_name.lower()
            matches = [n for n in _AVAILABLE_STOCKS if term in n.lower()]
            if len(matches) == 1:
                resolved = matches[0]
            elif len(matches) > 1:
                options = ", ".join(matches[:10])
                return (
                    f"Multiple stocks match '{stock_name}': {options}. "
                    "Please use the exact name."
                )
            else:
                return (
                    f"No stock database found for '{stock_name}'. "
                    "Use one of the exact names from the tool description."
                )

        # ---- retrieve last 200 rows (oldest-first for EMA calc) -------------
        rows = _query_stock_db(
            resolved,
            "SELECT date, price FROM prices ORDER BY date DESC LIMIT 200",
            (),
            "SELECT date, price FROM stock_prices WHERE stock_name = %s ORDER BY date DESC LIMIT 200",
            (resolved,),
        )

        if rows is None or not rows:
            return f"No price data found for '{stock_name}'."

        # Reverse to chronological order (oldest first)
        rows.reverse()

        # Parse prices — skip rows with empty/invalid price
        dated_prices = []
        for row_date, price_str in rows:
            try:
                dated_prices.append((row_date, float(price_str)))
            except (ValueError, TypeError):
                continue

        if len(dated_prices) < 20:
            return (
                f"Only {len(dated_prices)} valid price rows for "
                f"'{stock_name}' — need at least 20 to compute EMAs."
            )

        # ---- compute EMAs ---------------------------------------------------
        prices = [p for _, p in dated_prices]

        def _ema(data: list[float], period: int) -> list[float]:
            """Return EMA series. First value = SMA of first *period* points."""
            if len(data) < period:
                return []
            multiplier = 2.0 / (period + 1)
            # Seed with SMA
            sma = sum(data[:period]) / period
            ema_values = [sma]
            for price in data[period:]:
                ema_values.append(
                    price * multiplier + ema_values[-1] * (1 - multiplier)
                )
            return ema_values

        ema_20 = _ema(prices, 20)
        ema_50 = _ema(prices, 50)

        latest_price = prices[-1]
        latest_date = dated_prices[-1][0]

        # ---- build output ---------------------------------------------------
        lines = [f"EMA Analysis for {stock_name} (as of {latest_date}):"]
        lines.append(f"  Data points used: {len(prices)}")
        lines.append(f"  Latest close: {latest_price:.4f}")
        lines.append("")

        if ema_20:
            ema_20_val = ema_20[-1]
            diff_20 = latest_price - ema_20_val
            pct_20 = (diff_20 / ema_20_val) * 100 if ema_20_val else 0
            signal_20 = "ABOVE ▲ (bullish)" if diff_20 >= 0 else "BELOW ▼ (bearish)"
            lines.append(f"  20-day EMA: {ema_20_val:.4f}")
            lines.append(f"    Price is {signal_20} the 20-EMA by {abs(diff_20):.4f} ({abs(pct_20):.2f}%)")
        else:
            lines.append("  20-day EMA: insufficient data (need ≥20 rows)")

        lines.append("")

        if ema_50:
            ema_50_val = ema_50[-1]
            diff_50 = latest_price - ema_50_val
            pct_50 = (diff_50 / ema_50_val) * 100 if ema_50_val else 0
            signal_50 = "ABOVE ▲ (bullish)" if diff_50 >= 0 else "BELOW ▼ (bearish)"
            lines.append(f"  50-day EMA: {ema_50_val:.4f}")
            lines.append(f"    Price is {signal_50} the 50-EMA by {abs(diff_50):.4f} ({abs(pct_50):.2f}%)")
        else:
            lines.append("  50-day EMA: insufficient data (need ≥50 rows)")

        # ---- crossover signal -----------------------------------------------
        if ema_20 and ema_50:
            lines.append("")
            if ema_20[-1] > ema_50[-1]:
                lines.append("  EMA Crossover: 20-EMA is ABOVE 50-EMA → bullish trend")
            elif ema_20[-1] < ema_50[-1]:
                lines.append("  EMA Crossover: 20-EMA is BELOW 50-EMA → bearish trend")
            else:
                lines.append("  EMA Crossover: 20-EMA equals 50-EMA → neutral")

            # Check if crossover happened recently (last 3 data points)
            if len(ema_20) >= 3 and len(ema_50) >= 3:
                # Align: ema_50 starts later, so offset ema_20
                offset = len(ema_20) - len(ema_50)
                if offset >= 0 and len(ema_50) >= 3:
                    for i in range(-3, 0):
                        e20_prev = ema_20[offset + len(ema_50) + i - 1]
                        e50_prev = ema_50[i - 1]
                        e20_curr = ema_20[offset + len(ema_50) + i]
                        e50_curr = ema_50[i]
                        if e20_prev <= e50_prev and e20_curr > e50_curr:
                            lines.append("  ⚡ GOLDEN CROSS detected in last 3 days!")
                            break
                        elif e20_prev >= e50_prev and e20_curr < e50_curr:
                            lines.append("  ⚡ DEATH CROSS detected in last 3 days!")
                            break

        return "\n".join(lines)

    return _compute()


@tool(
    "deep_market_research",
    description=(
        "Perform comprehensive, multi-step market research using Tavily's "
        "Research API. This is NOT a simple web search — it conducts "
        "iterative searches, analyses multiple sources, deduplicates "
        "findings, and generates a structured research report with "
        "citations.\n\n"
        "Use this when:\n"
        "  • The user asks a complex question that requires deep analysis\n"
        "  • A simple search_the_internet call returned insufficient or "
        "    shallow results\n"
        "  • The question involves trends, comparisons, regulatory "
        "    changes, or sector-wide analysis\n"
        "  • The user explicitly asks for 'detailed research' or "
        "    'in-depth analysis'\n\n"
        "Do NOT use this for simple factual lookups — use "
        "search_the_internet instead. This tool is slower and consumes "
        "more API credits.\n\n"
        "Args:\n"
        "  query: A detailed, well-formed research question. The more "
        "specific and contextual, the better the report.\n"
        "  sub_topic: Optional focus area to narrow the research "
        "(e.g. 'regulatory impact', 'financial performance', "
        "'competitive landscape').\n"
    ),
)
def deep_market_research(query: str, sub_topic: str = "") -> str:
    """Comprehensive research via Tavily Research API.

    Unlike the simple search tool, this performs autonomous multi-step
    research: multiple searches, source evaluation, deduplication, and
    report synthesis — all server-side.

    Args:
        query:     A detailed research question.
        sub_topic: Optional focus area to narrow the research scope.

    Returns:
        A structured research report with findings and source URLs.
    """

    def _format_report(result: dict) -> str:
        """Format the raw Tavily research result into a clean report."""
        lines = ["═" * 60]
        lines.append("  DEEP MARKET RESEARCH REPORT")
        lines.append("═" * 60)

        # Main content / report body
        content = result.get("content", "")
        if content:
            lines.append("")
            lines.append(content)

        # Sources
        sources = result.get("sources", [])
        if sources:
            lines.append("")
            lines.append("─" * 60)
            lines.append(f"  SOURCES ({len(sources)} references)")
            lines.append("─" * 60)
            for i, src in enumerate(sources, 1):
                if isinstance(src, dict):
                    title = src.get("title", "Untitled")
                    url = src.get("url", "")
                    lines.append(f"  [{i}] {title}")
                    if url:
                        lines.append(f"      {url}")
                else:
                    lines.append(f"  [{i}] {src}")

        lines.append("")
        lines.append("═" * 60)
        return "\n".join(lines)

    # Enrich the query with sub-topic context if provided
    research_input = query
    if sub_topic:
        research_input = f"{query} — Focus area: {sub_topic}"

    max_attempts = tavily_pool.keys_remaining + 2
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            tavily_client = tavily_pool.get_client()

            logger.info(
                "Starting deep research (key #%d, attempt %d): '%s' "
                "(sub_topic=%s)",
                tavily_pool.active_key_number, attempt,
                query, sub_topic or "none",
            )

            # Step 1: Create the research task
            response = tavily_client.research(
                input=research_input,
                model="pro",
            )

            request_id = response.get("request_id")

            if not request_id:
                # Some SDK versions return the result directly
                if response.get("content"):
                    return _format_report(response)
                return (
                    "Research task created but no request_id returned. "
                    "Raw response: " + json.dumps(response, default=str)[:500]
                )

            # Step 2: Poll for completion
            max_wait = TAVILY_RESEARCH_MAX_WAIT
            poll_interval = TAVILY_RESEARCH_POLL_INTERVAL
            elapsed = 0

            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                result = tavily_client.get_research(request_id)
                status = result.get("status", "").lower()

                logger.debug(
                    "Research poll: status=%s, elapsed=%ds/%ds",
                    status, elapsed, max_wait,
                )

                if status == "completed":
                    return _format_report(result)
                elif status in ("failed", "error"):
                    error_msg = result.get("error", "Unknown error")
                    return f"Research task failed: {error_msg}"

            return (
                f"Research timed out after {max_wait}s. "
                "The query may be too broad — try narrowing it down or "
                "use search_the_internet for a quicker result."
            )

        except Exception as exc:
            last_error = exc
            if tavily_pool.should_rotate(exc):
                if tavily_pool.rotate(exc):
                    logger.info(
                        "deep_market_research: rotated key, retrying "
                        "(attempt %d)…", attempt,
                    )
                    continue
                else:
                    return (
                        "All Tavily API keys exhausted. "
                        "Deep research unavailable."
                    )
            else:
                delay = 2 ** attempt
                logger.warning(
                    "deep_market_research failed (attempt %d): %s "
                    "— retrying in %ds…", attempt, exc, delay,
                )
                time.sleep(delay)

    return (
        "Deep research unavailable — please try a simpler "
        f"search_the_internet query instead. Last error: {last_error}"
    )


@tool(
    "send_telegram_message",
    description=(
        "Sends a text message to the user's Telegram chat."
    ),
)
def send_telegram_message(chat_id: str, content: str) -> str:
    """Send *content* to a specific Telegram *chat_id*.

    Args:
        chat_id: The Telegram chat ID to send the message to.
        content: The message text to send.

    Returns:
        A confirmation or error string so the agent knows what happened.
    """

    @retry(
        max_retries=3,
        fallback=lambda e: (
            "Telegram unavailable — message NOT sent. "
            f"Error: {e}"
        ),
    )
    def _send(cid, text):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": cid, "text": text}
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return "Message sent successfully!"

    return _send(chat_id, content)


@tool(
    "escalate_to_opus",
    description=(
        "Route a complex reasoning task to Claude Opus 4.7 with extended "
        "thinking. This is the MOST EXPENSIVE tool — it must ONLY be "
        "triggered when one of the specific conditions below is met.\n\n"
        "**TRIGGER CONDITIONS (must match at least one):**\n"
        "  1. COMPARISON: The user asks to compare, rank, or evaluate "
        "3 or more stocks/companies simultaneously.\n"
        "  2. CONFLICTING EVIDENCE: Your tool results contradict each "
        "other — e.g. knowledge base is bullish but EMA shows bearish, "
        "or two news sources give opposite signals.\n"
        "  3. UNEXPLAINED MOVE: A stock moved >5% and after searching "
        "both the knowledge base AND internet, you found NO clear "
        "catalyst that explains the magnitude.\n"
        "  4. EXPLICIT REQUEST: The user literally says 'deep analysis', "
        "'detailed report', 'investment thesis', 'bull/bear case', or "
        "'compare X vs Y in detail'.\n"
        "  5. FORWARD-LOOKING: The user asks you to project, forecast, "
        "or assess future impact of a policy/event on markets.\n"
        "  6. MULTI-FACTOR: The question involves 3+ interacting "
        "factors (e.g. FX rates + interest rates + earnings + "
        "regulation all affecting one stock).\n\n"
        "**DO NOT TRIGGER if:**\n"
        "  • You can answer directly from a single tool's output\n"
        "  • The question is a simple factual lookup or price check\n"
        "  • Only 1-2 stocks are involved with clear evidence\n\n"
        "**CONTEXT FORMAT — you MUST structure it exactly like this:**\n"
        "  ```\n"
        "  ## From search_knowledge_base:\n"
        "  [paste the FULL raw output from that tool call]\n\n"
        "  ## From search_the_internet:\n"
        "  [paste the FULL raw output]\n\n"
        "  ## From compute_ema (TICKER):\n"
        "  [paste the FULL raw output]\n\n"
        "  ## From search_stock_price (TICKER):\n"
        "  [paste the FULL raw output]\n"
        "  ```\n"
        "  Include EVERY tool output you have gathered. Do NOT "
        "summarise or paraphrase — paste the raw text. Opus needs "
        "the original data to reason over, not your interpretation.\n\n"
        "Args:\n"
        "  question: The user's original question, word-for-word.\n"
        "  context: ALL raw tool outputs gathered so far, structured "
        "as shown above. One section per tool call, labelled with the "
        "tool name.\n"
    ),
)
def escalate_to_opus(question: str, context: str) -> str:
    """Escalate a complex reasoning task to Claude Opus with thinking.

    This tool calls the Anthropic API directly with extended thinking
    enabled, giving Opus a large thinking budget to work through
    difficult analytical problems.

    Args:
        question: The complex question or task.
        context:  All evidence and data gathered by prior tool calls.

    Returns:
        Opus's deep analysis as a string.
    """

    @retry(
        max_retries=2,
        fallback=lambda e: (
            "Opus escalation failed — falling back to standard analysis. "
            f"Error: {e}"
        ),
    )
    def _escalate(q, ctx):
        logger.info(
            "Escalating to Opus (model=%s, thinking_budget=%d): '%s'",
            OPUS_MODEL, OPUS_THINKING_BUDGET, q[:100],
        )

        prompt = (
            f"You are an expert Nigerian financial analyst. A junior "
            f"analyst has gathered the following evidence and needs your "
            f"deep analytical expertise.\n\n"
            f"## QUESTION\n{q}\n\n"
            f"## EVIDENCE & DATA GATHERED\n{ctx}\n\n"
            f"## YOUR TASK\n"
            f"Provide a thorough, deeply-reasoned analysis. Think "
            f"through this step by step. Consider:\n"
            f"- Materiality and magnitude of each piece of evidence\n"
            f"- Timing and causality\n"
            f"- Conflicting signals and how to weigh them\n"
            f"- Second and third-order effects\n"
            f"- What the evidence does NOT explain\n"
            f"- Your confidence level and key uncertainties\n\n"
            f"Be precise, quantitative where possible, and do NOT "
            f"hedge excessively. Give a clear analytical verdict."
        )

        response = _call_opus(prompt)

        # Extract text blocks from the response (skip thinking blocks)
        output_parts = []
        for block in response.content:
            if block.type == "text":
                output_parts.append(block.text)

        if not output_parts:
            return "Opus returned no text output."

        result = "\n".join(output_parts)

        # Log token usage
        usage = response.usage
        logger.info(
            "Opus response: %d input tokens, %d output tokens",
            usage.input_tokens, usage.output_tokens,
        )

        return f"═══ OPUS DEEP ANALYSIS ═══\n\n{result}\n\n═══ END OPUS ANALYSIS ═══"

    return _escalate(question, context)


# ---------------------------------------------------------------------------
# System Prompt — rebuilt on each import so the date is always fresh
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """Build the system prompt with the current date injected."""
    today = date.today().isoformat()  # e.g. "2026-05-19"
    return f"""\
You are a Nigerian financial reasoning agent designed to explain why a stock is moving at a given moment.

Today's date is **{today}** (UTC). Use this when the user refers to relative
timeframes like "last 2 months" or "since March".

    You have access to:
    1. A knowledge base of Nairametrics articles
    2. Live stock prices via TradingView (search_stock_price)
    3. Historical stock prices from local databases (search_historic_stock_price)
    4. Internet search for additional or recent information
    5. Deep market research via Tavily Research API (deep_market_research)

    Your primary goal:
    “Explain why a stock is moving right now, using strong, evidence-based reasoning.”

    ---

    ### CORE PRINCIPLE
    Do NOT assume that available news explains the stock move.
    Your job is to evaluate whether the evidence is sufficient — not just to summarize it.

    ---

### TOOL USAGE

1. Start with the knowledge base for Nigerian financial topics.
   - The search_knowledge_base tool accepts an optional **cutoff_date**
     parameter (YYYY-MM-DD). When the user mentions a timeframe (e.g.
     "in the last 3 months", "since January", "after March 13th"),
     calculate the corresponding date from today ({today}) and pass it
     as cutoff_date so only recent articles are returned.
   - If no timeframe is mentioned, omit cutoff_date (a sensible default
     is applied automatically).
2. After retrieving results, evaluate sufficiency:
   - If the information is weak, outdated, or not clearly explanatory -> use internet search.
3. Use both sources when necessary and clearly distinguish between them.
4. For **current/live** stock prices:
   - Use lookup_nigerian_ticker if unsure of the ticker
   - Then use search_stock_price
5. For **historical** stock prices (past dates, date ranges, comparisons):
   - Use search_historic_stock_price directly with the company name.
   - The tool description lists every available stock name.
   - You can query a single date, a date range, or the latest N days.
6. For **technical trend analysis** (is a stock bullish or bearish?):
   - Use compute_ema with the company name.
   - Returns 20-day and 50-day EMAs, above/below signals, and crossover detection.
   - One tool call — no need to fetch prices first.
7. For **deep research** (complex questions, trends, regulatory analysis):
   - Use deep_market_research when simple searches are insufficient.
   - This tool performs multi-step research: multiple searches, source
     evaluation, deduplication, and report synthesis.
   - It is SLOWER and costs more credits — only use when the question
     genuinely requires comprehensive analysis.
   - Good for: sector analysis, regulatory changes, competitive landscapes,
     multi-factor comparisons, or when the user explicitly requests depth.
   - Bad for: simple factual lookups or current prices.
8. For **complex reasoning** — escalate to Opus (escalate_to_opus):
   **ONLY trigger when at least one condition is true:**
   a. COMPARISON: User asks to compare/rank 3+ stocks simultaneously.
   b. CONFLICTING EVIDENCE: Your tool results contradict each other.
   c. UNEXPLAINED MOVE: Stock moved >5%%, no clear catalyst found after
      searching both knowledge base AND internet.
   d. EXPLICIT REQUEST: User literally says "deep analysis", "investment
      thesis", "bull/bear case", or "compare X vs Y in detail".
   e. FORWARD-LOOKING: User asks to project or forecast future impact.
   f. MULTI-FACTOR: Question involves 3+ interacting factors.

   **Workflow:** Gather ALL evidence first → paste raw tool outputs into
   the context parameter (do NOT summarise) → let Opus reason.

   **Never trigger for:** single-stock lookups, price checks, simple
   article summaries, or questions answerable from one tool's output.
9. Do NOT overuse tools -- only call them when they improve your explanation.

    ---

    ### CAUSAL ANALYSIS RULES

    When evaluating why a stock moved, apply these checks:

    1. MATERIALITY
    - Does the event affect revenue, earnings, regulation, guidance, or market structure?
    - Ignore generic or low-impact news.

    2. MAGNITUDE
    - Is the news strong enough to explain the size of the price move?
    - Large moves require strong catalysts.

    3. TIMING
    - Did the event occur before or during the price movement?
    - Ignore explanations that occur after the move.

    4. MARKET CONTEXT
    - Check if:
        - the broader market moved
        - the sector moved
        - macro factors (rates, FX, inflation) played a role

    5. MULTIPLE CAUSES
    - Consider that moves may have multiple drivers.
    - Weigh their relative importance.

    ---

    ### SUFFICIENCY CHECK (CRITICAL)

    Before giving a final answer, ask:

    - Is there a clear, strong, and timely catalyst?
    - Does the evidence convincingly explain the magnitude of the move?

    If NOT:
    → Explicitly say:

    "There is no strong evidence that fully explains this price movement."

    Then provide:
    - the most plausible partial explanations
    - what might be missing (e.g., institutional flows, breaking news, sentiment shifts)

    Do NOT force a conclusion.

    ---

    ### HALLUCINATION PREVENTION

    - Do NOT invent causes.
    - Do NOT treat correlation as causation.
    - Do NOT over-explain weak or generic news.
    - If evidence is limited, say so clearly.

    ---

    ### RESPONSE STRUCTURE

    1. **Summary (1–2 sentences)**
    - Direct answer or state if no strong explanation exists

    2. **Key Evidence**
    - Bullet points with:
        - specific data
        - dates
        - source attribution (knowledge base vs internet)

    3. **Causal Analysis**
    - Explain why the evidence does or does not explain the move
    - Reference materiality, magnitude, and timing

    4. **Other Possible Factors (if applicable)**
    - Macro, sector, sentiment, or technical drivers

    5. **Conclusion + Confidence**
    - High / Medium / Low confidence
    - Clearly state uncertainty if present

    ---

    ### CONVERSATIONAL RULES

    - Ask clarifying questions if needed
    - Be concise but thorough
    - Avoid generic “analyst-sounding” filler language
    - Do NOT respond like a bot — sound like a sharp financial analyst
    - Do NOT default to confident explanations — accuracy over completeness
    - Do NOT use emojis unless absolutely necessary for clarity

    ---

    ### FAILURE MODE TO AVOID (VERY IMPORTANT)

    A bad response:
    “The stock likely moved due to positive sentiment from recent developments.”

    A good response:
    “The available news is not sufficiently material to explain a 7% move. No clear catalyst identified. This suggests either missing information or broader market-driven movement.”

    ---

    Your priority is correctness, not completeness.
    It is better to say “insufficient evidence” than to give a weak or incorrect explanation.
    """


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

checkpoint = InMemorySaver()

agent = create_agent(
    model=model,
    system_prompt=_build_system_prompt(),
    checkpointer=checkpoint,
    tools=[
        search_knowledge_base,
        search_the_internet,
        deep_market_research,
        lookup_nigerian_ticker,
        search_stock_price,
        search_historic_stock_price,
        compute_ema,
        escalate_to_opus,
        send_telegram_message,
    ],
)


# ---------------------------------------------------------------------------
# Main Loop — for terminal testing (like robust_agent.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Research agent started.")
    config = {"configurable": {"thread_id": "terminal-test"}}

    while True:
        try:
            user_input = input("Human: ")
            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            print(f"AI: {response['messages'][-1].content}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            logger.info("Session ended by KeyboardInterrupt.")
            break

        except Exception as exc:
            logger.error(
                "Unhandled error in agent loop:\n%s",
                traceback.format_exc(),
            )
            print(
                f"\n⚠️  Something went wrong: {exc}\n"
                "The error has been logged. You can keep chatting.\n"
            )
