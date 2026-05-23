"""
FastAPI server — webhook, bot lifecycle, Mini App static files.

This is the single entry point. It handles:
    1. Telegram webhook (POST /webhook)
    2. Mini App API (mounted from api.py)
    3. Mini App static files (GET /app/*)
    4. Health check for UptimeRobot (GET /health)

Run locally:
    uvicorn research_agent.main:app --reload --port 8000

Run with ngrok for dev:
    ngrok http 8000
    Then set the webhook URL in BotFather.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from research_agent.agent import agent
from research_agent.api import router as api_router
from research_agent.config import (
    MINI_APP_DIR,
    TELEGRAM_BOT_TOKEN,
    WEBHOOK_SECRET,
)
from research_agent.reformulator import reformulate
from research_agent.sessions.database import init_db
from research_agent.sessions.manager import SessionManager

logger = logging.getLogger("research_agent.main")

sm = SessionManager()

# Default timeout for agent.invoke() — prevents infinite hangs
_AGENT_TIMEOUT = 120  # seconds


# ---------------------------------------------------------------------------
# Bot Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context):
    """Handle /start — welcome message."""
    await update.message.reply_text(
        "👋 I'm your Nigerian financial research assistant.\n\n"
        "Ask me anything about Nigerian markets, banking, economy, "
        "stock prices, or policy.\n\n"
        "Commands:\n"
        "  /new   — Start a fresh research session\n"
        "  /clear — Wipe current session history\n"
        "  /help  — Show this message\n\n"
        "Tap 📚 Sessions in the menu to browse your history."
    )


async def cmd_help(update: Update, context):
    """Handle /help — same as /start."""
    await cmd_start(update, context)


async def cmd_new(update: Update, context):
    """Handle /new — start a fresh session."""
    user_id = update.effective_user.id
    await sm.create_session(user_id)
    await update.message.reply_text(
        "✅ New session started. What would you like to research?"
    )


async def cmd_clear(update: Update, context):
    """Handle /clear — wipe current session."""
    user_id = update.effective_user.id
    session = await sm.get_active_session(user_id)
    if session:
        await sm.clear_session(session.id)
        await update.message.reply_text("🧹 Session cleared.")
    else:
        await update.message.reply_text("No active session to clear.")


async def handle_message(update: Update, context):
    """Handle regular text messages — the main research flow.

    Flow:
        1. Load (or create) active session
        2. Get conversation context (last 20 messages + summary)
        3. Reformulate the question (resolve pronouns)
        4. Run the agent
        5. Save both messages
        6. Auto-title if first exchange
        7. Maybe update rolling summary
    """
    user_id = update.effective_user.id
    user_msg = update.message.text

    if not user_msg or not user_msg.strip():
        return

    # 1. Session
    session = await sm.get_or_create_active(user_id)

    # 2. Context
    context_msgs = await sm.get_context_window(session.id)
    summary = await sm.get_summary(session.id)

    # 3. Reformulate (make vague follow-ups standalone)
    try:
        standalone_q = await asyncio.to_thread(
            reformulate, user_msg, context_msgs, summary
        )
    except Exception as exc:
        logger.warning("Reformulation failed: %s — using raw message", exc)
        standalone_q = user_msg

    # 4. Run agent
    thread_config = {"configurable": {"thread_id": session.id}}

    # Build the messages list: context + the reformulated question
    messages_for_agent = []
    if summary:
        messages_for_agent.append({
            "role": "system",
            "content": f"Conversation summary so far: {summary}",
        })
    for m in context_msgs:
        messages_for_agent.append(m)
    messages_for_agent.append({"role": "user", "content": standalone_q})

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                agent.invoke,
                {"messages": messages_for_agent},
                config=thread_config,
            ),
            timeout=_AGENT_TIMEOUT,
        )
        bot_reply = response["messages"][-1].content
    except asyncio.TimeoutError:
        logger.error("Agent timed out after %ds", _AGENT_TIMEOUT)
        bot_reply = (
            "⚠️ The research took too long and was stopped. "
            "Try a more specific question."
        )
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        bot_reply = (
            "⚠️ Something went wrong while researching your question. "
            "Please try again."
        )

    # 5. Save messages
    await sm.add_message(session.id, "user", user_msg)
    await sm.add_message(session.id, "assistant", bot_reply)

    # 6. Auto-title on first exchange
    msg_count = await sm.get_message_count(session.id)
    if msg_count == 2:   # first user + first assistant
        await sm.auto_title(session.id, user_msg, bot_reply)

    # 7. Maybe update summary
    await sm.maybe_update_summary(session.id)

    # 8. Reply
    # Telegram has a 4096 char limit — split at whitespace boundaries
    if len(bot_reply) <= 4096:
        await update.message.reply_text(bot_reply)
    else:
        chunks = _split_message(bot_reply, 4096)
        for chunk in chunks:
            await update.message.reply_text(chunk)


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split a long message into chunks at whitespace boundaries."""
    chunks = []
    while len(text) > max_len:
        # Find last whitespace before the limit
        split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len  # No whitespace found, hard split
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        chunks.append(text)
    return chunks


# ---------------------------------------------------------------------------
# FastAPI Lifespan — init bot + DB on startup, cleanup on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the Telegram bot and database on startup."""
    # Force a startup banner to stdout so Railway always shows something
    print("[STARTUP] Nairametrics Research Agent starting…", flush=True)

    # 1. Init database
    try:
        await init_db()
        logger.info("Database initialised.")
        print("[STARTUP] Database initialised.", flush=True)
    except Exception as exc:
        logger.error("Database init failed: %s", exc, exc_info=True)
        print(f"[STARTUP] Database init FAILED: {exc}", flush=True)
        # Non-fatal — health check can still respond

    # 2. Build Telegram bot application
    bot_app = None
    try:
        bot_app = (
            Application.builder()
            .token(TELEGRAM_BOT_TOKEN)
            .build()
        )

        # 3. Register handlers
        bot_app.add_handler(CommandHandler("start", cmd_start))
        bot_app.add_handler(CommandHandler("help", cmd_help))
        bot_app.add_handler(CommandHandler("new", cmd_new))
        bot_app.add_handler(CommandHandler("clear", cmd_clear))
        bot_app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )

        # 4. Initialise and start the bot
        await bot_app.initialize()
        await bot_app.start()

        # Store in app state so the webhook endpoint can access it
        app.state.bot_app = bot_app
        app.state.bot = bot_app.bot
        logger.info("Telegram bot started.")
        print("[STARTUP] Telegram bot started.", flush=True)
    except Exception as exc:
        logger.error("Telegram bot init failed: %s", exc, exc_info=True)
        print(f"[STARTUP] Telegram bot init FAILED: {exc}", flush=True)
        # Store None so webhook can check and return 503 gracefully
        app.state.bot_app = None
        app.state.bot = None

    print("[STARTUP] Ready to serve requests.", flush=True)
    yield

    # 5. Cleanup
    if bot_app is not None:
        try:
            await bot_app.stop()
            await bot_app.shutdown()
        except Exception:
            pass
    logger.info("Telegram bot stopped.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nairametrics Research Agent",
    lifespan=lifespan,
)

# CORS — the Mini App WebView may need this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Mini App API
app.include_router(api_router)

# Mount Mini App static files (non-fatal if directory is missing)
try:
    app.mount("/app", StaticFiles(directory=str(MINI_APP_DIR), html=True), name="mini-app")
    logger.info("Mini App static files mounted from %s", MINI_APP_DIR)
except Exception as exc:
    logger.warning("Could not mount Mini App static files: %s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check for Railway / UptimeRobot keep-alive pings."""
    bot_ok = getattr(app.state, "bot_app", None) is not None
    return {"status": "ok", "bot": "connected" if bot_ok else "unavailable"}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(None),
):
    """Receive Telegram updates via webhook."""

    # Verify the secret token
    if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret token")

    # Guard: bot must be initialised
    bot_app = getattr(app.state, "bot_app", None)
    bot = getattr(app.state, "bot", None)
    if bot_app is None or bot is None:
        raise HTTPException(status_code=503, detail="Bot not initialised")

    # Parse the update and put it in the bot's queue
    data = await request.json()
    update = Update.de_json(data, bot)
    await bot_app.update_queue.put(update)

    return {"ok": True}
