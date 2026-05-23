"""
Telegram Mini App authentication — HMAC-SHA256 validation.

When a user opens the Mini App, Telegram provides initData (a signed
query string). We validate it server-side to confirm the request is
genuine and extract the user ID.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from urllib.parse import unquote

from fastapi import HTTPException, Request

from research_agent.config import TELEGRAM_BOT_TOKEN

logger = logging.getLogger("research_agent.auth")

# Explicitly check APP_ENV — default is "prod" for safety
_APP_ENV = os.getenv("APP_ENV", "prod").lower()
_IS_DEV = _APP_ENV == "dev"

if _IS_DEV:
    logger.warning(
        "AUTH: Running in dev mode (APP_ENV=dev). "
        "Auth bypasses are enabled — do NOT use this in production."
    )


def validate_init_data(init_data_str: str, bot_token: str,
                       max_age_seconds: int = 3600) -> dict | None:
    """Validate Telegram Mini App initData and return the user dict.

    Args:
        init_data_str:    The raw query string from the X-Telegram-Init-Data header.
        bot_token:        Your bot's API token.
        max_age_seconds:  Reject data older than this (default 1 hour).

    Returns:
        Parsed user dict (id, first_name, etc.) or None if invalid.
    """
    if not init_data_str:
        return None

    try:
        # 1. Parse the query string
        params = dict(
            item.split("=", 1) for item in init_data_str.split("&") if "=" in item
        )
        received_hash = params.pop("hash", None)
        if not received_hash:
            return None

        # 2. Check auth_date freshness
        auth_date = int(params.get("auth_date", 0))
        if time.time() - auth_date > max_age_seconds:
            return None

        # 3. Sort remaining params and build data-check string
        sorted_params = sorted(params.items())
        data_check_string = "\n".join(
            f"{k}={unquote(v)}" for k, v in sorted_params
        )

        # 4. Generate secret key: HMAC-SHA256("WebAppData", bot_token)
        secret_key = hmac.new(
            "WebAppData".encode(), bot_token.encode(), hashlib.sha256
        ).digest()

        # 5. Compute HMAC-SHA256(secret_key, data_check_string)
        computed_hash = hmac.new(
            secret_key, data_check_string.encode(), hashlib.sha256
        ).hexdigest()

        # 6. Constant-time comparison
        if not hmac.compare_digest(computed_hash, received_hash):
            return None

        # 7. Parse the user object
        user_str = unquote(params.get("user", "{}"))
        user = json.loads(user_str)
        return user

    except Exception:
        return None


async def verify_telegram_auth(request: Request) -> int:
    """FastAPI dependency — extracts and validates the Telegram user ID.

    Reads the X-Telegram-Init-Data header, validates the HMAC signature,
    and returns the authenticated user's Telegram ID.

    In dev mode (APP_ENV=dev), if validation fails or no header is
    provided, falls back to extracting the user ID from raw init data
    or returns a fixed dev user ID (12345).

    Raises:
        HTTPException 401 if validation fails (production only).
    """
    init_data = request.headers.get("X-Telegram-Init-Data", "")

    # No init data at all
    if not init_data:
        if _IS_DEV:
            return 12345  # Fixed dev user ID for local browser testing
        raise HTTPException(status_code=401, detail="Missing Telegram auth")

    # Try proper HMAC validation first
    user = validate_init_data(init_data, TELEGRAM_BOT_TOKEN)
    if user and "id" in user:
        return user["id"]

    # Validation failed — in dev mode, try to extract user ID from raw data
    if _IS_DEV:
        try:
            params = dict(
                item.split("=", 1) for item in init_data.split("&") if "=" in item
            )
            user_str = unquote(params.get("user", "{}"))
            user_data = json.loads(user_str)
            if "id" in user_data:
                return user_data["id"]
        except Exception:
            pass
        return 12345  # Final fallback for dev

    raise HTTPException(status_code=401, detail="Invalid Telegram auth")
