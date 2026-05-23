"""
Mini App REST API — endpoints for the Telegram WebView frontend.

All endpoints validate Telegram initData via the verify_telegram_auth
dependency. The Mini App frontend calls these to list sessions, browse
history, switch sessions, etc.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from research_agent.auth import verify_telegram_auth
from research_agent.sessions.manager import SessionManager

router = APIRouter(prefix="/api", tags=["mini-app"])

sm = SessionManager()
logger = logging.getLogger("research_agent.api")

# Default timeout for agent.invoke() — prevents infinite hangs
_AGENT_TIMEOUT = 120  # seconds

# Tool name → user-friendly label for streaming status indicators
_TOOL_LABELS = {
    "search_knowledge_base": "Searching knowledge base",
    "search_the_internet": "Searching the internet",
    "deep_market_research": "Conducting deep research",
    "lookup_nigerian_ticker": "Looking up ticker",
    "search_stock_price": "Fetching live price",
    "search_historic_stock_price": "Fetching historic prices",
    "compute_ema": "Computing EMA",
    "escalate_to_opus": "Deep analysis with Opus",
    "send_telegram_message": "Sending Telegram message",
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RenameRequest(BaseModel):
    title: str

class MessageRequest(BaseModel):
    content: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _verify_session_ownership(session_id: str, user_id: int):
    """Ensure the session belongs to the authenticated user."""
    session = await sm.load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return session


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


async def _build_agent_messages(session_id: str, user_content: str):
    """Build the context + reformulated message list for the agent.

    Shared by both the streaming and non-streaming endpoints.
    """
    from research_agent.reformulator import reformulate

    context_msgs = await sm.get_context_window(session_id)
    summary = await sm.get_summary(session_id)

    # Reformulate in a thread (sync function)
    try:
        loop = asyncio.get_running_loop()
        refined_query = await loop.run_in_executor(
            None, lambda: reformulate(user_content, context_msgs, summary)
        )
    except Exception:
        refined_query = user_content

    messages_for_agent = []
    if summary:
        messages_for_agent.append({
            "role": "system",
            "content": f"Conversation summary so far: {summary}",
        })
    for m in context_msgs:
        messages_for_agent.append(m)
    messages_for_agent.append({"role": "user", "content": refined_query})

    return messages_for_agent


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, body: MessageRequest,
                       user_id: int = Depends(verify_telegram_auth)):
    """Accept a user message, run the agent, return the AI response."""
    from research_agent.agent import agent

    # Verify ownership
    await _verify_session_ownership(session_id, user_id)

    # 1. Save the user message
    await sm.add_message(session_id, "user", body.content)

    # 2. Build messages
    messages_for_agent = await _build_agent_messages(session_id, body.content)

    # 3. Run the research agent with timeout
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: agent.invoke(
                    {"messages": messages_for_agent},
                    config={"configurable": {"thread_id": session_id}},
                )
            ),
            timeout=_AGENT_TIMEOUT,
        )
        reply = result["messages"][-1].content
    except asyncio.TimeoutError:
        logger.error("Agent timed out after %ds for session %s", _AGENT_TIMEOUT, session_id)
        reply = "The research took too long. Try a more specific question."
    except Exception as e:
        logger.error("Agent error for session %s: %s", session_id, e, exc_info=True)
        # Sanitize — don't leak internals to the client
        reply = "I encountered an error while researching. Please try again."

    # 4. Save the assistant response
    await sm.add_message(session_id, "assistant", reply)

    # 5. Auto-title on first exchange
    msg_count = await sm.get_message_count(session_id)
    if msg_count <= 2:
        try:
            await sm.auto_title(session_id, body.content, reply)
        except Exception:
            pass

    # 6. Maybe update summary
    await sm.maybe_update_summary(session_id)

    return {
        "message": {
            "role": "assistant",
            "content": reply,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }


# ---------------------------------------------------------------------------
# SSE Streaming Endpoint
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(session_id: str, body: MessageRequest,
                         user_id: int = Depends(verify_telegram_auth)):
    """Stream the agent's response token-by-token via Server-Sent Events.

    SSE event types:
        token      — {"content": "..."} — a text delta from the LLM
        tool_start — {"tool": "...", "label": "..."} — tool invocation began
        tool_end   — {"tool": "..."} — tool invocation finished
        done       — {"content": "..."} — final complete message
        error      — {"message": "..."} — something went wrong
    """
    from research_agent.agent import agent

    # Verify ownership (must happen before we return the StreamingResponse)
    await _verify_session_ownership(session_id, user_id)

    # Save the user message immediately
    await sm.add_message(session_id, "user", body.content)

    # Build context
    messages_for_agent = await _build_agent_messages(session_id, body.content)

    async def sse_generator():
        """Async generator that yields SSE events."""
        full_response = ""

        try:
            async for event in agent.astream_events(
                {"messages": messages_for_agent},
                config={"configurable": {"thread_id": session_id}},
                version="v2",
            ):
                kind = event.get("event", "")
                meta = event.get("metadata", {})

                # --- Token deltas from the final LLM response ----------------
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Only stream tokens from the agent node, not from
                        # internal tool-calling sub-chains
                        node = meta.get("langgraph_node", "")
                        if node == "agent":
                            content = chunk.content
                            # Handle string content
                            if isinstance(content, str) and content:
                                full_response += content
                                yield _sse_event("token", {"content": content})
                            # Handle list content (some models return list of dicts)
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text = block.get("text", "")
                                        if text:
                                            full_response += text
                                            yield _sse_event("token", {"content": text})

                # --- Tool lifecycle ------------------------------------------
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    label = _TOOL_LABELS.get(tool_name, f"Running {tool_name}")
                    yield _sse_event("tool_start", {
                        "tool": tool_name,
                        "label": label,
                    })

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    yield _sse_event("tool_end", {"tool": tool_name})

        except asyncio.TimeoutError:
            logger.error("Stream timed out for session %s", session_id)
            yield _sse_event("error", {
                "message": "The research took too long. Try a more specific question."
            })
            full_response = full_response or "The research took too long. Try a more specific question."

        except Exception as e:
            logger.error("Stream error for session %s: %s", session_id, e, exc_info=True)
            yield _sse_event("error", {
                "message": "I encountered an error while researching. Please try again."
            })
            full_response = full_response or "I encountered an error while researching. Please try again."

        # --- Finalize --------------------------------------------------------
        # If no tokens were streamed (e.g. the agent returned a direct message
        # without streaming), extract from the last message
        if not full_response:
            # The agent may have completed via invoke-style internally
            full_response = "I wasn't able to generate a response. Please try again."

        yield _sse_event("done", {"content": full_response})

        # Save the complete response to the session
        await sm.add_message(session_id, "assistant", full_response)

        # Auto-title on first exchange
        msg_count = await sm.get_message_count(session_id)
        if msg_count <= 2:
            try:
                await sm.auto_title(session_id, body.content, full_response)
            except Exception:
                pass

        # Maybe update summary
        await sm.maybe_update_summary(session_id)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/sessions")
async def list_sessions(page: int = 0,
                        user_id: int = Depends(verify_telegram_auth)):
    """Return a paginated list of the user's sessions."""
    sessions = await sm.list_sessions(user_id, page=page)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str,
                       user_id: int = Depends(verify_telegram_auth)):
    """Return all messages for a session (full chat history)."""
    await _verify_session_ownership(session_id, user_id)
    messages = await sm.get_messages(session_id)
    return {"messages": messages}


@router.post("/sessions", status_code=201)
async def create_session(user_id: int = Depends(verify_telegram_auth)):
    """Create a new session and make it active."""
    session = await sm.create_session(user_id)
    return {"id": session.id, "title": session.title}


@router.post("/sessions/{session_id}/switch")
async def switch_session(session_id: str,
                         user_id: int = Depends(verify_telegram_auth)):
    """Switch the active session."""
    await _verify_session_ownership(session_id, user_id)
    await sm.switch_session(user_id, session_id)
    session = await sm.load_session(session_id)
    title = session.title if session else "Unknown"
    return {"ok": True, "title": title}


@router.put("/sessions/{session_id}/rename")
async def rename_session(session_id: str, body: RenameRequest,
                         user_id: int = Depends(verify_telegram_auth)):
    """Rename a session."""
    await _verify_session_ownership(session_id, user_id)
    await sm.rename_session(session_id, body.title)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str,
                         user_id: int = Depends(verify_telegram_auth)):
    """Delete a session and all its messages."""
    await _verify_session_ownership(session_id, user_id)
    await sm.delete_session(session_id)
    return {"ok": True}
