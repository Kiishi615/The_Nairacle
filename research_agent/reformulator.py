"""
Query Reformulator — makes the agent conversational.

Rewrites vague follow-up messages into standalone questions before
they hit the agent. Resolves pronouns ("What about their dividends?"
→ "What are Zenith Bank's dividend payments?") using conversation context.

Pattern borrowed from doc_assistant.py's reformulation chain.
"""

import logging

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from research_agent.config import CHAT_MODEL

logger = logging.getLogger("research_agent.reformulator")

# ---------------------------------------------------------------------------
# Model — same as the agent
# ---------------------------------------------------------------------------

model = init_chat_model(CHAT_MODEL)

# ---------------------------------------------------------------------------
# Reformulation Chain
# ---------------------------------------------------------------------------

reformulation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest message as a standalone question that can "
     "be understood WITHOUT any conversation history. Resolve all pronouns "
     "and references using the conversation context provided. "
     "Do NOT answer the question — only rewrite it. "
     "If it is already a standalone question, return it as-is."),
    ("human",
     "Conversation summary:\n{summary}\n\n"
     "Recent messages:\n{recent_messages}\n\n"
     "User's latest message: {question}\n\n"
     "Standalone question:"),
])

reformulation_chain = reformulation_prompt | model | StrOutputParser()


def reformulate(question: str, recent_messages: list[dict],
                summary: str = "") -> str:
    """Rewrite *question* as a standalone query using conversation context.

    Args:
        question:        The user's raw message.
        recent_messages: List of recent messages [{"role": ..., "content": ...}].
        summary:         Rolling conversation summary (may be empty).

    Returns:
        A standalone question string suitable for vector search / agent input.
        Falls back to the original question on any error.
    """
    # If no context exists, the question is already standalone
    if not recent_messages and not summary:
        return question

    try:
        # Format recent messages for the prompt
        formatted = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
            for m in recent_messages[-10:]   # last 10 for reformulation context
        )

        result = reformulation_chain.invoke({
            "question": question,
            "recent_messages": formatted,
            "summary": summary or "(none)",
        })

        return result.strip()

    except Exception as exc:
        logger.warning("Reformulation failed: %s — using raw question", exc)
        return question
