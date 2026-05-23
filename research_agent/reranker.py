"""
Reranker — cross-encoder reranking for Pinecone retrieval results.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to reorder candidate documents
by query–passage relevance, improving precision over embedding-only search.
"""

import logging
import threading

from sentence_transformers import CrossEncoder

logger = logging.getLogger("research_agent.reranker")

# ---------------------------------------------------------------------------
# Lazy-loaded singleton — model loads once on first call (thread-safe)
# ---------------------------------------------------------------------------
_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    """Return the cross-encoder, loading it on first use (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            # Double-check after acquiring lock
            if _model is None:
                logger.info("Loading cross-encoder reranker model …")
                _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Reranker model loaded.")
    return _model


def rerank(query: str, matches: list, top_k: int) -> list:
    """Rerank Pinecone matches using the cross-encoder.

    Args:
        query:   The user's search query.
        matches: List of Pinecone ScoredVector objects (must have .metadata).
        top_k:   How many results to keep after reranking.

    Returns:
        A list of (match, ce_score) tuples sorted by cross-encoder score,
        trimmed to *top_k*.
    """
    if not matches:
        return []

    model = _get_model()

    # Build (query, passage) pairs for the cross-encoder
    passages = []
    for m in matches:
        meta = m.metadata
        text = (
            f"{meta.get('title', '')}. "
            f"{meta.get('summary', '')}"
        )
        passages.append(text)

    scores = model.predict([(query, p) for p in passages])

    # Pair each match with its cross-encoder score and sort descending
    scored = sorted(
        zip(matches, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return scored[:top_k]
