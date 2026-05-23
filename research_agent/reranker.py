"""
Reranker — cross-encoder reranking for Pinecone retrieval results.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to reorder candidate documents
by query–passage relevance, improving precision over embedding-only search.
"""

import logging
from pinecone import Pinecone
from research_agent.config import PINECONE_API_KEY

logger = logging.getLogger("research_agent.reranker")

# ---------------------------------------------------------------------------
# Pinecone Client for Inference API
# ---------------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)


def rerank(query: str, matches: list, top_k: int) -> list:
    """Rerank Pinecone matches using Pinecone's Serverless Inference API.

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

    # Build passages for the inference API
    passages = []
    for m in matches:
        meta = m.metadata
        text = (
            f"{meta.get('title', '')}. "
            f"{meta.get('summary', '')}"
        )
        passages.append(text)

    try:
        # Send to Pinecone Inference API
        result = pc.inference.rerank(
            model="pinecone-rerank-v0",
            query=query,
            documents=passages,
            top_n=top_k,
            return_documents=False
        )

        scored = []
        # Pinecone returns results in result.data
        for r in result.data:
            # Safely handle both object attribute and dictionary access
            idx = r.index if hasattr(r, 'index') else r['index']
            score = r.score if hasattr(r, 'score') else r['score']
            original_match = matches[idx]
            scored.append((original_match, score))

        return scored
    except Exception as exc:
        logger.error("Pinecone Inference rerank failed: %s", exc)
        # Fallback to original score ordering if API fails
        return [(m, m.score) for m in matches[:top_k]]
