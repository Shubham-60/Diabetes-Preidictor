"""
CrossEncoder-based reranking for FAISS retrieval results.
"""

from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, docs: List[str], top_k: int = 2) -> List[str]:
    """Rerank candidate docs and return the top_k most relevant passages."""
    if not docs:
        return []

    pairs = [[query, doc] for doc in docs]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        return docs[:top_k]

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
