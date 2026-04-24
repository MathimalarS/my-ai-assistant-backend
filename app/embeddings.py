"""
Embedding generation using sentence-transformers.
Model: all-MiniLM-L6-v2 (fast, 384-dim, MIT license).

For production, swap to:
  - OpenAI text-embedding-3-small  (1536-dim, API call)
  - Cohere embed-v3               (1024-dim, API call)
  - BGE-large-en-v1.5             (1024-dim, local)
"""

import numpy as np
from functools import lru_cache
from typing import Union

# Lazy-load the model to avoid slowing down server startup
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings.
    Returns np.ndarray of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings  # already L2-normalised


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns np.ndarray of shape (embedding_dim,).
    """
    return embed_texts([query])[0]
