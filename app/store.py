"""
In-memory vector store using cosine similarity.
In production, replace with Pinecone, Weaviate, Qdrant, or pgvector.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    embedding: np.ndarray
    page: Optional[int] = None
    chunk_index: int = 0


@dataclass
class DocumentMeta:
    document_id: str
    filename: str
    chunk_count: int
    status: str = "ready"


class VectorStore:
    """
    Simple in-memory vector store.
    Stores chunk embeddings and performs cosine similarity search.
    Production alternatives: Pinecone, Qdrant, Weaviate, Chroma, pgvector.
    """

    def __init__(self):
        self._chunks: dict[str, list[Chunk]] = {}   # document_id → [Chunk]
        self._docs: dict[str, DocumentMeta] = {}    # document_id → DocumentMeta

    # ── Write ────────────────────────────────────────────────────────────────

    def add_document(self, document_id: str, filename: str, chunks: list[Chunk]):
        self._chunks[document_id] = chunks
        self._docs[document_id] = DocumentMeta(
            document_id=document_id,
            filename=filename,
            chunk_count=len(chunks),
            status="ready",
        )

    def delete_document(self, document_id: str):
        self._chunks.pop(document_id, None)
        self._docs.pop(document_id, None)

    # ── Read ─────────────────────────────────────────────────────────────────

    def has_document(self, document_id: str) -> bool:
        return document_id in self._docs

    def list_documents(self) -> list[DocumentMeta]:
        return list(self._docs.values())

    def similarity_search(
        self, query_embedding: np.ndarray, document_id: str, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        Cosine similarity search over chunks of a specific document.
        Returns list of (chunk, score) sorted by score descending.
        """
        chunks = self._chunks.get(document_id, [])
        if not chunks:
            return []

        # Stack all embeddings into a matrix for efficient dot product
        matrix = np.stack([c.embedding for c in chunks])  # (N, D)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        scores = matrix_norms @ query_norm  # cosine similarity

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(chunks[i], float(scores[i])) for i in top_indices]


# Singleton — shared across the app
vector_store = VectorStore()
