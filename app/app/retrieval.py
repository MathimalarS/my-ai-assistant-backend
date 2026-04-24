"""
Retrieval: embed the query, search the vector store, return ranked chunks.
"""

from app.embeddings import embed_query
from app.store import vector_store, Chunk


def retrieve_chunks(
    question: str,
    document_id: str,
    top_k: int = 5,
    min_score: float = 0.20,
) -> list[dict]:
    """
    1. Embed the question
    2. Cosine-search the vector store
    3. Filter by min_score threshold
    4. Return list of chunk dicts with score attached
    """
    query_embedding = embed_query(question)
    results = vector_store.similarity_search(query_embedding, document_id, top_k=top_k)

    output = []
    for chunk, score in results:
        if score < min_score:
            continue
        output.append({
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "score": round(score, 4),
        })

    return output
