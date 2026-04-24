"""
Document ingestion pipeline:
  1. Parse (PDF or plain text)
  2. Chunk with overlap
  3. Embed chunks
  4. Store in vector store
"""

import uuid
import re
from typing import Optional

from app.store import vector_store, Chunk, DocumentMeta
from app.embeddings import embed_texts


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """
    Split text into overlapping chunks by word count.
    Returns list of dicts: {text, chunk_index, page (None for plain text)}.
    
    Why overlap? Answers often span chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        # Skip near-empty chunks
        if len(chunk_text.strip()) > 20:
            chunks.append({
                "text": chunk_text,
                "chunk_index": idx,
                "page": None,
            })
            idx += 1

        start += chunk_size - overlap  # slide with overlap

    return chunks


def chunk_pdf_by_page(pages: list[str], chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """
    Chunk a PDF page-by-page, preserving page numbers.
    Each page is further split if longer than chunk_size words.
    """
    all_chunks = []
    idx = 0
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for c in page_chunks:
            c["page"] = page_num
            c["chunk_index"] = idx
            idx += 1
        all_chunks.extend(page_chunks)
    return all_chunks


# ─── Parsers ─────────────────────────────────────────────────────────────────

def parse_txt(content: bytes) -> list[dict]:
    text = content.decode("utf-8", errors="replace")
    # Normalise whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return chunk_text(text)


def parse_pdf(content: bytes) -> list[dict]:
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return chunk_pdf_by_page(pages)
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")


def parse_md(content: bytes) -> list[dict]:
    # Treat markdown as plain text (strip markdown syntax for embedding quality)
    text = content.decode("utf-8", errors="replace")
    text = re.sub(r"#{1,6}\s*", "", text)          # headings
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold/italic
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)  # code
    return chunk_text(text)


# ─── Ingestion entry point ────────────────────────────────────────────────────

async def ingest_document(content: bytes, filename: str, ext: str) -> DocumentMeta:
    """
    Full pipeline: parse → chunk → embed → store.
    Returns DocumentMeta with the new document_id.
    """
    # 1. Parse
    parsers = {".pdf": parse_pdf, ".txt": parse_txt, ".md": parse_md}
    raw_chunks = parsers[ext](content)

    if not raw_chunks:
        raise ValueError("Document appears to be empty or could not be parsed.")

    # 2. Embed all chunk texts in one batch (efficient)
    texts = [c["text"] for c in raw_chunks]
    embeddings = embed_texts(texts)  # shape: (N, D)

    # 3. Build Chunk objects
    document_id = str(uuid.uuid4())
    chunks = [
        Chunk(
            chunk_id=f"{document_id}:{i}",
            document_id=document_id,
            text=raw_chunks[i]["text"],
            embedding=embeddings[i],
            page=raw_chunks[i].get("page"),
            chunk_index=raw_chunks[i]["chunk_index"],
        )
        for i in range(len(raw_chunks))
    ]

    # 4. Store
    vector_store.add_document(document_id, filename, chunks)

    return DocumentMeta(
        document_id=document_id,
        filename=filename,
        chunk_count=len(chunks),
        status="ready",
    )
