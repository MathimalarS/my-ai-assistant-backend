"""
DocMind — RAG Document Q&A Backend
Serves the frontend and exposes endpoints matching index.html exactly:
  GET  /             → serves index.html
  GET  /health       → { llm_provider, ollama_online, models }
  POST /upload       → { document_id, filename, chunk_count }
  GET  /documents    → [ { document_id, filename, chunk_count }, ... ]
  DELETE /documents/{id}
  POST /ask          → { answer, citations: [{text, score, page}] }

Requirements:
  pip install fastapi uvicorn[standard] chromadb sentence-transformers pypdf httpx python-multipart
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import pypdf
import httpx
import uuid
import os
import io
import json
import logging
from typing import Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docmind")

app = FastAPI(title="DocMind")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://my-ai-assistant-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",  "200"))  # ~200 words keeps prompts lean
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP","20"))

# index.html lives in ../frontend/index.html relative to this file
# or set FRONTEND_DIR env var to point elsewhere
_here = Path(__file__).parent
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", _here.parent / "frontend"))

# ── Embedding model ───────────────────────────────────────────────────────
logger.info(f"Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

# ── ChromaDB ──────────────────────────────────────────────────────────────
chroma = chromadb.PersistentClient(path=str(_here / "chroma_db"))

# ── Persistent document registry (survives restarts) ─────────────────────
REGISTRY_FILE = _here / "registry.json"

def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except Exception as e:
            logger.warning(f"Could not load registry: {e}")
    return {}

def _save_registry():
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))

registry: dict = _load_registry()
logger.info(f"Registry loaded: {len(registry)} document(s)")


# ── Helpers ───────────────────────────────────────────────────────────────

def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            # prefix each page's text so we can recover page numbers later
            pages.append(f"<<PAGE:{i+1}>>\n{t}")
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str) -> list[dict]:
    """
    Returns list of { text, page } dicts.
    Page is extracted from <<PAGE:N>> markers inserted during PDF extraction.
    """
    import re
    # Split into page sections
    parts = re.split(r"<<PAGE:(\d+)>>", text)

    chunks = []
    current_page = None

    if len(parts) == 1:
        # Plain text — no page markers
        words = text.split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if chunk.strip():
                chunks.append({"text": chunk, "page": None})
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # PDF with page markers: parts = ["", "1", "page1 text", "2", "page2 text", ...]
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        words = page_text.split()
        j = 0
        while j < len(words):
            chunk = " ".join(words[j : j + CHUNK_SIZE])
            if chunk.strip():
                chunks.append({"text": chunk, "page": page_num})
            j += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts, show_progress_bar=False).tolist()


def get_collection(doc_id: str):
    return chroma.get_or_create_collection(
        f"doc_{doc_id}", metadata={"hnsw:space": "cosine"}
    )


# ── Serve frontend ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve index.html — put it in ../frontend/index.html or set FRONTEND_DIR."""
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(
            404,
            detail=(
                f"index.html not found at {html_path}. "
                "Place your index.html in the frontend/ folder next to backend/, "
                "or set the FRONTEND_DIR environment variable."
            )
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# Serve any static assets the frontend might request (css, js, images)
@app.get("/favicon.ico")
async def favicon():
    # Return empty 204 so browser stops retrying
    from fastapi.responses import Response
    return Response(status_code=204)


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    index.html reads: data.llm_provider  → shown in header pill
    """
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {
            "status":       "ok",
            "ollama_online": True,
            "llm_provider":  OLLAMA_MODEL,
            "models":        models,
        }
    except Exception as e:
        return {
            "status":        "degraded",
            "ollama_online": False,
            "llm_provider":  OLLAMA_MODEL,
            "error":         str(e),
        }


# ── Upload ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    index.html expects: { document_id, filename, chunk_count }
    """
    raw = await file.read()
    filename = file.filename or "document"

    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20 MB).")

    text = extract_text(filename, raw)
    if not text.strip():
        raise HTTPException(400, "Could not extract any text from this file.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "Document appears empty after processing.")

    doc_id = str(uuid.uuid4())[:12]
    texts_only = [c["text"] for c in chunks]
    embeddings = embed(texts_only)

    col = get_collection(doc_id)
    col.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts_only,
        metadatas=[
            {"chunk_index": i, "source": filename, "page": str(c["page"] or "")}
            for i, c in enumerate(chunks)
        ],
    )

    entry = {"document_id": doc_id, "filename": filename, "chunk_count": len(chunks)}
    registry[doc_id] = entry
    _save_registry()

    logger.info(f"Indexed: {filename} → {len(chunks)} chunks (id={doc_id})")
    return entry


# ── Documents list ────────────────────────────────────────────────────────

@app.get("/documents")
async def list_documents():
    return list(registry.values())


# ── Delete ────────────────────────────────────────────────────────────────

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in registry:
        raise HTTPException(404, "Document not found.")
    try:
        chroma.delete_collection(f"doc_{doc_id}")
    except Exception:
        pass
    del registry[doc_id]
    _save_registry()
    return {"deleted": doc_id}


# ── Ask ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:    str
    document_id: str
    top_k:       Optional[int] = 3


@app.post("/ask")
async def ask(req: AskRequest):
    """
    RAG pipeline → answer + citations.

    index.html citation shape: { text, score, page }
    It also calls renderAnswerWithCitations() which looks for [CHUNK-N] tags,
    so we instruct the LLM to use that exact notation.
    """
    if req.document_id not in registry:
        raise HTTPException(404, "Document not found. Upload it first.")

    col = get_collection(req.document_id)
    total_chunks = col.count()
    if total_chunks == 0:
        raise HTTPException(400, "No chunks indexed for this document.")

    # Clamp top_k to however many chunks actually exist
    n = min(req.top_k or 3, total_chunks)
    logger.info(f"Query: doc={req.document_id} chunks={total_chunks} top_k={n} q={req.question!r}")

    # 1. Retrieve
    q_emb = embed([req.question])[0]
    results = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks    = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # 2. Build context + citation list
    context_parts = []
    citations = []
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metas, distances)):
        context_parts.append(f"[CHUNK-{i+1}]\n{chunk}")
        page_raw = meta.get("page", "")
        citations.append({
            "text":  chunk,
            "score": round(max(0.0, 1.0 - float(dist)), 4),
            "page":  int(page_raw) if str(page_raw).isdigit() else None,
        })

    context = "\n\n---\n\n".join(context_parts)

    system = "Answer using ONLY the context below. Cite chunks as [CHUNK-N]. Be concise."
    user_msg = (
        f"Context:\n\n{context}\n\n"
        f"Question: {req.question}\n\n"
        "Answer (use [CHUNK-N] citations):"
    )

    # 3. Call Ollama
    ollama_url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model":  OLLAMA_MODEL,
        "stream": False,
        "options": {
                    "temperature": 0.2,
                    "num_ctx": 2048,       # limit context window
                    "num_predict": 512,   # limit output tokens
                },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            logger.info(f"Calling Ollama model={OLLAMA_MODEL} url={ollama_url}")
            r = await client.post(ollama_url, json=payload)

            # Surface the raw Ollama error body before raise_for_status swallows it
            if r.status_code != 200:
                body_preview = r.text[:500]
                logger.error(f"Ollama {r.status_code}: {body_preview}")
                raise HTTPException(
                    502,
                    f"Ollama returned HTTP {r.status_code}. "
                    f"Make sure the model is pulled: ollama pull {OLLAMA_MODEL}. "
                    f"Details: {body_preview}"
                )

            resp_json = r.json()
            logger.info(f"Ollama response keys: {list(resp_json.keys())}")

            # Handle both /api/chat and /api/generate response shapes
            if "message" in resp_json:
                answer = resp_json["message"]["content"]
            elif "response" in resp_json:
                answer = resp_json["response"]
            else:
                raise HTTPException(
                    502,
                    f"Unexpected Ollama response shape: {str(resp_json)[:300]}"
                )

    except HTTPException:
        raise   # re-raise our own HTTPExceptions unchanged
    except httpx.ConnectError:
        raise HTTPException(
            503,
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is it running? Try: ollama serve"
        )
    except httpx.TimeoutException:
        raise HTTPException(
            504,
            f"Ollama timed out after 180s. "
            "Try a smaller/faster model: ollama pull llama3.2:1b"
        )
    except Exception as e:
        logger.exception("Unexpected error calling Ollama")
        raise HTTPException(500, f"LLM error: {type(e).__name__}: {e}")

    return {"answer": answer, "citations": citations}


# ── Debug endpoints (visit in browser to diagnose) ────────────────────────

@app.get("/debug/ollama")
async def debug_ollama():
    """Check Ollama connectivity and list available models."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            tags = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in tags.json().get("models", [])]
    except Exception as e:
        return {"ollama_reachable": False, "error": str(e), "url": OLLAMA_BASE_URL}

    model_ok = any(OLLAMA_MODEL in m for m in models)
    return {
        "ollama_reachable": True,
        "url": OLLAMA_BASE_URL,
        "configured_model": OLLAMA_MODEL,
        "model_available": model_ok,
        "all_models": models,
        "fix": None if model_ok else f"Run: ollama pull {OLLAMA_MODEL}",
    }


@app.get("/debug/ping-llm")
async def debug_ping_llm():
    """Send a tiny prompt to Ollama and return the raw response."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "stream": False,
                    "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
                },
            )
            return {
                "status_code": r.status_code,
                "response_keys": list(r.json().keys()) if r.status_code == 200 else None,
                "raw": r.json(),
            }
    except Exception as e:
        return {"error": type(e).__name__, "detail": str(e)}


@app.get("/debug/docs")
async def debug_docs():
    """Show all documents currently in the registry + their chunk counts."""
    result = []
    for doc_id, meta in registry.items():
        try:
            col = get_collection(doc_id)
            actual_chunks = col.count()
        except Exception as e:
            actual_chunks = f"ERROR: {e}"
        result.append({**meta, "chroma_count": actual_chunks})
    return result


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
