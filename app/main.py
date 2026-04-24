"""
DocMind — RAG Document Q&A Backend
Supports: Anthropic Claude, OpenAI, Ollama (set LLM_PROVIDER env var)
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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY      = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "20"))

_here = Path(__file__).parent
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", _here.parent / "frontend"))

# ── Embedding model ───────────────────────────────────────────────────────
logger.info(f"Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

# ── ChromaDB ──────────────────────────────────────────────────────────────
chroma = chromadb.PersistentClient(path=str(_here / "chroma_db"))

# ── Registry ─────────────────────────────────────────────────────────────
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
            pages.append(f"<<PAGE:{i+1}>>\n{t}")
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str) -> list[dict]:
    import re
    parts = re.split(r"<<PAGE:(\d+)>>", text)
    chunks = []

    if len(parts) == 1:
        words = text.split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i: i + CHUNK_SIZE])
            if chunk.strip():
                chunks.append({"text": chunk, "page": None})
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        words = page_text.split()
        j = 0
        while j < len(words):
            chunk = " ".join(words[j: j + CHUNK_SIZE])
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


# ── LLM Providers ─────────────────────────────────────────────────────────

async def call_anthropic(system: str, user: str) -> str:
    if not ANTHROPIC_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set in environment variables.")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1024,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        if r.status_code != 200:
            raise HTTPException(502, f"Anthropic error {r.status_code}: {r.text[:300]}")
        data = r.json()
        return data["content"][0]["text"]


async def call_openai(system: str, user: str) -> str:
    if not OPENAI_KEY:
        raise HTTPException(500, "OPENAI_API_KEY not set in environment variables.")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "max_tokens": 1024,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        if r.status_code != 200:
            raise HTTPException(502, f"OpenAI error {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]


async def call_ollama(system: str, user: str) -> str:
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        if r.status_code != 200:
            raise HTTPException(502, f"Ollama error: {r.text[:300]}")
        return r.json()["message"]["content"]


async def call_llm(system: str, user: str) -> str:
    if LLM_PROVIDER == "anthropic":
        return await call_anthropic(system, user)
    elif LLM_PROVIDER == "openai":
        return await call_openai(system, user)
    elif LLM_PROVIDER == "ollama":
        return await call_ollama(system, user)
    else:
        raise HTTPException(500, f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ── Frontend ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h2>Backend is running. Frontend not found.</h2>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "model": ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL,
        "documents_loaded": len(registry),
    }


# ── Upload ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
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


# ── Documents ─────────────────────────────────────────────────────────────

@app.get("/documents")
async def list_documents():
    return list(registry.values())


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
    question: str
    document_id: str
    top_k: Optional[int] = 3


@app.post("/ask")
async def ask(req: AskRequest):
    if req.document_id not in registry:
        raise HTTPException(404, "Document not found. Upload it first.")

    col = get_collection(req.document_id)
    total_chunks = col.count()
    if total_chunks == 0:
        raise HTTPException(400, "No chunks indexed for this document.")

    n = min(req.top_k or 3, total_chunks)

    q_emb = embed([req.question])[0]
    results = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks    = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

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

    answer = await call_llm(system, user_msg)
    return {"answer": answer, "citations": citations}


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
