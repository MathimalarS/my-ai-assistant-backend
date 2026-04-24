# DocMind — Document Q&A Assistant

> RAG-powered document question answering with citations, built with FastAPI.

```
Documents → Embeddings → Vector DB → LLM → Answer with citations
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI App                            │
│                                                              │
│  POST /upload                      POST /ask                 │
│   │                                 │                        │
│   ▼                                 ▼                        │
│  [Ingestion Pipeline]            [Retrieval Pipeline]        │
│   │                                 │                        │
│   ├─ Parse  (PDF / TXT / MD)        ├─ embed_query()         │
│   ├─ Chunk  (512 words, 64 overlap) ├─ cosine_search()       │
│   ├─ embed_texts()                  ├─ top-k chunks          │
│   └─ VectorStore.add_document()     └─ generate_answer()     │
│                                          │                   │
│                                 ┌────────▼──────────┐        │
│                                 │  LLM (RAG prompt) │        │
│                                 │  Anthropic Claude │        │
│                                 │  OpenAI / Ollama  │        │
│                                 └───────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### Key modules

| File | Responsibility |
|------|---------------|
| `app/main.py` | FastAPI routes & middleware |
| `app/models.py` | Shared Pydantic models (no circular imports) |
| `app/ingestion.py` | Parse → chunk → embed → store |
| `app/embeddings.py` | sentence-transformers wrapper (all-MiniLM-L6-v2) |
| `app/store.py` | In-memory vector store (cosine similarity) |
| `app/retrieval.py` | Query embedding + similarity search |
| `app/llm.py` | RAG prompt engineering + multi-provider LLM calls |
| `frontend/index.html` | Single-file polished chat UI |

---

## Quick Start

### 1. Clone & configure

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

### 2. Run locally

```bash
chmod +x start.sh
./start.sh
# → http://localhost:8000
```

Or manually:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. Run with Docker

```bash
docker compose up --build
```

---

## API Reference

### `POST /upload`
Upload a document. Returns a `document_id` for subsequent queries.

```bash
curl -F "file=@paper.pdf" http://localhost:8000/upload
```

```json
{
  "document_id": "3f8a1b2c-...",
  "filename": "paper.pdf",
  "chunk_count": 142,
  "status": "ready"
}
```

### `POST /ask`
Ask a question about an uploaded document.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main finding?", "document_id": "3f8a1b2c-..."}'
```

```json
{
  "answer": "The main finding is that... [CHUNK-1]",
  "citations": [
    {
      "chunk_id": "3f8a1b2c-...:4",
      "page": 2,
      "text": "Our results demonstrate that...",
      "score": 0.87
    }
  ],
  "document_id": "3f8a1b2c-...",
  "question": "What is the main finding?"
}
```

### `GET /documents`
List all uploaded documents.

### `DELETE /documents/{document_id}`
Remove a document and its embeddings.

### `GET /health`
Health check with LLM provider info.

### `GET /docs`
Auto-generated Swagger UI.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `ollama` |
| `ANTHROPIC_API_KEY` | — | Required for Claude |
| `OPENAI_API_KEY` | — | Required for GPT-4o |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `llama3.2` | Local model name |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

---

## Supported File Types

| Format | Notes |
|--------|-------|
| `.pdf` | Text-based PDFs. Scanned PDFs need OCR pre-processing. |
| `.txt` | Any plain text. |
| `.md`  | Markdown (stripped before embedding). |

---

## What This Demonstrates

- **RAG pipeline** — end-to-end retrieval-augmented generation
- **Vector similarity search** — cosine similarity over dense embeddings
- **Prompt engineering** — structured JSON output with citation tracking
- **API design** — clean REST API with FastAPI + Pydantic v2
- **Multi-provider LLM** — Anthropic / OpenAI / Ollama swap via env var
- **Document parsing** — PDF and plain text chunking with overlap
- **Clean architecture** — no circular imports, separated models, modular design
