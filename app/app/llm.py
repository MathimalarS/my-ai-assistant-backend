"""
LLM layer: build a RAG prompt, call an LLM, parse citations.

Supports:
  - Anthropic Claude (default, via ANTHROPIC_API_KEY)
  - OpenAI GPT-4o   (via OPENAI_API_KEY)
  - Ollama local    (via OLLAMA_HOST, no key needed)

Set LLM_PROVIDER env var to: "anthropic" | "openai" | "ollama"
"""

import os
import json
import re

from app.models import Citation


# ─── Prompt engineering ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Your job is to answer questions ONLY using the provided document excerpts.

Rules:
1. Answer based SOLELY on the excerpts below — do not use outside knowledge.
2. Every factual claim MUST be backed by a [CHUNK-N] citation.
3. If the answer is not in the excerpts, say: "I could not find this information in the document."
4. Be concise and clear. Use bullet points for lists.
5. Format citations as [CHUNK-1], [CHUNK-2], etc. inline in your answer.

Respond in this JSON format (no markdown fences):
{
  "answer": "Your answer with [CHUNK-N] citations inline.",
  "used_chunks": [1, 2]
}"""


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        page_info = f", page {c['page']}" if c['page'] else ""
        parts.append(f"[CHUNK-{i+1}] (score: {c['score']:.2f}{page_info})\n{c['text']}")
    excerpt_block = "\n\n".join(parts)
    return f"DOCUMENT EXCERPTS:\n{excerpt_block}\n\nQUESTION: {question}"


# ─── LLM Providers ───────────────────────────────────────────────────────────

async def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


async def _call_openai(system: str, user: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


async def _call_ollama(system: str, user: str) -> str:
    import httpx
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{host}/api/chat", json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        })
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ─── Main entry point ─────────────────────────────────────────────────────────

async def generate_answer(question: str, chunks: list[dict]) -> tuple[str, list[Citation]]:
    """
    Call the configured LLM and parse the structured response.
    Returns (answer_text, [Citation, ...]).
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    system = SYSTEM_PROMPT
    user = build_user_prompt(question, chunks)

    callers = {
        "anthropic": _call_anthropic,
        "openai": _call_openai,
        "ollama": _call_ollama,
    }
    if provider not in callers:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use: anthropic | openai | ollama")

    raw = await callers[provider](system, user)

    # Parse JSON response
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return raw.strip(), []

    answer = data.get("answer", "No answer generated.")
    used_indices = data.get("used_chunks", [])  # 1-based

    citations = []
    for idx in used_indices:
        i = idx - 1  # convert to 0-based
        if 0 <= i < len(chunks):
            c = chunks[i]
            citations.append(Citation(
                chunk_id=c["chunk_id"],
                page=c["page"],
                text=c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
                score=c["score"],
            ))

    return answer, citations
