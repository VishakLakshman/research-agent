"""
Phase 3 – FastAPI Backend
Exposes the LangGraph agent over HTTP.

Endpoints:
    POST /ask          → JSON response {"answer": ..., "sources": [...]}
    POST /ask/stream   → SSE stream of answer tokens

Install deps:
    pip install fastapi uvicorn sse-starlette python-dotenv
    (plus all deps from phases 1 & 2)

Run:
    uvicorn phase3_api:app --reload --port 8000
"""

import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from mistralai.client import Mistral
import chromadb

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
CHAT_MODEL = "mistral-small-latest"
EMBED_MODEL = "mistral-embed"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_kb"
TOP_K = 5

app = FastAPI(title="Research Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared clients
_mistral = Mistral(api_key=MISTRAL_API_KEY)
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

SYSTEM_PROMPT = """You are a research assistant with access to a curated knowledge base.
Answer the user's question using ONLY the context passages provided below.
If the context does not contain enough information, say so clearly — do not hallucinate.
Always cite which source document your answer draws from."""


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Shared retrieval logic ─────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int) -> tuple[list[str], list[str]]:
    """Embed query and retrieve top_k chunks. Returns (chunks, sources)."""
    embed_resp = _mistral.embeddings.create(model=EMBED_MODEL, inputs=[query])
    query_embedding = embed_resp.data[0].embedding

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    chunks = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    sources = list({m.get("source", "unknown") for m in metas})
    return chunks, sources


def build_messages(query: str, chunks: list[str], sources: list[str]) -> list[dict]:
    context_block = "\n\n".join(
        f"[Source: {src}]\n{chunk}"
        for src, chunk in zip(sources, chunks)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Blocking endpoint – returns complete answer as JSON."""
    try:
        chunks, sources = retrieve_context(req.query, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    if not chunks:
        return AskResponse(
            answer="I couldn't find relevant information in the knowledge base.",
            sources=[],
        )

    messages = build_messages(req.query, chunks, sources)

    try:
        response = _mistral.chat.complete(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return AskResponse(answer=answer, sources=sources)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """SSE streaming endpoint – streams answer tokens as they arrive."""
    try:
        chunks, sources = retrieve_context(req.query, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    messages = build_messages(req.query, chunks, sources)

    async def token_generator():
        # Yield sources as first event so the client can display them
        import json
        yield {"event": "sources", "data": json.dumps(sources)}

        # Stream tokens
        with _mistral.chat.stream(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        ) as stream:
            for chunk in stream:
                delta = chunk.data.choices[0].delta.content if chunk.data.choices else None
                if delta:
                    yield {"event": "token", "data": delta}

        yield {"event": "done", "data": ""}

    return EventSourceResponse(token_generator())


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phase3_api:app", host="0.0.0.0", port=8000, reload=True)