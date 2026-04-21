"""
Phase 2 – LangGraph Research Agent
Retrieves relevant chunks from ChromaDB, then calls Mistral to synthesise an answer.

Install deps:
    pip install langgraph langchain-core mistralai chromadb python-dotenv
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pathlib import Path

from mistralai.client import Mistral
import chromadb

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
CHAT_MODEL = "mistral-small-latest"
EMBED_MODEL = "mistral-embed"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_kb"
TOP_K = 5  # number of chunks to retrieve


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    retrieved_chunks: list[str]
    sources: list[str]
    answer: str


# ── Shared clients (initialised once) ────────────────────────────────────────

_mistral = Mistral(api_key=MISTRAL_API_KEY)
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# ── Node: embed the query ─────────────────────────────────────────────────────

def embed_query_node(state: AgentState) -> AgentState:
    """Embed the user query and retrieve top-K chunks from ChromaDB."""
    query = state["query"]

    embed_resp = _mistral.embeddings.create(model=EMBED_MODEL, inputs=[query])
    query_embedding = embed_resp.data[0].embedding

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )

    chunks = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    sources = [m.get("source", "unknown") for m in metas]

    return {
        **state,
        "retrieved_chunks": chunks,
        "sources": sources,
    }


# ── Node: generate answer ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant with access to a curated knowledge base.
Answer the user's question using ONLY the context passages provided below.
If the context does not contain enough information, say so clearly — do not hallucinate.
Always cite which source document your answer draws from."""


def generate_answer_node(state: AgentState) -> AgentState:
    """Call Mistral with retrieved context to produce a grounded answer."""
    query = state["query"]
    chunks = state["retrieved_chunks"]
    sources = state["sources"]

    if not chunks:
        answer = "I couldn't find relevant information in the knowledge base for your query."
        return {**state, "answer": answer}

    context_block = "\n\n".join(
        f"[Source: {src}]\n{chunk}"
        for src, chunk in zip(sources, chunks)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {query}",
        },
    ]

    response = _mistral.chat.complete(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    return {
        **state,
        "answer": answer,
        "messages": state["messages"] + [AIMessage(content=answer)],
    }


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", embed_query_node)
    graph.add_node("generate", generate_answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


research_agent = build_graph()


# ── Convenience function ──────────────────────────────────────────────────────

def ask(query: str) -> dict:
    """
    Run the agent for a single query.
    Returns {"answer": str, "sources": list[str]}.
    """
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "retrieved_chunks": [],
        "sources": [],
        "answer": "",
    }
    result = research_agent.invoke(initial_state)
    return {
        "answer": result["answer"],
        "sources": list(set(result["sources"])),
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the main topic of the documents?"
    out = ask(query)
    print("\nAnswer:\n", out["answer"])
    print("\nSources:", out["sources"])