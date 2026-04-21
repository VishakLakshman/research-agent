"""
Phase 1 – Ingestion Pipeline
Parses PDF and ePub files, chunks text, embeds via Mistral, stores in ChromaDB.

Install deps:
    pip install mistralai chromadb pdfplumber ebooklib beautifulsoup4 python-dotenv langchain-text-splitters
"""

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

import pdfplumber
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from mistralai.client import Mistral
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
EMBED_MODEL = "mistral-embed"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_kb"
CHUNK_SIZE = 512        # tokens (approx characters / 4)
CHUNK_OVERLAP = 64


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_pdf(path: str) -> str:
    """Extract full text from a PDF file."""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)


def parse_epub(path: str) -> str:
    """Extract full text from an ePub file."""
    book = epub.read_epub(path)
    text_parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            text_parts.append(text)
    return "\n\n".join(text_parts)


def load_document(path: str) -> str:
    """Dispatch to the right parser based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    elif ext == ".epub":
        return parse_epub(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,      # approx char count
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[str], client: Mistral) -> list[list[float]]:
    """Embed a list of text chunks using Mistral embed model."""
    # Mistral supports batching up to 128 inputs
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


# ── Storage ───────────────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def stable_id(source: str, index: int) -> str:
    """Deterministic chunk ID so re-ingesting is idempotent."""
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Main entry ────────────────────────────────────────────────────────────────

def ingest_file(file_path: str):
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    collection = get_collection()

    print(f"[ingest] Loading: {file_path}")
    raw_text = load_document(file_path)
    print(f"[ingest] Extracted {len(raw_text):,} chars")

    chunks = chunk_text(raw_text)
    print(f"[ingest] Split into {len(chunks)} chunks")

    print("[ingest] Embedding chunks via Mistral...")
    embeddings = embed_chunks(chunks, mistral_client)

    source_name = Path(file_path).name
    ids = [stable_id(source_name, i) for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    print(f"[ingest] Stored {len(chunks)} chunks from '{source_name}' into ChromaDB")


def ingest_directory(directory: str):
    """Ingest all PDF and ePub files in a directory."""
    files = list(Path(directory).glob("**/*.pdf")) + list(Path(directory).glob("**/*.epub"))
    if not files:
        print(f"[ingest] No PDF or ePub files found in {directory}")
        return
    for f in files:
        ingest_file(str(f))
    print(f"[ingest] Done. {len(files)} file(s) ingested.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python phase1_ingest.py <file_or_directory>")
        sys.exit(1)
    target = sys.argv[1]
    if Path(target).is_dir():
        ingest_directory(target)
    else:
        ingest_file(target)