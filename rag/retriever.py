"""
rag/retriever.py
----------------
Shared retrieval module used by all three specialist agents.

Exposes one function:
    retrieve(query, n_results) -> list[RetrievedChunk]

Each RetrievedChunk has:
    .text     — the passage text
    .source   — the original docs URL
    .score    — cosine similarity (0–1, higher = more relevant)
"""

from dataclasses import dataclass
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "databricks_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Minimum similarity score — chunks below this are too weak to include
SCORE_THRESHOLD = 0.30


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


# ---------------------------------------------------------------------------
# Lazy singleton — load ChromaDB once, reuse across all agent calls
# ---------------------------------------------------------------------------
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, n_results: int = 5) -> list[RetrievedChunk]:
    """
    Embed the query locally and find the most relevant doc chunks.

    Args:
        query:     The user's question or topic string.
        n_results: How many chunks to return (before threshold filtering).

    Returns:
        List of RetrievedChunk, sorted best-first, filtered by SCORE_THRESHOLD.
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]   # ChromaDB cosine distance: 0=identical, 2=opposite

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert cosine distance → similarity score (0–1)
        score = 1 - (dist / 2)
        if score >= SCORE_THRESHOLD:
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                score=round(score, 3),
            ))

    # Sort best-first (already sorted by ChromaDB, but let's be explicit)
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a clean context block for the LLM prompt.
    Each chunk is numbered and source-attributed.
    """
    if not chunks:
        return "No relevant documentation found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk.source}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)
