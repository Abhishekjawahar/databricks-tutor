"""
rag/ingest.py
-------------
Run this ONCE to build the knowledge base.

Usage:
    python -m rag.ingest

What it does:
    1. Scrapes a curated list of Databricks documentation pages
    2. Cleans and chunks the text into ~500 token passages
    3. Embeds each chunk locally using HuggingFace sentence-transformers
    4. Persists everything to a local ChromaDB collection on disk
"""

import os
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import track
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

console = Console()

# ---------------------------------------------------------------------------
# Databricks docs pages to ingest
# Add or remove URLs freely — any public docs page works.
# ---------------------------------------------------------------------------
DOCS_URLS = [
    # Core concepts
    "https://docs.databricks.com/en/introduction/index.html",
    "https://docs.databricks.com/en/lakehouse/index.html",
    "https://docs.databricks.com/en/delta/index.html",
    "https://docs.databricks.com/en/delta/tutorial.html",

    # Unity Catalog
    "https://docs.databricks.com/en/data-governance/unity-catalog/index.html",

    # Compute & Clusters
    "https://docs.databricks.com/en/compute/index.html",
    "https://docs.databricks.com/en/clusters/cluster-config-best-practices.html",

    # Notebooks & Workflows
    "https://docs.databricks.com/en/notebooks/index.html",
    "https://docs.databricks.com/en/workflows/index.html",

    # Spark & SQL
    "https://docs.databricks.com/en/spark/index.html",
    "https://docs.databricks.com/en/sql/index.html",

    # MLflow & ML
    "https://docs.databricks.com/en/mlflow/index.html",
    "https://docs.databricks.com/en/machine-learning/index.html",

    # Data ingestion
    "https://docs.databricks.com/en/ingestion/index.html",
    "https://docs.databricks.com/en/structured-streaming/index.html",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500        # characters per chunk (roughly ~125 tokens)
CHUNK_OVERLAP = 80      # overlap to preserve context across chunk boundaries
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "databricks_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, free, runs locally


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scrape_page(url: str) -> str:
    """Fetch a docs page and return clean plain text."""
    headers = {"User-Agent": "DatabricksTutor/1.0 (educational project)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        console.print(f"  [yellow]⚠ Skipped {url}: {e}[/yellow]")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav, footer, and sidebar noise
    for tag in soup.find_all(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()

    # Target the main article content if available, else full body
    main = soup.find("article") or soup.find("main") or soup.body
    if not main:
        return ""

    text = main.get_text(separator="\n")

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]          # drop blank lines
    return "\n".join(lines)


def chunk_text(text: str, url: str) -> list[dict]:
    """
    Split text into overlapping chunks.
    Each chunk carries metadata so we can show the source to the user.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunk_id = hashlib.md5(f"{url}-{start}".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": url,
                    "start_char": start,
                },
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def ingest():
    console.rule("[bold purple]Databricks Tutor — Knowledge Base Ingestion[/bold purple]")
    console.print(f"  Embedding model : [cyan]{EMBED_MODEL}[/cyan]")
    console.print(f"  Vector store    : [cyan]{CHROMA_PATH}[/cyan]")
    console.print(f"  Pages to scrape : [cyan]{len(DOCS_URLS)}[/cyan]\n")

    # 1. Set up ChromaDB with local sentence-transformer embeddings
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete and recreate the collection for a clean rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
        console.print("[yellow]  Existing collection deleted — rebuilding from scratch.[/yellow]")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # 2. Scrape, chunk, and upsert
    all_ids, all_texts, all_metas = [], [], []
    total_chunks = 0

    for url in track(DOCS_URLS, description="Scraping docs..."):
        text = scrape_page(url)
        if not text:
            continue

        chunks = chunk_text(text, url)
        for c in chunks:
            all_ids.append(c["id"])
            all_texts.append(c["text"])
            all_metas.append(c["metadata"])
        total_chunks += len(chunks)

        # Be polite to the docs server
        time.sleep(0.5)

    # 3. Batch upsert into ChromaDB (max 500 per call)
    console.print(f"\n  Embedding [bold]{total_chunks}[/bold] chunks — this takes ~1 min on first run...")
    batch_size = 500
    for i in range(0, len(all_ids), batch_size):
        collection.upsert(
            ids=all_ids[i : i + batch_size],
            documents=all_texts[i : i + batch_size],
            metadatas=all_metas[i : i + batch_size],
        )

    console.print(f"\n[bold green]✓ Ingestion complete![/bold green]")
    console.print(f"  {total_chunks} chunks stored in [cyan]{CHROMA_PATH}/{COLLECTION_NAME}[/cyan]")
    console.print("  Run [bold]python main.py[/bold] to start the tutor.\n")


if __name__ == "__main__":
    ingest()
