# 🧠 Databricks Multi-Agent Tutor

> A multi-agent RAG system that teaches Databricks — built entirely with free, open-source tools.

---

## What it does

Ask any Databricks question in plain English. An **orchestrator agent** classifies your intent and routes it to the right specialist:

| Agent | Handles | Example prompt |
|---|---|---|
| 🧠 Concept Tutor | Explanations, definitions, conceptual overviews | *"What is Delta Lake?"* |
| 💻 Code Guide | PySpark, SQL, notebooks, debugging | *"Write a PySpark job to read a Delta table filtered by date"* |
| 📝 Quiz Master | MCQs, flashcards, answer evaluation | *"Quiz me on Unity Catalog"* |

Every response is grounded in **real Databricks documentation** retrieved at query time via RAG — the agents can't hallucinate facts that contradict the docs.

---

## Architecture

```
User input
    │
    ▼
┌─────────────────────────────┐
│      Orchestrator Agent      │  classify_intent() — one LLM call, returns "concept" | "code" | "quiz"
└──────────────┬──────────────┘
               │  conditional edge (LangGraph)
       ┌───────┼───────┐
       ▼       ▼       ▼
  Concept   Code    Quiz
   Tutor    Guide   Master
       │       │       │
       └───────┴───────┘
               │
               ▼
        RAG Pipeline
   ┌────────────────────┐
   │  1. Retrieve        │  ChromaDB cosine search → top-5 chunks
   │  2. Filter          │  Score threshold ≥ 0.30
   │  3. Format context  │  Numbered, source-attributed passages
   │  4. Generate        │  Groq LLM (Llama 3.1 70B)
   └────────────────────┘
               │
               ▼
        Vector Store
   ChromaDB (local disk)
   Embedded with: all-MiniLM-L6-v2
   Source: Databricks public docs (15 pages)
```

### Key design decisions

**Why LangGraph?**
LangGraph models the agent workflow as an explicit state machine with typed state. This makes the routing logic inspectable, testable, and easy to extend — add a new agent by adding one node and one edge.

**Why Groq?**
Groq's free tier gives access to Llama 3.1 70B with extremely low latency (~1–2s per response). No credit card required.

**Why local embeddings?**
`sentence-transformers/all-MiniLM-L6-v2` runs on CPU in ~50ms per query. No API call, no cost, no rate limits — embeddings stay free at any scale.

**Why ChromaDB?**
Zero-config local vector store that persists to disk. No Docker, no server, no cloud account. The entire knowledge base is a folder.

**Chunk strategy**
500-character chunks with 80-character overlap. Overlap preserves context at boundaries so a sentence split across two chunks doesn't lose meaning in either.

---

## Free stack

| Component | Tool | Cost |
|---|---|---|
| Agent orchestration | LangGraph | Free / open source |
| LLM inference | Groq API (Llama 3.1 70B) | Free tier |
| Embeddings | HuggingFace sentence-transformers | Free / local |
| Vector store | ChromaDB | Free / local |
| Knowledge base | Databricks public docs | Free / public |

**Total running cost: $0**

---

## Project structure

```
databricks-tutor/
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py      # Intent classification + routing
│   ├── concept_tutor.py     # Conceptual explanation agent
│   ├── code_guide.py        # Code generation agent
│   └── quiz_master.py       # Quiz + flashcard agent
├── rag/
│   ├── __init__.py
│   ├── ingest.py            # Scrape docs → chunk → embed → ChromaDB
│   └── retriever.py         # Query ChromaDB → ranked passages
├── chroma_db/               # Auto-created by ingest.py (gitignored)
├── graph.py                 # LangGraph state machine
├── main.py                  # Interactive CLI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/databricks-tutor.git
cd databricks-tutor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free Groq API key

1. Sign up at [console.groq.com](https://console.groq.com) — no credit card needed
2. Create an API key
3. Copy `.env.example` to `.env` and paste your key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

### 3. Build the knowledge base (one-time)

```bash
python -m rag.ingest
```

This scrapes 15 Databricks documentation pages, embeds them locally, and saves everything to `./chroma_db/`. Takes ~1–2 minutes. Only needs to run once — or again if you add new URLs to `DOCS_URLS` in `rag/ingest.py`.

### 4. Run the tutor

```bash
python main.py
```

---

## Example session

```
You > What is Delta Lake?

  ↳ Orchestrator routed 'concept' → concept_tutor

┌─ 🧠 Concept Tutor ──────────────────────────────────────┐
│                                                           │
│  ## 🧠 Concept: Delta Lake                               │
│                                                           │
│  **Plain English summary**                                │
│  Delta Lake is an open-source storage layer that adds    │
│  reliability to data lakes...                            │
│                                                           │
│  **Analogy**                                             │
│  Think of Delta Lake as Git for your data...             │
│                                                           │
└───────────────────────────────────────────────────────────┘

Sources used:
  https://docs.databricks.com/en/delta/index.html
  https://docs.databricks.com/en/delta/tutorial.html
```

---

## Extending the project

**Add a new specialist agent**
1. Create `agents/my_agent.py` with a `run_my_agent(state) -> dict` function
2. Add it as a node in `graph.py`: `graph.add_node("my_agent", run_my_agent)`
3. Add a routing case in `route_to_specialist()`
4. Add a terminal edge: `graph.add_edge("my_agent", END)`

**Add more knowledge**
Add URLs to `DOCS_URLS` in `rag/ingest.py` and re-run `python -m rag.ingest`. Any public webpage works — not just Databricks docs.

**Swap the LLM**
Change `GROQ_MODEL` in `.env`. Options on Groq's free tier:
- `llama-3.1-70b-versatile` (default — best quality)
- `llama-3.1-8b-instant` (fastest)
- `mixtral-8x7b-32768` (longest context window)

**Add a web UI**
Replace `main.py` with a [Gradio](https://gradio.app) or [Streamlit](https://streamlit.io) app — both are free. The `run_tutor()` function in `graph.py` is the only integration point needed.

---

## Concepts demonstrated

- **Multi-agent architecture** — orchestrator pattern with specialist agents
- **RAG (Retrieval-Augmented Generation)** — grounding LLM responses in real documentation
- **LangGraph state machines** — typed state, conditional edges, composable nodes
- **Local embeddings** — inference without API dependency
- **Vector similarity search** — cosine distance with score thresholding
- **Prompt engineering** — structured output formats enforced via system prompts

---

## License

MIT — use it, fork it, build on it.
