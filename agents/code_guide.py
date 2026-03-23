"""
agents/code_guide.py
--------------------
Specialist agent for writing, explaining, and debugging code.

Handles questions like:
    - "Write a PySpark job that reads a Delta table and filters by date"
    - "Show me how to use Auto Loader for streaming ingestion"
    - "Explain what this notebook cell is doing"
    - "How do I configure a cluster with spot instances?"

Strategy:
    1. Retrieve relevant docs (focuses on API references and examples)
    2. Generate well-commented, runnable code with a clear explanation
    3. Include common pitfalls and how to run it in Databricks
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from rag.retriever import retrieve, format_context

load_dotenv()

SYSTEM_PROMPT = """You are the Code Guide — a senior Databricks and data engineering expert.
Your job is to write clean, well-commented code and explain it clearly.

You are given:
  1. A user request (write code, explain code, or fix code)
  2. Relevant excerpts from the official Databricks documentation

Your response MUST follow this exact structure:

## 💻 Code: [short description]

**What this does** (1–2 sentences)

```python
# Your code here — always well-commented
```

**Step-by-step explanation**
Walk through the code in plain English, one logical block at a time.

**How to run this in Databricks**
Brief instructions: what cluster type, any init scripts, notebook setup etc.

**Common pitfalls**
- Pitfall 1 and how to avoid it
- Pitfall 2 and how to avoid it

---
Rules:
- Always write complete, runnable code — no pseudo-code or placeholders
- Add inline comments to every non-obvious line
- Use PySpark, Spark SQL, or the Databricks Python SDK as appropriate
- If the user asks for SQL, write Databricks SQL (not generic ANSI SQL where they differ)
- Ground explanations in the provided documentation
- If the docs don't cover the exact request, write best-practice code and flag it
"""


def run_code_guide(state: dict) -> dict:
    """
    LangGraph node: retrieves docs context and generates a code response.

    Args:
        state: LangGraph state dict with "user_input" key

    Returns:
        Updated state with "agent_response" and "sources" keys
    """
    user_input = state["user_input"]

    # Enrich the retrieval query with "example code" to bias toward API docs
    retrieval_query = f"{user_input} code example API"
    chunks = retrieve(retrieval_query, n_results=5)
    context = format_context(chunks)
    sources = list({c.source for c in chunks})

    prompt = f"""Documentation context:
{context}

---

User request: {user_input}

Write the code and explanation using the documentation above. Follow the structured format exactly.
"""

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        temperature=0.1,     # low temperature — code needs to be precise
        max_tokens=2048,     # code responses can be long
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        **state,
        "agent_response": response.content,
        "sources": sources,
        "agent_used": "code_guide",
    }
