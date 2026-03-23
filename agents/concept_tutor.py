"""
agents/concept_tutor.py
-----------------------
Specialist agent for conceptual explanations.

Handles questions like:
    - "What is Delta Lake?"
    - "Explain Unity Catalog to me"
    - "How does Spark's execution model work?"
    - "What's the difference between a managed and external table?"

Strategy:
    1. Retrieve the top 5 most relevant doc chunks for the user's question
    2. Build a structured prompt with the retrieved context
    3. Ask the LLM to explain clearly with an analogy, key points, and a tip
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from rag.retriever import retrieve, format_context

load_dotenv()

SYSTEM_PROMPT = """You are the Concept Tutor — a friendly, expert Databricks instructor.
Your job is to explain Databricks concepts clearly to learners of all levels.

You are given:
  1. A user question
  2. Relevant excerpts from the official Databricks documentation

Your response MUST follow this exact structure:

## 🧠 Concept: [topic name]

**Plain English summary** (2–3 sentences max, no jargon)

**Analogy**
A real-world analogy that makes the concept click intuitively.

**Key points**
- Point 1
- Point 2
- Point 3 (add more if needed, keep each one concise)

**How it fits in Databricks**
One short paragraph placing this concept in the broader Databricks ecosystem.

**Pro tip**
One practical thing a learner should know or watch out for.

---
Rules:
- Ground every claim in the provided documentation context
- If the docs don't cover something, say so rather than guessing
- Keep the tone warm and encouraging — learners may be brand new
- Do NOT use the phrase "Great question!"
"""


def run_concept_tutor(state: dict) -> dict:
    """
    LangGraph node: retrieves docs context and generates a concept explanation.

    Args:
        state: LangGraph state dict with "user_input" key

    Returns:
        Updated state with "agent_response" and "sources" keys
    """
    user_input = state["user_input"]

    # RAG: fetch the most relevant documentation chunks
    chunks = retrieve(user_input, n_results=5)
    context = format_context(chunks)
    sources = list({c.source for c in chunks})   # deduplicated source URLs

    prompt = f"""Documentation context:
{context}

---

User question: {user_input}

Answer using the documentation above. Follow the structured format exactly.
"""

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        temperature=0.3,     # slight creativity for analogies, still grounded
        max_tokens=1024,
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        **state,
        "agent_response": response.content,
        "sources": sources,
        "agent_used": "concept_tutor",
    }
