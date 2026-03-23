"""
agents/orchestrator.py
----------------------
The orchestrator is the entry point for every user message.
It classifies the intent and decides which specialist agent to invoke.

Routing logic:
    "concept"  -> concept_tutor   (explain, define, how does X work)
    "code"     -> code_guide      (write, fix, show me how to code X)
    "quiz"     -> quiz_master     (test me, quiz, flashcard, practice)
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are the orchestrator of a Databricks learning assistant.
Your ONLY job is to classify the user's message into exactly one of these three categories:

  concept  — the user wants an explanation, definition, or conceptual overview
             Examples: "What is Delta Lake?", "Explain Unity Catalog", "How does Spark work?"

  code     — the user wants code written, reviewed, debugged, or explained line-by-line
             Examples: "Write a PySpark job to read parquet", "Fix this notebook error",
                       "Show me how to use Auto Loader"

  quiz     — the user wants to be tested, quizzed, or given flashcards
             Examples: "Quiz me on Delta Lake", "Test my knowledge of clusters",
                       "Give me a flashcard on Z-Ordering"

Reply with ONLY the single word: concept, code, or quiz.
No punctuation. No explanation. Just the one word.
"""


def classify_intent(state: dict) -> dict:
    """
    LangGraph node: reads the latest user message, returns routing decision.

    Args:
        state: LangGraph state dict containing at minimum:
               - "messages": list of chat messages
               - "user_input": the raw user string

    Returns:
        Updated state with "route" key set to "concept" | "code" | "quiz"
    """
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        temperature=0,       # deterministic — we want consistent routing
        max_tokens=10,       # only need one word back
    )

    user_input = state["user_input"]

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ])

    # Sanitise the response — strip whitespace, lowercase, fallback to "concept"
    raw = response.content.strip().lower()
    route = raw if raw in {"concept", "code", "quiz"} else "concept"

    return {**state, "route": route}
