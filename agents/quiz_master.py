"""
agents/quiz_master.py
---------------------
Specialist agent for testing and reinforcing knowledge.

Handles requests like:
    - "Quiz me on Delta Lake"
    - "Give me 3 flashcards on Unity Catalog"
    - "Test my understanding of Spark execution"
    - "I think the answer is X — am I right?"

Two modes detected from user input:
    GENERATE  — user wants a quiz or flashcards generated
    EVALUATE  — user is submitting an answer to a previous question

Strategy:
    1. Retrieve relevant docs to ground questions in real content
    2. Generate MCQs with one correct answer and three plausible distractors
       OR evaluate a submitted answer with kind, detailed feedback
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from rag.retriever import retrieve, format_context

load_dotenv()

GENERATE_PROMPT = """You are the Quiz Master — an expert Databricks instructor who creates
engaging, accurate quiz questions to test and reinforce learning.

You are given:
  1. A topic the user wants to be quizzed on
  2. Relevant excerpts from the official Databricks documentation

Your response MUST follow this exact structure:

## 📝 Quiz: [topic]

[For each question — generate 3 questions unless the user specifies a number]

**Question [N]**
[The question text]

A) [Option]
B) [Option]
C) [Option]
D) [Option]

<details>
<summary>Show answer</summary>

**Correct answer: [Letter]) [Option text]**

**Explanation:** [2–3 sentences explaining why this is correct, grounded in the docs]

</details>

---
Rules:
- Every question must be grounded in the retrieved documentation
- Distractors (wrong answers) should be plausible — not obviously silly
- Avoid "all of the above" and "none of the above"
- Questions should test understanding, not just memorisation
- Vary question types: definition, comparison, scenario-based, troubleshooting
"""

EVALUATE_PROMPT = """You are the Quiz Master — an expert Databricks instructor giving
warm, detailed feedback on a learner's answer.

You are given:
  1. The learner's submitted answer (may include what question they were answering)
  2. Relevant excerpts from the official Databricks documentation

Your response MUST follow this exact structure:

## ✅ Feedback

**Your verdict:** [Correct / Partially correct / Incorrect]

**What you got right**
[Specific things the learner understood correctly]

**What to refine**
[Specific misconceptions or gaps, explained kindly]

**The full answer**
[Complete, correct explanation grounded in the documentation]

**Remember this**
One memorable takeaway or mnemonic to help it stick.

---
Rules:
- Always be encouraging — learning Databricks takes time
- Be specific about what was right and wrong, not vague
- Ground the correct answer in the provided documentation
"""


def _detect_mode(user_input: str) -> str:
    """
    Simple heuristic to detect if the user wants a quiz generated
    or is submitting an answer for evaluation.
    """
    evaluation_signals = [
        "i think", "my answer", "is it", "am i right", "is that correct",
        "the answer is", "i believe", "i'd say", "i would say",
    ]
    lower = user_input.lower()
    return "evaluate" if any(s in lower for s in evaluation_signals) else "generate"


def run_quiz_master(state: dict) -> dict:
    """
    LangGraph node: generates a quiz or evaluates a submitted answer.

    Args:
        state: LangGraph state dict with "user_input" key

    Returns:
        Updated state with "agent_response" and "sources" keys
    """
    user_input = state["user_input"]
    mode = _detect_mode(user_input)

    # For evaluation, also pull in prior quiz context if available
    retrieval_query = user_input
    chunks = retrieve(retrieval_query, n_results=5)
    context = format_context(chunks)
    sources = list({c.source for c in chunks})

    system_prompt = GENERATE_PROMPT if mode == "generate" else EVALUATE_PROMPT

    prompt = f"""Documentation context:
{context}

---

User input: {user_input}

{"Generate quiz questions" if mode == "generate" else "Evaluate this answer"} using the documentation above. Follow the structured format exactly.
"""

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        temperature=0.4,     # slight variation so repeated quizzes feel fresh
        max_tokens=1500,
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])

    return {
        **state,
        "agent_response": response.content,
        "sources": sources,
        "agent_used": "quiz_master",
        "quiz_mode": mode,
    }
