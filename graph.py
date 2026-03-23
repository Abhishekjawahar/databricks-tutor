"""
graph.py
--------
The LangGraph state machine for the Databricks Multi-Agent Tutor.

Flow:
    user_input
        │
        ▼
    [orchestrator]  ── classifies intent ──► route: "concept" | "code" | "quiz"
        │
        ├── "concept" ──► [concept_tutor] ──┐
        ├── "code"    ──► [code_guide]    ──┤──► [END]
        └── "quiz"    ──► [quiz_master]   ──┘

State is a typed dict that flows through every node.
Each node receives the full state and returns a partial update.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from agents.orchestrator import classify_intent
from agents.concept_tutor import run_concept_tutor
from agents.code_guide import run_code_guide
from agents.quiz_master import run_quiz_master


# ---------------------------------------------------------------------------
# 1. Shared state schema
#    Every node reads from and writes to this typed dict.
#    LangGraph merges partial updates — nodes only need to return
#    the keys they change.
# ---------------------------------------------------------------------------

class TutorState(TypedDict):
    # Set by the caller before graph invocation
    user_input: str

    # Set by the orchestrator node
    route: Literal["concept", "code", "quiz"]

    # Set by whichever specialist agent runs
    agent_response: str
    agent_used: str
    sources: list[str]

    # Optional — only set by quiz_master
    quiz_mode: str


# ---------------------------------------------------------------------------
# 2. Routing function
#    LangGraph calls this after the orchestrator node to decide
#    which specialist edge to follow.
# ---------------------------------------------------------------------------

def route_to_specialist(state: TutorState) -> Literal["concept_tutor", "code_guide", "quiz_master"]:
    """
    Pure routing function — reads the 'route' key set by the orchestrator
    and returns the name of the next node to execute.

    This is used as the conditional edge function in the graph definition.
    """
    mapping = {
        "concept": "concept_tutor",
        "code":    "code_guide",
        "quiz":    "quiz_master",
    }
    # Default to concept_tutor if something unexpected slips through
    return mapping.get(state["route"], "concept_tutor")


# ---------------------------------------------------------------------------
# 3. Graph definition
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph state machine.

    Returns a compiled graph ready to be invoked with:
        graph.invoke({"user_input": "..."})
    """
    graph = StateGraph(TutorState)

    # -- Add nodes (each node is a function: state -> partial state update)
    graph.add_node("orchestrator",   classify_intent)
    graph.add_node("concept_tutor",  run_concept_tutor)
    graph.add_node("code_guide",     run_code_guide)
    graph.add_node("quiz_master",    run_quiz_master)

    # -- Entry point
    graph.set_entry_point("orchestrator")

    # -- Conditional edge: after orchestrator, route based on intent
    graph.add_conditional_edges(
        source="orchestrator",
        path=route_to_specialist,
        path_map={
            "concept_tutor": "concept_tutor",
            "code_guide":    "code_guide",
            "quiz_master":   "quiz_master",
        },
    )

    # -- Terminal edges: all specialists lead to END
    graph.add_edge("concept_tutor", END)
    graph.add_edge("code_guide",    END)
    graph.add_edge("quiz_master",   END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 4. Public helper — single entry point for the whole pipeline
# ---------------------------------------------------------------------------

def run_tutor(user_input: str) -> TutorState:
    """
    Run the full multi-agent pipeline for a single user query.

    Args:
        user_input: The raw question or request from the user.

    Returns:
        The final TutorState with agent_response, sources, and agent_used populated.

    Example:
        result = run_tutor("What is Delta Lake?")
        print(result["agent_response"])
        print(result["sources"])
    """
    graph = build_graph()

    initial_state: TutorState = {
        "user_input":     user_input,
        "route":          "concept",   # default — overwritten by orchestrator
        "agent_response": "",
        "agent_used":     "",
        "sources":        [],
        "quiz_mode":      "",
    }

    result = graph.invoke(initial_state)
    return result


# ---------------------------------------------------------------------------
# 5. Visualise the graph (optional, useful for the README / portfolio)
# ---------------------------------------------------------------------------

def print_graph_structure():
    """
    Print a Mermaid diagram of the graph to stdout.
    Paste the output into https://mermaid.live to render it.
    """
    graph = build_graph()
    print("\n--- Mermaid diagram (paste into mermaid.live) ---\n")
    print(graph.get_graph().draw_mermaid())
    print("\n-------------------------------------------------\n")


if __name__ == "__main__":
    print_graph_structure()
