"""
main.py
-------
Interactive CLI demo for the Databricks Multi-Agent Tutor.

Usage:
    # First time only — build the knowledge base:
    python -m rag.ingest

    # Then start the tutor:
    python main.py

Commands during the session:
    /quit     — exit
    /sources  — show sources from last response
    /agent    — show which agent handled last response
    /help     — show available commands
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

load_dotenv()

console = Console()

AGENT_LABELS = {
    "concept_tutor": ("🧠 Concept Tutor", "purple"),
    "code_guide":    ("💻 Code Guide",    "cyan"),
    "quiz_master":   ("📝 Quiz Master",   "green"),
}

WELCOME = """
# Databricks Multi-Agent Tutor

Ask me anything about Databricks. I'll route your question to the right specialist automatically.

**Try asking:**
- *"What is Delta Lake and why does it matter?"*
- *"Write a PySpark job that reads a Delta table and filters rows by date"*
- *"Quiz me on Unity Catalog"*

Type `/help` for commands.
"""


def check_knowledge_base():
    """Warn the user if the ChromaDB knowledge base hasn't been built yet."""
    chroma_path = "./chroma_db"
    if not os.path.exists(chroma_path):
        console.print(Panel(
            "[bold yellow]Knowledge base not found.[/bold yellow]\n\n"
            "Run this first to build it:\n\n"
            "  [bold cyan]python -m rag.ingest[/bold cyan]\n\n"
            "This takes ~2 minutes and only needs to be done once.",
            title="⚠  Setup required",
            border_style="yellow",
        ))
        sys.exit(1)


def check_api_key():
    """Exit early with a clear message if the Groq API key is missing."""
    if not os.getenv("GROQ_API_KEY"):
        console.print(Panel(
            "[bold yellow]GROQ_API_KEY not set.[/bold yellow]\n\n"
            "1. Get a free key at [link=https://console.groq.com]console.groq.com[/link]\n"
            "2. Copy [cyan].env.example[/cyan] to [cyan].env[/cyan]\n"
            "3. Paste your key into [cyan].env[/cyan]",
            title="⚠  Missing API key",
            border_style="yellow",
        ))
        sys.exit(1)


def print_welcome():
    console.print(Markdown(WELCOME))
    console.print(Rule(style="dim"))


def print_help():
    console.print(Panel(
        "[bold]/quit[/bold]    — exit the tutor\n"
        "[bold]/sources[/bold] — show documentation sources from last response\n"
        "[bold]/agent[/bold]   — show which specialist handled last response\n"
        "[bold]/help[/bold]    — show this message",
        title="Commands",
        border_style="dim",
    ))


def print_response(result: dict):
    """Render the agent response with a labelled panel."""
    agent_key = result.get("agent_used", "concept_tutor")
    label, color = AGENT_LABELS.get(agent_key, ("Tutor", "white"))

    console.print()
    console.print(Panel(
        Markdown(result["agent_response"]),
        title=f"[bold {color}]{label}[/bold {color}]",
        border_style=color,
        padding=(1, 2),
    ))


def print_sources(sources: list[str]):
    if not sources:
        console.print("[dim]No sources available.[/dim]")
        return

    console.print("\n[bold]Sources used:[/bold]")
    for url in sources:
        console.print(f"  [dim blue][link={url}]{url}[/link][/dim blue]")


def print_routing(route: str, agent: str):
    """Show a subtle routing indicator so the multi-agent architecture is visible."""
    console.print(
        f"\n[dim]  ↳ Orchestrator routed [bold]{route!r}[/bold] "
        f"→ [bold]{agent}[/bold][/dim]"
    )


def main():
    check_api_key()
    check_knowledge_base()

    # Import here so missing chroma_db gives a clean error above first
    from graph import run_tutor

    print_welcome()

    last_result = None

    while True:
        try:
            # Prompt
            console.print()
            user_input = console.input("[bold green]You >[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/sources":
            if last_result:
                print_sources(last_result.get("sources", []))
            else:
                console.print("[dim]No previous response yet.[/dim]")
            continue

        if user_input.lower() == "/agent":
            if last_result:
                agent = last_result.get("agent_used", "unknown")
                route = last_result.get("route", "unknown")
                console.print(f"\n  Route: [bold]{route}[/bold] → Agent: [bold]{agent}[/bold]")
            else:
                console.print("[dim]No previous response yet.[/dim]")
            continue

        # --- Run the multi-agent pipeline ---
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            try:
                result = run_tutor(user_input)
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                console.print("[dim]Check your GROQ_API_KEY and that the knowledge base is built.[/dim]")
                continue

        last_result = result

        # Show routing decision (makes the multi-agent architecture visible)
        print_routing(result.get("route", "?"), result.get("agent_used", "?"))

        # Render the response
        print_response(result)

        # Always show sources below the response
        print_sources(result.get("sources", []))

        console.print(Rule(style="dim"))


if __name__ == "__main__":
    main()
