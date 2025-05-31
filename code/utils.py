from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def print_title():
    console.print(Panel.fit("[bold blue]Test[/bold blue]", padding=(1, 4)))

def print_question(q):
    console.print(f"\n[bold green]Question:[/bold green] {q}")

def print_context_snippets(snippets):
    console.print(Panel("\n\n".join(snippets), title="Retrieved Passages", style="dim"))

def print_answer(ans):
    console.print(Panel(Markdown(ans), title="Answer", border_style="blue"))

def generate_paraphrases(query, n=3):
    """
    Return a list of paraphrased versions of the query.
    This version uses rule-based variants; swap with GPT-generated if needed.
    """
    base = query.strip().strip("?").capitalize()
    paraphrases = [
        base,
        f"What do we know about {base}?",
        f"Could you explain {base}?",
        f"Summarize the concept of {base}.",
        f"I'd like to learn about {base}.",
        f"Please provide insight into {base}.",
    ]
    return paraphrases[:n]
