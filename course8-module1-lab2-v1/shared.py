from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import CompiledStateGraph
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_TEMPERATURE = 0.7


def get_llm(temperature: float = DEFAULT_TEMPERATURE) -> ChatAnthropic:
    """Return a Claude Haiku 4.5 chat model.

    Temperature 0.7 matches the IBM notebook default;
    override lower where determinism matters (classifier / router seams).
    """
    return ChatAnthropic(
        model_name=DEFAULT_MODEL,
        temperature=temperature,
        timeout=None,
        stop=None,
    )

def describe_graph(app: CompiledStateGraph) -> None:
    """Print the graph topology — nodes and edges — for a quick sanity check."""
    graph = app.get_graph()
    print("Nodes:", list(graph.nodes))
    print("Edges:")
    for edge in graph.edges:
        print(f"  {edge.source} -> {edge.target}")


def render_graph_png(app: CompiledStateGraph, path: str) -> None:
    """Render the graph as a Mermaid PNG (via mermaid.ink) and write to disk."""
    png_bytes = app.get_graph().draw_mermaid_png()
    Path(path).write_bytes(png_bytes)