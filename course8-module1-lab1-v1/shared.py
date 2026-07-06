import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from dotenv import load_dotenv

load_dotenv()


def make_chat_model(provider: str = "anthropic") -> BaseChatModel:
    """Return a chat model for the given provider.

    Defaults to Anthropic (Claude Haiku) to match Hooperman's production model.
    OpenAI kept as an option for the parallel-pattern cross-provider comparison.
    """
    if provider == "anthropic":
        return ChatAnthropic(model_name="claude-haiku-4-5", timeout=None, stop=None)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini")
    raise ValueError(f"Unknown provider: {provider!r}. Expected 'anthropic' or 'openai'.")


# Provider selected at import time via LLM_PROVIDER env var (default: anthropic).
# This module-level singleton is what the pattern modules import as `llm`.
llm: BaseChatModel = make_chat_model(os.environ.get("LLM_PROVIDER", "anthropic"))


def describe_graph(compiled_app) -> None:
    """Print topology of a compiled LangGraph app."""
    graph = compiled_app.get_graph()
    print("WORKFLOW TOPOLOGY")
    print("=================")
    print(f"Nodes: {list(graph.nodes)}")
    print(f"Edges: {[(e.source, e.target) for e in graph.edges]}")


def render_graph_png(compiled_app, path: str) -> None:
    """Write a Mermaid PNG of the compiled graph to disk."""
    png_bytes = compiled_app.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(png_bytes)


def extract_content(response: BaseMessage) -> str:
    """Narrow BaseMessage.content to str at the LLM boundary.

    LangChain types content as str | list[str | dict] because some models return
    structured content blocks. For our text-generation prompts the content is
    always str at runtime, but pyright sees the union. Narrow at the seam.
    """
    return response.content if isinstance(response.content, str) else str(response.content)