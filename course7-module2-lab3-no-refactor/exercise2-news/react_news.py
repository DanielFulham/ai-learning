"""Exercise 2 — ReAct agent with a news-summariser tool alongside search.

The summariser is LLM-as-tool: a second ChatOpenAI instance with a focused
system prompt handles extraction, invoked by the outer agent's tool call.
Same pattern as Lab 26's sql_db_query_checker. Cost implication: every
turn firing this tool costs two LLM invocations (outer decision + inner
summarisation).
"""

import os
import json
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from typing import Annotated, Sequence, TypedDict


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

if not os.environ.get("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY not set in environment (.env)")
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set in environment (.env)")


# ---------------------------------------------------------------------------
# Models, prompt, state
# ---------------------------------------------------------------------------

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

# Separate model instance for the summariser tool. Same underlying model,
# different invocation context — no bound tools, focused system prompt.
summariser_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities
3. Provide clear, helpful responses based on your reasoning and any tool results

Always explain your thinking process to help users understand your approach.
"""),
    MessagesPlaceholder(variable_name="scratch_pad"),
])


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

search = TavilySearch(max_results=5)


@tool
def search_tool(query: str):
    """
    Search the web for information using Tavily API.

    :param query: The search query string
    :return: Search results related to the query
    """
    return search.invoke(query)


_SUMMARISER_SYSTEM = """You are a news summarisation assistant. You will be given raw text
containing one or more news articles or search results. Produce a concise,
factual summary. For each distinct article, include:
- Headline (if identifiable)
- Date (if identifiable)
- 2-3 short bullet points covering the main facts

If the content is empty, malformed, or contains no article-shaped material,
say so plainly. Do not invent details. Do not add editorial commentary."""


@tool
def news_summarizer_tool(news_content: str) -> str:
    """
    Summarise news articles from raw content or search results.

    Takes the text output of a prior search or scrape and returns a
    structured, per-article summary. Best invoked AFTER search_tool has
    produced results; pass the tool_message content as the argument.

    :param news_content: Raw news content or JSON-serialised search results
    :return: A formatted per-article summary
    """
    response = summariser_model.invoke([
        SystemMessage(content=_SUMMARISER_SYSTEM),
        HumanMessage(content=news_content),
    ])
    return str(response.content)


tools = [search_tool, news_summarizer_tool]
tools_by_name = {t.name: t for t in tools}

model_react = chat_prompt | model.bind_tools(tools)


# ---------------------------------------------------------------------------
# Graph nodes and router
# ---------------------------------------------------------------------------

def call_model(state: AgentState):
    """Invoke the model with the current conversation state."""
    response = model_react.invoke({"scratch_pad": state["messages"]})
    return {"messages": [response]}


def tool_node(state: AgentState):
    """Execute all tool calls from the last message in the state."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        raise TypeError(
            f"tool_node expected AIMessage as last message, got {type(last).__name__}"
        )

    outputs: list[ToolMessage] = []
    for tool_call in last.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result, default=str),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def should_continue(state: AgentState):
    """Route from `agent`: to `tools` if the model emitted tool calls, else END."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(
            f"should_continue expected AIMessage as last message, got {type(last_message).__name__}"
        )

    if not last_message.tool_calls:
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
workflow.set_entry_point("agent")

graph = workflow.compile()


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def print_stream(stream) -> None:
    """Print each new message emitted between state snapshots."""
    seen = 0
    for s in stream:
        messages = s["messages"]
        for message in messages[seen:]:
            if isinstance(message, tuple):
                print(message)
            elif isinstance(message, BaseMessage):
                message.pretty_print()
            else:
                print(f"[unexpected type {type(message).__name__}]: {message}")
        seen = len(messages)


def render_graph_artefacts(
    compiled_graph, output_dir: str = ".", stem: str = "graph_news"
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mermaid_src = compiled_graph.get_graph().draw_mermaid()
    (out / f"{stem}.mmd").write_text(mermaid_src, encoding="utf-8")

    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        (out / f"{stem}.png").write_bytes(png_bytes)
    except urllib.error.URLError as e:
        print(f"[render_graph_artefacts] mermaid.ink unreachable, skipping PNG: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_graph_artefacts(graph)

    inputs: AgentState = {
        "messages": [
            HumanMessage(
                content="Find recent AI news and summarize the top 3 articles"
            )
        ]
    }

    print("=" * 60)
    print("Exercise 2 — News summariser (LLM-as-tool) added to ReAct agent")
    print("=" * 60)
    print_stream(graph.stream(inputs, stream_mode="values"))