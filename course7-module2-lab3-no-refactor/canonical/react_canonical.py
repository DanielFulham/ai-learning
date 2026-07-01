import os
import json
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
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

# ---------------------------------------------------------------------------
# Model, prompt, state
# ---------------------------------------------------------------------------

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

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
    """The state of the agent.

    `add_messages` is the reducer that appends node returns of the shape
    `{"messages": [...]}` to the running transcript. See:
    https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    """

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


@tool
def recommend_clothing(weather: str) -> str:
    """
    Returns a clothing recommendation based on the provided weather description.

    This function examines the input string for specific keywords or temperature indicators 
    (e.g., "snow", "freezing", "rain", "85°F") to suggest appropriate attire. It handles 
    common weather conditions like snow, rain, heat, and cold by providing simple and practical 
    clothing advice.

    :param weather: A brief description of the weather (e.g., "Overcast, 64.9°F")
    :return: A string with clothing recommendations suitable for the weather
    """
    weather = weather.lower()
    if "snow" in weather or "freezing" in weather:
        return "Wear a heavy coat, gloves, and boots."
    elif "rain" in weather or "wet" in weather:
        return "Bring a raincoat and waterproof shoes."
    elif "hot" in weather or "85" in weather:
        return "T-shirt, shorts, and sunscreen recommended."
    elif "cold" in weather or "50" in weather:
        return "Wear a warm jacket or sweater."
    else:
        return "A light jacket should be fine."


tools = [search_tool, recommend_clothing]
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
    """Execute all tool calls from the last message in the state.

    The last message is always an AIMessage with non-empty tool_calls — an
    invariant enforced by the graph topology (only should_continue routes
    here, only when tool_calls is non-empty). The isinstance check documents
    that invariant in code and fails loudly if the topology is ever changed
    without updating this node.

    Plain strings pass through as-is; anything else gets JSON-encoded.
    Avoids wrapping `"A light jacket..."` as `'"A light jacket..."'` — the
    model shouldn't have to unquote its own tool results.
    """
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        raise TypeError(
            f"tool_node expected AIMessage as last message, got {type(last).__name__}"
        )

    outputs: list[ToolMessage] = []
    for tool_call in last.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        content = (
            tool_result
            if isinstance(tool_result, str)
            else json.dumps(tool_result, default=str)
        )
        outputs.append(
            ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def should_continue(state: AgentState):
    """Route from `agent`: to `tools` if the model emitted tool calls, else END.

    Runs after `call_model`, so the last message is always an AIMessage.
    """
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
    {
        "continue": "tools",
        "end": END,
    },
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


def render_graph_artefacts(compiled_graph, output_dir: str = ".") -> None:
    """Write Mermaid source + PNG to disk. No display step, no IPython."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mermaid_src = compiled_graph.get_graph().draw_mermaid()
    (out / "graph.mmd").write_text(mermaid_src, encoding="utf-8")

    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        (out / "graph.png").write_bytes(png_bytes)
    except urllib.error.URLError as e:
        print(f"[render_graph_artefacts] mermaid.ink unreachable, skipping PNG: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY not set in environment (.env)")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment (.env)")
    
    render_graph_artefacts(graph)

    inputs: AgentState = {
        "messages": [
            HumanMessage(
                content="What's the weather like in Zurich, and what should I wear based on the temperature?"
            )
        ]
    }

    print("=" * 60)
    print("Compiled graph — full ReAct loop")
    print("=" * 60)
    print_stream(graph.stream(inputs, stream_mode="values"))