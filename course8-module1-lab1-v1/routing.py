from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from shared import llm, extract_content

ROUTER_NODE = "router_node"
SUMMARIZE = "summarize"
TRANSLATE = "translate"


class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str


class RouterDecision(BaseModel):
    """Structured routing decision emitted by the router node."""

    intent: Literal["summarize", "translate"] = Field(
        ...,
        description=(
            "Classify the user's request. Return 'summarize' if the user wants "
            "a passage summarised. Return 'translate' if the user wants text "
            "translated into French."
        ),
    )


router_llm = llm.with_structured_output(RouterDecision)


def router_node(state: RouterState) -> dict[str, str]:
    routing_prompt = (
        f"Classify the following user request.\n\n"
        f'User request: "{state["user_input"]}"'
    )
    response = router_llm.invoke(routing_prompt)
    if not isinstance(response, RouterDecision):
        raise TypeError(f"Expected RouterDecision, got {type(response).__name__}")
    return {"task_type": response.intent}


def route_by_intent(state: RouterState) -> str:
    return state["task_type"]


def summarize_node(state: RouterState) -> dict[str, str]:
    prompt = f"Please summarize the following passage:\n\n{state['user_input']}"
    response = llm.invoke(prompt)
    content = extract_content(response)
    return {"output": content}


def translate_node(state: RouterState) -> dict[str, str]:
    prompt = f"Translate the following text to French:\n\n{state['user_input']}"
    response = llm.invoke(prompt)
    content = extract_content(response)
    return {"output": content}


workflow = StateGraph(RouterState)
workflow.add_node(ROUTER_NODE, router_node)
workflow.add_node(SUMMARIZE, summarize_node)
workflow.add_node(TRANSLATE, translate_node)
workflow.add_edge(START, ROUTER_NODE)
workflow.add_conditional_edges(
    ROUTER_NODE,
    route_by_intent,
    {SUMMARIZE: SUMMARIZE, TRANSLATE: TRANSLATE},
)
workflow.add_edge(SUMMARIZE, END)
workflow.add_edge(TRANSLATE, END)

app = workflow.compile(checkpointer=InMemorySaver())