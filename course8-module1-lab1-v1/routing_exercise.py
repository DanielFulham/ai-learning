from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from shared import llm, extract_content

ROUTER_NODE = "router_node"
RIDE_HAILING = "ride_hailing"
RESTAURANT_ORDER = "restaurant_order"
GROCERIES = "groceries"
DEFAULT_HANDLER = "default_handler"


class ExerciseRouterState(TypedDict):
    user_input: str
    task_type: str
    output: str


class RouterDecision(BaseModel):
    """Structured routing decision for the 4-way service router."""

    intent: Literal[
        "ride_hailing",
        "restaurant_order",
        "groceries",
        "default_handler",
    ] = Field(
        ...,
        description=(
            "Classify the user's request into exactly one service category. "
            "Return 'ride_hailing' if the user wants a ride, taxi, or transport. "
            "Return 'restaurant_order' if the user wants to order food from a "
            "restaurant. "
            "Return 'groceries' if the user wants to buy grocery items. "
            "Return 'default_handler' if the request does not fit any of the above."
        ),
    )


router_llm = llm.with_structured_output(RouterDecision)


def router_node(state: ExerciseRouterState) -> dict[str, str]:
    routing_prompt = (
        f"Classify the following user request.\n\n"
        f'User request: "{state["user_input"]}"'
    )
    response = router_llm.invoke(routing_prompt)
    if not isinstance(response, RouterDecision):
        raise TypeError(f"Expected RouterDecision, got {type(response).__name__}")
    return {"task_type": response.intent}


def route_by_intent(
    state: ExerciseRouterState,
) -> Literal["ride_hailing", "restaurant_order", "groceries", "default_handler"]:
    intent = state["task_type"]
    if intent == "ride_hailing":
        return "ride_hailing"
    if intent == "restaurant_order":
        return "restaurant_order"
    if intent == "groceries":
        return "groceries"
    if intent == "default_handler":
        return "default_handler"
    raise ValueError(f"Unknown intent: {intent!r}")


def ride_hailing_node(state: ExerciseRouterState) -> dict[str, str]:
    """Process ride-hailing requests by extracting pickup, dropoff, and preferences."""
    prompt = f"""
You are a ride-hailing assistant. Based on the user's request, extract and organise:

- Pickup location
- Destination
- Preferred ride type (if mentioned)
- Any special requirements
- Estimated timing preferences

User Request: "{state['user_input']}"

Provide a clear summary of the ride request with all available details.
"""
    response = llm.invoke(prompt)
    content = extract_content(response)
    return {"output": content.strip()}


def restaurant_order_node(state: ExerciseRouterState) -> dict[str, str]:
    """Process restaurant orders by organising menu items, quantities, and preferences."""
    prompt = f"""
You are a restaurant ordering assistant. Based on the user's request, organise:

- Menu items requested
- Quantities for each item
- Special modifications or dietary restrictions
- Delivery or pickup preference
- Any timing requirements

User Request: "{state['user_input']}"

Provide a clear, organised summary of the restaurant order with all details.
"""
    response = llm.invoke(prompt)
    content = extract_content(response)
    return {"output": content.strip()}


def groceries_node(state: ExerciseRouterState) -> dict[str, str]:
    """Process grocery delivery requests with driver pickup service."""
    prompt = f"""
You are a grocery delivery assistant for a service where our drivers shop for customers.

Based on the user's request, organise:

Shopping List:
- Grocery items needed
- Quantities or amounts
- Brand preferences (if mentioned)
- Dietary restrictions or organic preferences

Store Information:
- Preferred store or location
- Budget considerations
- Instructions for finding items

Delivery Details:
- Delivery address (if provided)
- Preferred delivery time window
- Special delivery instructions
- Contact information for driver coordination

Driver Instructions:
- Substitution preferences (if item unavailable)
- Handling of out-of-stock items
- Items requiring special handling (fragile, cold)
- Payment method (if mentioned)

User Request: "{state['user_input']}"

Provide a comprehensive delivery order summary the driver can use to shop and deliver efficiently.
"""
    response = llm.invoke(prompt)
    content = extract_content(response)
    return {"output": content.strip()}


def default_handler_node(state: ExerciseRouterState) -> dict[str, str]:
    prompt = f"""
I couldn't classify your request into a specific category.

User request: "{state['user_input']}"

I can help with:
- Ride hailing services
- Restaurant orders
- Grocery shopping

Please rephrase your request to match one of these services, or ask for customer support if
you need help with something else.

Would you like to:
1. Rephrase your request for one of our services
2. Connect with customer support for additional assistance
"""
    response = llm.invoke(prompt)
    content = extract_content(response)
    return {"output": content.strip()}


workflow = StateGraph(ExerciseRouterState)
workflow.add_node(ROUTER_NODE, router_node)
workflow.add_node(RIDE_HAILING, ride_hailing_node)
workflow.add_node(RESTAURANT_ORDER, restaurant_order_node)
workflow.add_node(GROCERIES, groceries_node)
workflow.add_node(DEFAULT_HANDLER, default_handler_node)
workflow.add_edge(START, ROUTER_NODE)
workflow.add_conditional_edges(
    ROUTER_NODE,
    route_by_intent,
    {
        RIDE_HAILING: RIDE_HAILING,
        RESTAURANT_ORDER: RESTAURANT_ORDER,
        GROCERIES: GROCERIES,
        DEFAULT_HANDLER: DEFAULT_HANDLER,
    },
)
workflow.add_edge(RIDE_HAILING, END)
workflow.add_edge(RESTAURANT_ORDER, END)
workflow.add_edge(GROCERIES, END)
workflow.add_edge(DEFAULT_HANDLER, END)

app = workflow.compile(checkpointer=InMemorySaver())