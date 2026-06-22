import random
import string
from typing import TypedDict

from langgraph.graph import StateGraph, END


class ChainState(TypedDict, total=False):
    n: int
    letter: str


def add_node(state: ChainState) -> ChainState:
    n = state.get("n", 0) + 1
    letter = random.choice(string.ascii_lowercase)
    print(f"Adding 1 to n and selecting a random letter: n={n}, letter='{letter}'")
    return {"n": n, "letter": letter}


def print_out_node(state: ChainState) -> ChainState:
    print("Current n:", state.get("n", 0), "Letter:", state.get("letter", ""))
    return state


def stop_condition(state: ChainState) -> bool:
    return state.get("n", 0) >= 13


workflow = StateGraph(ChainState)
workflow.add_node("AddNode", add_node)
workflow.add_node("PrintOutNode", print_out_node)
workflow.set_entry_point("AddNode")
workflow.add_edge("AddNode", "PrintOutNode")
workflow.add_conditional_edges(
    "PrintOutNode",
    stop_condition,
    {True: END, False: "AddNode"},
)

counter_app = workflow.compile()


if __name__ == "__main__":
    result = counter_app.invoke({})
    print("Final state:", result)