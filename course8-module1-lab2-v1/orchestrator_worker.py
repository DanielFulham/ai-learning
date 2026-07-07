"""Orchestrator-Worker pattern: meal planning via dynamic Send() dispatch.

Demonstrates AI_PATTERNS.md §4 with async chef workers fanning out from an
orchestrator planner, results merged via Annotated reducer, joined by a
synthesizer node.
"""

import asyncio
import operator
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from shared import describe_graph, get_llm, render_graph_png

ORCHESTRATOR_NODE = "orchestrator"
CHEF_WORKER_NODE = "chef_worker"
SYNTHESIZER_NODE = "synthesizer"

# ---------------------------------------------------------------------------
# Planner schema
# ---------------------------------------------------------------------------


class Dish(BaseModel):
    """A single dish with cuisine origin and ingredient list."""

    name: str = Field(description="Name of the dish (e.g. 'Spaghetti Bolognese', 'Chicken Curry').")
    ingredients: list[str] = Field(
        description="Ingredients needed for this dish, one item per list element."
    )
    location: str = Field(
        description="Cuisine or cultural origin (e.g. 'Italian', 'Indian', 'Mexican')."
    )


class Dishes(BaseModel):
    """Structured breakdown of a meal request into per-dish sections."""

    sections: list[Dish] = Field(
        description="One section per dish, each with name, ingredients, and cuisine."
    )


class MealPlanState(TypedDict):
    """State passed between orchestrator, chef workers, and synthesizer."""

    meals: str
    """User's input listing the meals to prepare."""

    sections: list[Dish]
    """Orchestrator output: one Dish per meal, with ingredients and cuisine."""

    completed_menu: Annotated[list[str], operator.add]
    """Worker output: cooking instructions per dish."""

    final_meal_guide: str
    """Synthesizer output: joined menu ready for display."""


class WorkerState(TypedDict):
    """State handed to each chef_worker invocation via Send()."""

    section: Dish
    completed_menu: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

dish_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that generates a structured dish list.\n\n"
            "For each meal the user names, return a section with:\n"
            "- the name of the dish\n"
            "- the list of ingredients needed\n"
            "- the cuisine or cultural origin of the food",
        ),
        ("human", "The meals to prepare: {meals}"),
    ]
)

# ---------------------------------------------------------------------------
# Planner pipe: prompt → LLM → structured Dishes output
# ---------------------------------------------------------------------------

planner_pipe = dish_prompt | get_llm().with_structured_output(Dishes)

# ---------------------------------------------------------------------------
# Chef worker prompt and pipe
# ---------------------------------------------------------------------------

chef_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world-class chef.\n\n"
            "Introduce yourself briefly and walk through preparing the dish requested.\n"
            "Your response should include:\n"
            "- A hello, your name, and your culinary background\n"
            "- A clear list of preparation steps\n"
            "- A full explanation of the cooking process",
        ),
        (
            "human",
            "You are a chef from {location}. Prepare the dish: {name}.\n"
            "Use these ingredients: {ingredients}.",
        ),
    ]
)

chef_pipe = chef_prompt | get_llm()

# ---------------------------------------------------------------------------
# Orchestrator node
# ---------------------------------------------------------------------------


def orchestrator(state: MealPlanState) -> dict[str, list[Dish]]:
    """Break the user's meal input into structured Dish sections."""
    result = planner_pipe.invoke({"meals": state["meals"]})
    if not isinstance(result, Dishes):
        raise TypeError(f"planner_pipe returned {type(result).__name__}, expected Dishes")
    return {"sections": result.sections}


# ---------------------------------------------------------------------------
# Chef worker dispatch and node
# ---------------------------------------------------------------------------


def dispatch_to_chefs(state: MealPlanState) -> list[Send]:
    """Fan out one chef_worker per dish via LangGraph's Send API."""
    return [Send(CHEF_WORKER_NODE, {"section": s}) for s in state["sections"]]


async def chef_worker(state: WorkerState) -> dict[str, list[str]]:
    """Generate cooking instructions for one dish."""
    section = state["section"]
    meal_plan = await chef_pipe.ainvoke(
        {
            "name": section.name,
            "location": section.location,
            "ingredients": section.ingredients,
        }
    )
    if not isinstance(meal_plan.content, str):
        raise TypeError(
            f"chef_pipe returned content of type {type(meal_plan.content).__name__}, expected str"
        )
    return {"completed_menu": [meal_plan.content]}


# ---------------------------------------------------------------------------
# Synthesizer node
# ---------------------------------------------------------------------------


def synthesizer(state: MealPlanState) -> dict[str, str]:
    """Join per-dish cooking instructions into the final menu."""
    completed_menu = "\n\n---\n\n".join(state["completed_menu"])
    return {"final_meal_guide": completed_menu}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(MealPlanState)

_builder.add_node(ORCHESTRATOR_NODE, orchestrator)
_builder.add_node(CHEF_WORKER_NODE, chef_worker)
_builder.add_node(SYNTHESIZER_NODE, synthesizer)

_builder.add_edge(START, ORCHESTRATOR_NODE)

_builder.add_conditional_edges(
    ORCHESTRATOR_NODE,
    dispatch_to_chefs,
    [CHEF_WORKER_NODE],
)

_builder.add_edge(CHEF_WORKER_NODE, SYNTHESIZER_NODE)
_builder.add_edge(SYNTHESIZER_NODE, END)

app = _builder.compile(checkpointer=InMemorySaver())

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    async def _run() -> None:
        describe_graph(app)
        png_path = Path(__file__).parent / "graph_orchestrator_worker.png"
        render_graph_png(app, str(png_path))
        print(f"Graph rendered to {png_path.name}\n")

        result = await app.ainvoke(
            {
                "meals": "Spaghetti Bolognese and Chicken Stir Fry",
                "sections": [],
                "completed_menu": [],
                "final_meal_guide": "",
            },
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )

        print("=" * 60)
        print("FINAL MEAL GUIDE")
        print("=" * 60)
        print(result["final_meal_guide"])

    asyncio.run(_run())
