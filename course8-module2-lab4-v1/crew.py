"""Crew definitions for NourishBot's recipe and analysis workflows.

Two crews, plain-Python (no @CrewBase decorators, no base class inheritance):

- `build_recipe_crew()` — three-agent sequential pipeline:
  ingredient_detection → dietary_filtering → recipe_suggestion.
- `build_analysis_crew()` — single-agent crew wrapping the analyse_nutrition
  tool. Wrapping is framework overhead; the tool itself already collapses
  vision + structured output into one call (see tools.py).

Design notes:

- Kickoff-time inputs, not construction-time. Both crews take
  `crew.kickoff(inputs=...)` with kickoff-inputs interpolated into
  `{uploaded_image}` and `{dietary_restrictions}` template slots in
  tasks.yaml. Crews are constructed once at app start, reused across
  Gradio requests. Matches L35's TypedDict + kickoff-inputs seam pattern.

- Task inter-propagation via explicit `context=[upstream]`, not
  `depends_on=` + `input_data=lambda`. The latter is not a real CrewAI Task
  API; `context=[]` is the canonical shape (L35 finding, verified against
  CrewAI 1.15.x docs).

- Structured output via `output_pydantic=` on the two user-facing tasks
  (nutrient_analysis, recipe_suggestion), producing typed .pydantic
  attribute access on CrewOutput. `output_json=` (IBM's usage) is the
  legacy API.

- `allow_delegation=False` on every agent. IBM had `True` on
  dietary_filtering_agent — L34 anti-pattern (50%+ token bloat when it
  fires, no delegation actually needed for a text-filter task).

- `max_iter` tuned per L35: retrieval/tool-call agents want low iteration
  counts (3), synthesis agents want moderate (3 for output_pydantic
  validation-retry room). Analysis at 2 because the collapse-in-tool
  workflow needs one tool call and one final-answer pass.

- Single-agent crew (analysis) finding: wrapping a single-tool workflow in
  CrewAI forces an extra LLM call for the agent to restructure the tool's
  already-typed output into its own final answer. V1 preserves framework
  shape; this cost is real and worth banking as a finding about where the
  structure-guarantee actually pays off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, TypedDict

import yaml
from crewai import LLM, Agent, Crew, Process, Task

from llm import build_crewai_llm
from models import NutrientAnalysisOutput, RecipeSuggestionOutput
from tools import (
    analyse_nutrition,
    extract_ingredients,
    filter_by_dietary_restriction,
    filter_ingredients,
)

_CONFIG_DIR: Final[Path] = Path(__file__).parent / "config"

_agents_config: Final[dict[str, Any]] = yaml.safe_load(
    (_CONFIG_DIR / "agents.yaml").read_text()
)
_tasks_config: Final[dict[str, Any]] = yaml.safe_load(
    (_CONFIG_DIR / "tasks.yaml").read_text()
)


class RecipeInputs(TypedDict):
    uploaded_image: str
    dietary_restrictions: str


class AnalysisInputs(TypedDict):
    uploaded_image: str


def _build_ingredient_detection_agent(llm: LLM) -> Agent:
    return Agent(
        config=_agents_config["ingredient_detection_agent"],
        tools=[extract_ingredients, filter_ingredients],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
        verbose=True,
    )


def _build_dietary_filtering_agent(llm: LLM) -> Agent:
    return Agent(
        config=_agents_config["dietary_filtering_agent"],
        tools=[filter_by_dietary_restriction],
        llm=llm,
        allow_delegation=False,  # IBM had True; L34 anti-pattern
        max_iter=3,
        verbose=True,
    )


def _build_nutrient_analysis_agent(llm: LLM) -> Agent:
    return Agent(
        config=_agents_config["nutrient_analysis_agent"],
        tools=[analyse_nutrition],
        llm=llm,
        allow_delegation=False,
        max_iter=2,
        verbose=True,
    )


def _build_recipe_suggestion_agent(llm: LLM) -> Agent:
    return Agent(
        config=_agents_config["recipe_suggestion_agent"],
        tools=[],  # Pure synthesis from upstream context; no tools needed.
        llm=llm,
        allow_delegation=False,
        max_iter=3,
        verbose=True,
    )


def _build_ingredient_detection_task(agent: Agent) -> Task:
    task_config = _tasks_config["ingredient_detection_task"]
    return Task(
        description=task_config["description"],
        expected_output=task_config["expected_output"],
        agent=agent,
    )


def _build_dietary_filtering_task(agent: Agent, upstream: Task) -> Task:
    task_config = _tasks_config["dietary_filtering_task"]
    return Task(
        description=task_config["description"],
        expected_output=task_config["expected_output"],
        agent=agent,
        context=[upstream],
    )


def _build_nutrient_analysis_task(agent: Agent) -> Task:
    task_config = _tasks_config["nutrient_analysis_task"]
    return Task(
        description=task_config["description"],
        expected_output=task_config["expected_output"],
        agent=agent,
        output_pydantic=NutrientAnalysisOutput,
    )


def _build_recipe_suggestion_task(agent: Agent, upstream: Task) -> Task:
    task_config = _tasks_config["recipe_suggestion_task"]
    return Task(
        description=task_config["description"],
        expected_output=task_config["expected_output"],
        agent=agent,
        context=[upstream],
        output_pydantic=RecipeSuggestionOutput,
    )


def build_recipe_crew() -> Crew:
    llm = build_crewai_llm()

    detection_agent = _build_ingredient_detection_agent(llm)
    filtering_agent = _build_dietary_filtering_agent(llm)
    recipe_agent = _build_recipe_suggestion_agent(llm)

    detection_task = _build_ingredient_detection_task(detection_agent)
    filtering_task = _build_dietary_filtering_task(filtering_agent, detection_task)
    recipe_task = _build_recipe_suggestion_task(recipe_agent, filtering_task)

    return Crew(
        agents=[detection_agent, filtering_agent, recipe_agent],
        tasks=[detection_task, filtering_task, recipe_task],
        process=Process.sequential,
        verbose=True,
    )


def build_analysis_crew() -> Crew:
    llm = build_crewai_llm()

    analysis_agent = _build_nutrient_analysis_agent(llm)
    analysis_task = _build_nutrient_analysis_task(analysis_agent)

    return Crew(
        agents=[analysis_agent],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True,
    )
