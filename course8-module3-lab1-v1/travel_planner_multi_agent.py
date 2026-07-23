"""Multi-agent travel planner: coordinator + destination + weather + culture.

Uses HandoffTool to wrap each specialist agent as a tool the coordinator
can invoke; AskPermissionRequirement gates the three handoff calls. A
handoff runs the specialist to completion in a fresh instance and returns
its output text as the tool result — the specialist's own iterations,
tools, and LLM calls are nested inside that one coordinator-visible call.
"""

import asyncio
import os

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.ask_permission import (
    AskPermissionRequirement,
)
from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool
from dotenv import load_dotenv

from helpers.hitl import stdin_approval_handler
from helpers.metrics import print_preview, print_run_metrics

if not load_dotenv():
    raise RuntimeError(".env not found - check cwd or run from lab root")
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY missing from .env")


TRAVEL_QUERY = (
    "I'm planning a 1 week trip to Singapore in September time "
    "as a first-time visitor. I want to experience traditional culture, visit "
    "historical sites, and interact with locals. I speak only English and want "
    "to be respectful of their customs. What should I know about the "
    "destination, weather expectations, and language/cultural tips?"
)


async def travel_planner_example() -> None:
    llm = ChatModel.from_name("anthropic:claude-haiku-4-5")

    destination_expert = RequirementAgent(
        llm=llm,
        tools=[WikipediaTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a Destination Research Expert. Provide detailed, factual "
            "information about landmarks, activities, best times to visit, "
            "transportation, and safety. Cite sources."
        ),
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=5,
                consecutive_allowed=False,
            ),
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=4,
                consecutive_allowed=False,
            ),
        ],
    )

    travel_meteorologist = RequirementAgent(
        llm=llm,
        tools=[OpenMeteoTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a Travel Meteorologist. Provide actionable weather "
            "guidance: climate patterns, packing suggestions, activity "
            "planning, and weather-related travel risks."
        ),
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=2,
            ),
            ConditionalRequirement(
                OpenMeteoTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=1,
            ),
        ],
    )

    language_and_culture_expert = RequirementAgent(
        llm=llm,
        tools=[WikipediaTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a Language & Cultural Expert. Provide essential phrases, "
            "cultural etiquette, communication tips, dining customs, and "
            "cultural sensitivities. Emphasise respectful travel practices."
        ),
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=3,
                consecutive_allowed=False,
            ),
        ],
    )

    handoff_to_destination = HandoffTool(
        destination_expert,
        name="DestinationResearch",
        description=(
            "Consult the Destination Research Expert for information about "
            "travel destinations, attractions, and practical guidance."
        ),
    )
    handoff_to_weather = HandoffTool(
        travel_meteorologist,
        name="WeatherPlanning",
        description=(
            "Consult the Travel Meteorologist for weather forecasts, climate "
            "analysis, and weather-appropriate recommendations."
        ),
    )
    handoff_to_language = HandoffTool(
        language_and_culture_expert,
        name="LanguageCulturalGuidance",
        description=(
            "Consult the Language & Cultural Expert for essential phrases, "
            "cultural etiquette, and communication guidance."
        ),
    )

    travel_coordinator = RequirementAgent(
        llm=llm,
        tools=[
            handoff_to_destination,
            handoff_to_weather,
            handoff_to_language,
            ThinkTool(),
        ],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are the Travel Coordinator. Delegate to specialist agents "
            "for destination, weather, and cultural guidance. Synthesise "
            "responses into a comprehensive travel plan."
        ),
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(ThinkTool, consecutive_allowed=False),
            AskPermissionRequirement(
                [handoff_to_destination, handoff_to_weather, handoff_to_language],
                handler=stdin_approval_handler,
            ),
        ],
    )

    result = await travel_coordinator.run(TRAVEL_QUERY)

    print_run_metrics(result.state)
    print_preview(result.output[-1].text, label="Travel Plan Preview", limit=400)


if __name__ == "__main__":
    asyncio.run(travel_planner_example())
