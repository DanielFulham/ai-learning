"""RequirementAgent with full requirement scheduling: think-first, no
consecutive thoughts, Wikipedia mandatory but only after thinking.

Reconstructs a ReAct-like pattern at the scheduling layer. Cross-refs
F-L38-arch-17: last run's forced-Think agent skipped Wikipedia entirely.
This cell tests whether min_invocations=1 forces the tool despite the
model wanting to skip - the empirical test for BeeAI's 'predictable
controlled behavior' framing.
"""

import asyncio
import os

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from dotenv import load_dotenv

from helpers.metrics import print_run_metrics

if not load_dotenv():
    raise RuntimeError(".env not found - check cwd or run from lab root")
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY missing from .env")


SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment
and risk analysis.
Your methodology:
- Analyze the threat landscape systematically
- Research authoritative sources when available
- Provide comprehensive risk assessment with actionable recommendations
- Focus on practical, implementable security measures"""


ANALYSIS_QUERY = (
    "Analyze the cybersecurity risks of quantum computing for financial "
    "institutions. What are the main threats, timeline for concern, and "
    "recommended preparation strategies?"
)


async def controlled_execution_example() -> None:
    llm = ChatModel.from_name("anthropic:claude-haiku-4-5")

    agent = RequirementAgent(
        llm=llm,
        tools=[ThinkTool(), WikipediaTool()],
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=3,
                consecutive_allowed=False,
            ),
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=2,
            ),
        ],
    )

    result = await agent.run(ANALYSIS_QUERY)

    print_run_metrics(result.state)

    analysis = result.output[-1].text
    preview = analysis[:200] + "..." if len(analysis) > 200 else analysis
    print(f"\nAnalysis preview:\n{preview}")


if __name__ == "__main__":
    asyncio.run(controlled_execution_example())
