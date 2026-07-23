"""RequirementAgent with AskPermissionRequirement + custom approval handler.

The custom handler surfaces tool arguments in the approval prompt, rather
than the framework default which shows the tool name only.

AskPermissionRequirement's interrupt only fires when the tool actually
invokes. min_invocations=1 on the WikipediaTool ConditionalRequirement
below forces the invocation; without it the model can skip the tool and
the approval prompt never fires.
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
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from dotenv import load_dotenv

from helpers.hitl import stdin_approval_handler
from helpers.metrics import print_preview, print_run_metrics

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


async def hitl_agent_example() -> None:
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
                max_invocations=2,
                consecutive_allowed=False,
            ),
            AskPermissionRequirement(
                WikipediaTool,
                handler=stdin_approval_handler,
            ),
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=1,
            ),
        ],
    )

    result = await agent.run(ANALYSIS_QUERY)

    print_run_metrics(result.state)
    print_preview(result.output[-1].text)


if __name__ == "__main__":
    asyncio.run(hitl_agent_example())
