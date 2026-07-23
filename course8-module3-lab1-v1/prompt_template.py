"""Prompt template + ChatModel via BeeAI's own template module.

Modernised from the lab's hand-rolled SimplePromptTemplate class,
which reimplements (badly) what beeai_framework.template provides.
"""

import asyncio
import os

from beeai_framework.backend import ChatModel, UserMessage
from beeai_framework.template import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel

if not load_dotenv():
    raise RuntimeError(".env not found - check cwd or run from lab root")
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY missing from .env")


class ProjectScenario(BaseModel):
    project_name: str
    business_problem: str
    data_description: str
    timeline: str
    success_metrics: str


PROJECT_EVAL_TEMPLATE = PromptTemplate(
    schema=ProjectScenario,
    template="""You are a senior data scientist evaluating a machine learning project proposal.

Project Details:
- Project Name: {{project_name}}
- Business Problem: {{business_problem}}
- Available Data: {{data_description}}
- Timeline: {{timeline}}
- Success Metrics: {{success_metrics}}

Please provide:
1. Feasibility assessment (1-10 scale)
2. Key technical challenges
3. Recommended approach
4. Risk mitigation strategies
5. Expected outcomes

Be specific and actionable in your recommendations.""",
)


SCENARIOS = [
    ProjectScenario(
        project_name="Smart Inventory Optimization",
        business_problem="Reduce inventory costs while maintaining 95% product availability",
        data_description="2 years of sales data, supplier lead times, seasonal patterns," \
        "500K records",
        timeline="3 months development, 1 month testing",
        success_metrics="15% cost reduction, maintain 95% availability, <2% forecast error",
    ),
    ProjectScenario(
        project_name="Fraud Detection System",
        business_problem="Detect fraudulent transactions in real-time with minimal false positives",
        data_description="1M transaction records, user behavior data, device fingerprints",
        timeline="6 months development, 2 months validation",
        success_metrics="95% fraud detection rate, <1% false positive rate, <100ms response time",
    ),
]


async def prompt_template_example() -> None:
    llm = ChatModel.from_name("anthropic:claude-haiku-4-5")

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n=== Project Evaluation {i}: {scenario.project_name} ===")

        rendered_prompt = PROJECT_EVAL_TEMPLATE.render(scenario)
        print("\nRendered prompt:")
        print(rendered_prompt)

        result = await llm.run([UserMessage(content=rendered_prompt)])

        print("\n### LLM response: ###\n")
        print(result.last_message.text)


if __name__ == "__main__":
    asyncio.run(prompt_template_example())
