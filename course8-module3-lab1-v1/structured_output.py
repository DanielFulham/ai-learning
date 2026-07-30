"""Structured output via BeeAI ChatModel.run(response_format=...)."""

import asyncio
import os

from beeai_framework.backend import (
    ChatModel,
    SystemMessage,
    UserMessage,
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field

if not load_dotenv():
    raise RuntimeError(".env not found - check cwd or run from lab root")
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY missing from .env")


class BusinessPlan(BaseModel):
    """A comprehensive business plan structure."""

    business_name: str = Field(description="Catchy name for the business")
    elevator_pitch: str = Field(description="30-second description of the business")
    target_market: str = Field(description="Primary target audience")
    unique_value_proposition: str = Field(description="What makes this business special")
    revenue_streams: list[str] = Field(description="Ways the business will make money")
    startup_costs: str = Field(description="Estimated initial investment needed")
    key_success_factors: list[str] = Field(description="Critical elements for success")


async def structured_output_example() -> None:
    llm = ChatModel.from_name("anthropic:claude-haiku-4-5")

    prompt = (
        "Create a business plan for a mobile app that helps people find "
        "and book unique local experiences in their city."
    )
    messages = [
        SystemMessage(content="You are an expert business consultant and entrepreneur."),
        UserMessage(content=prompt),
    ]

    result = await llm.run(messages, response_format=BusinessPlan)

    assert isinstance(result.output_structured, BusinessPlan)
    plan = result.output_structured

    print(f"User: {prompt}\n")
    print("AI-Generated Business Plan:")
    print(f"Business Name: {plan.business_name}")
    print(f"Elevator Pitch: {plan.elevator_pitch}")
    print(f"Target Market: {plan.target_market}")
    print(f"Unique Value Proposition: {plan.unique_value_proposition}")
    print(f"Revenue Streams: {', '.join(plan.revenue_streams)}")
    print(f"Startup Costs: {plan.startup_costs}")
    print("Key Success Factors:")
    for factor in plan.key_success_factors:
        print(f"  - {factor}")


if __name__ == "__main__":
    asyncio.run(structured_output_example())
