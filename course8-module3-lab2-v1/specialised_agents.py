"""Notebook cell 13: specialised agents.

Classic (what the notebook ships):
    tech_expert = ConversableAgent(name=..., system_message=..., llm_config={...})
    creative_writer = ConversableAgent(...)
    business_analyst = ConversableAgent(...)
    for agent in agents:
        print(f"- {agent.name}: {agent.system_message.split('.')[0]}.")

The cell constructs three agents and never runs them. It prints names and
the first sentence of each system message, which demonstrates only that a
constructor accepts a string.

v1.0 (here): construct the three, then ask all three the SAME question and
print the answers side by side, with per-agent usage.

Deliberate deviation from the notebook. The three roles differ ONLY by
their system prompt, so an identical question isolates the prompt as the
variable and makes two things measurable that the original cell asserts
without evidence:

  1. Prompt shape drives response shape. Same input, three different
     answers.
  2. System-prompt verbosity is a per-call cost. Each agent's prompt-token
     count includes its own system message on every turn, so the wordiest
     role is the most expensive role forever. The notebook's prompts are
     multi-line and include filler like "Always consider best practices
     and performance implications" - carried over verbatim so the cost of
     that verbosity is visible rather than edited away.
"""

from __future__ import annotations

import asyncio

from ag2 import Agent
from ag2.config.anthropic import AnthropicConfig
from ag2.usage import UsageReport

from helpers.config import build_config
from helpers.metrics import print_usage

QUESTION = "Should a small team adopt a microservices architecture?"

TECH_EXPERT_PROMPT = """You are a senior software engineer with expertise in Python, AI, and
system design.
Provide technical, detailed explanations with code examples when appropriate.
Always consider best practices and performance implications."""

CREATIVE_WRITER_PROMPT = """You are a creative writer and storyteller.
Your responses are engaging, imaginative, and use vivid descriptions.
You excel at making complex topics accessible through stories and analogies."""

BUSINESS_ANALYST_PROMPT = """You are a business analyst focused on ROI, efficiency, and
strategic planning.
Always consider business impact, costs, and practical implementation.
Provide actionable recommendations with clear metrics."""

ROLES: tuple[tuple[str, str], ...] = (
    ("tech_expert", TECH_EXPERT_PROMPT),
    ("creative_writer", CREATIVE_WRITER_PROMPT),
    ("business_analyst", BUSINESS_ANALYST_PROMPT),
)


async def ask_role(name: str, prompt: str, config: AnthropicConfig) -> tuple[str, UsageReport]:
    """Put QUESTION to one role and return its answer and usage."""
    agent = Agent(name, prompt, config=config)
    reply = await agent.ask(QUESTION)
    body = reply.body
    assert body is not None, f"{name} produced no text body"
    return body, await reply.usage()


def print_role_answer(name: str, body: str) -> None:
    print(f"\n{'=' * 70}\n[{name}]\n{'=' * 70}\n{body}")


async def main() -> None:
    config = build_config()
    print(f"Question put to all three roles:\n  {QUESTION}")

    for name, prompt in ROLES:
        body, usage = await ask_role(name, prompt, config)
        print_role_answer(name, body)
        print_usage(usage, label=name)


if __name__ == "__main__":
    asyncio.run(main())
