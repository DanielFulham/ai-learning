"""Notebook cell 25: structured output.

Classic (what the notebook ships):
    llm_config = {"config_list": [...], "response_format": TicketSummary}
    support_agent.initiate_chat(recipient=support_agent, ...)   # talks to ITSELF

Two problems with the shipped cell, independent of the version fork:
  1. The agent is its own recipient. That is not a meaningful two-agent
     exchange; it is a single completion wearing a conversation costume.
  2. AG2 0.12.1's release notes record REMOVING JSON-mode structured-output
     tests for Anthropic. The notebook's exact combination (response_format
     + Anthropic) is the one the maintainers stopped testing on Classic.

v1.0 (here):
    Agent(..., response_schema=TicketSummary)
    reply = await agent.ask(...)
    summary = await reply.content()        # validated instance, or None

``AgentReply.content()`` validates against the schema and takes a
``retries`` argument: on ValidationError it feeds the error back to the
model and re-asks, then raises once the budget is spent. The retry is
visible rather than swallowed.

PROBE: ``content()`` returns ``TResult | None``. Determine empirically
whether a schema the model cannot satisfy yields None, raises
ValidationError, or returns an instance with fields elided. The assert
below deliberately fails loudly on None so the None path is observable.
"""

from __future__ import annotations

import asyncio

from ag2 import Agent
from pydantic import BaseModel

from helpers.config import build_config
from helpers.metrics import print_usage

TICKET = (
    "Ticket: John Doe is unable to reset his password and has an important meeting in 30 minutes."
)


class TicketSummary(BaseModel):
    customer_name: str
    issue_type: str
    urgency_level: str
    recommended_action: str


async def main() -> None:
    agent = Agent(
        "support_agent",
        (
            "You are a support assistant. Summarise a customer ticket into the required fields. "
            "urgency_level must be one of Low, Medium, High."
        ),
        config=build_config(),
        response_schema=TicketSummary,
    )

    reply = await agent.ask(TICKET)
    summary = await reply.content(retries=1)

    assert summary is not None, (
        "response_schema produced None - see the probe note in the module docstring"
    )
    assert isinstance(summary, TicketSummary), f"expected TicketSummary, got {type(summary)!r}"

    print("\n[structured output]")
    print(f"  customer_name      : {summary.customer_name}")
    print(f"  issue_type         : {summary.issue_type}")
    print(f"  urgency_level      : {summary.urgency_level}")
    print(f"  recommended_action : {summary.recommended_action}")

    print_usage(await reply.usage(), label="structured_output")


if __name__ == "__main__":
    asyncio.run(main())
