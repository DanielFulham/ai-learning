"""Notebook cell 18: human-in-the-loop bug triage.

Classic (what the notebook ships):
    human = ConversableAgent(name="human", human_input_mode="ALWAYS")
    human.initiate_chat(recipient=triage_bot, message=initial_prompt)

The human is a second AGENT whose reply is whatever you type; every turn
reaching it blocks on stdin.

v1.0 (here): ONE agent plus a ``hitl_hook``. The hook is not a
participant - it answers ``HumanInputRequest`` events raised by
``ctx.input(...)`` from INSIDE a tool. Second half of the UserProxyAgent
split; see plotting_agent.py for the code-execution half.

That inversion closes L38's failure mode. BeeAI's
``AskPermissionRequirement`` announced the gate through the same channel
that enforced it, so strong models reasoned around the tool and the
approval never fired. Here ``ctx: Context`` is injected by fast_depends
and excluded from the JSON schema, so the model cannot perceive the gate
and cannot route around it.

Default with no ``hitl_hook`` is ``HumanInputNotProvidedError`` - fail
loud rather than auto-approve.
"""

from __future__ import annotations

import asyncio
import random

from ag2 import Agent, Context, tool
from ag2.events import HumanInputRequest

from helpers.config import build_config
from helpers.metrics import print_usage

BUGS = (
    "App crashes when opening user profile.",
    "Minor UI misalignment on settings page.",
    "Password reset email not sent consistently.",
    "Typo in the About Us footer text.",
    "Database connection timeout under heavy load.",
    "Login form allows SQL injection attack.",
)

TRIAGE_PROMPT = """You are a bug triage assistant. You are given bug report summaries.

For each bug, decide a priority: high, medium, or low.
Escalate ONLY the high-priority ones using the escalate_bug tool - crashes,
security holes, and data loss. Do not escalate cosmetic issues.

When you have processed every bug, summarise what was escalated, what was
declined by the reviewer, and what you left unescalated."""

APPROVALS = {"y", "yes"}


async def console_hitl(request: HumanInputRequest) -> str:
    """Answer a HumanInputRequest from the terminal.

    ``input()`` blocks the event loop, so it goes to a worker thread. The
    hook may be sync or async; async plus ``to_thread`` is the only shape
    that does not stall every other coroutine while a human thinks.
    """
    return await asyncio.to_thread(input, f"\n  {request.content}\n  > ")


@tool
async def escalate_bug(summary: str, reason: str, ctx: Context) -> str:
    """Escalate a bug to the on-call engineer. Requires human approval.

    Args:
        summary: The bug report being escalated.
        reason: Why this bug warrants escalation.
    """
    answer = await ctx.input(f"Escalate '{summary}'?\n  Reason: {reason}\n  Approve? [y/n]")
    if answer.strip().lower() not in APPROVALS:
        return f"DECLINED by reviewer, not escalated: {summary}"
    return f"ESCALATED to on-call: {summary}"


async def main() -> None:
    selected = random.sample(BUGS, 3)
    listing = "\n".join(f"{i}. {bug}" for i, bug in enumerate(selected, start=1))

    agent = Agent(
        "triage_bot",
        TRIAGE_PROMPT,
        config=build_config(),
        tools=[escalate_bug],
        hitl_hook=console_hitl,
    )

    print(f"Triaging:\n{listing}")

    reply = await agent.ask(f"Please triage the following bug reports:\n\n{listing}")
    body = reply.body
    assert body is not None, "triage_bot produced no text body"
    print(f"\n[triage_bot]\n{body}")

    print_usage(await reply.usage(), label="hitl_agent")


if __name__ == "__main__":
    asyncio.run(main())
