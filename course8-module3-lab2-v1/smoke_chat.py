"""Notebook cell 10: two-agent student/tutor exchange.

Classic (what the notebook ships):
    student = ConversableAgent(name=..., system_message=..., llm_config={...})
    chat_result = student.initiate_chat(recipient=tutor, message=..., max_turns=2,
                                        summary_method="reflection_with_llm")
    chat_result.summary

v1.0 (here):
    Agent(name, prompt, config=AnthropicConfig(...))
    reply = await tutor.ask(message)            # returns AgentReply
    reply.body                                  # text of this turn
    await reply.usage()                         # UsageReport over the stream

Direction matters and does not map by position. ``initiate_chat`` names the
SENDER (``student.initiate_chat(recipient=tutor, ...)`` = student speaks);
``ask`` names the RESPONDENT (``tutor.ask(...)`` = tutor answers). Mapping
one onto the other positionally inverts the roles and still produces
plausible output - see the first-run correction in the lab notes.

``stream=`` is passed explicitly so all three turns share one conversation
log. Without it each ``ask`` opens its own stream and ``usage()`` reports
only the last one.

There is no ``summary_method`` equivalent in v1.0. Classic's
``reflection_with_llm`` ran an extra LLM call to summarise the exchange;
here any summarisation would be an explicit ``ask``, so the cost is
visible rather than hidden behind a kwarg.
"""

from __future__ import annotations

import asyncio

from ag2 import Agent

from helpers.config import build_config
from helpers.metrics import print_usage

QUESTION = "Can you explain what a neural network is?"

TUTOR_PROMPT = (
    "You are a helpful tutor who provides clear and concise explanations "
    "suitable for a beginner."
)

STUDENT_PROMPT = (
    "You are a curious student. You ask clear, specific questions to learn "
    "new concepts."
)


async def main() -> None:
    config = build_config()

    tutor = Agent("tutor", TUTOR_PROMPT, config=config)
    student = Agent("student", STUDENT_PROMPT, config=config)

    # Turn 1: the tutor is asked the question. The notebook passes QUESTION
    # as a literal under the student's name; the student's model does not
    # speak until turn 2.
    answer = await tutor.ask(QUESTION)
    answer_text = answer.body
    assert answer_text is not None, "tutor produced no text body"
    print(f"\n[tutor]\n{answer_text}")

    # Turn 2: the student reads the tutor's answer and asks a follow-up.
    # Its own stream, so the tutor's turn arrives as a user message rather
    # than as the student's own prior output.
    followup = await student.ask(answer_text)
    followup_text = followup.body
    assert followup_text is not None, "student produced no text body"
    print(f"\n[student]\n{followup_text}")

    # Turn 3: back to the tutor, continuing ITS chain via reply.ask so the
    # tutor keeps its own context.
    final = await answer.ask(followup_text)
    final_text = final.body
    assert final_text is not None, "tutor produced no text body"
    print(f"\n[tutor]\n{final_text}")

    # Usage is per-stream. Two streams, two reports, summed by hand.
    tutor_usage = await final.usage()
    student_usage = await followup.usage()
    print_usage(tutor_usage, label="tutor stream")
    print_usage(student_usage, label="student stream")


if __name__ == "__main__":
    asyncio.run(main())
