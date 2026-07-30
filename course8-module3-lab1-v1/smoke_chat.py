"""Smoke test: BeeAI ChatModel via LiteLLM, Anthropic Haiku 4.5."""

import asyncio
import os

from beeai_framework.backend import (
    ChatModel,
    ChatModelParameters,
    SystemMessage,
    UserMessage,
)
from dotenv import load_dotenv

if not load_dotenv():
    raise RuntimeError(".env not found - check cwd or run from lab root")
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY missing from .env")


async def basic_chat_example() -> None:
    llm = ChatModel.from_name(
        "anthropic:claude-haiku-4-5",
        ChatModelParameters(temperature=0),
    )

    prompt = (
        "Help me brainstorm a unique business idea for a food "
        "delivery service that doesn't exist yet."
    )
    messages = [
        SystemMessage(content="You are a helpful AI assistant and creative writing expert."),
        UserMessage(content=prompt),
    ]

    result = await llm.run(messages)

    print(f"User: {prompt}")
    print(f"Assistant: {result.last_message.text}")


if __name__ == "__main__":
    asyncio.run(basic_chat_example())
