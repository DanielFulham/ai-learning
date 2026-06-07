"""Composition root. Builds a fully-wired ToolCallingAgent.

The only file in the project allowed to import from both infra/ and application/.
Stateless — returns a fresh agent per call. Provider selection lives here so
entry points never see concrete infra types.
"""

from langchain_core.tools import BaseTool

from application.agent import ToolCallingAgent
from infra.openai_chat_model import OpenAIChatModelProvider
from interfaces.chat_model_provider_interface import ChatModelProviderInterface


def initialise(
    tools: list[BaseTool],
    provider: ChatModelProviderInterface | None = None,
) -> ToolCallingAgent:
    if provider is None:
        provider = OpenAIChatModelProvider()
    llm = provider.create()
    return ToolCallingAgent(llm, tools)