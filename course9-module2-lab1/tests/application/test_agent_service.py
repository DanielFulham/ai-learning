import asyncio
from collections.abc import Callable, Sequence
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool

from application.agent_service import AgentService
from interfaces.tool_source_interface import ToolSourceInterface


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class FakeToolModel(GenericFakeChatModel):
    """A scripted chat model that accepts tools without needing a real provider."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        return self


def make_model(*replies: AIMessage) -> FakeToolModel:
    return FakeToolModel(messages=iter(replies))


def make_tool_source(tools: Sequence[BaseTool]) -> tuple[MagicMock, MagicMock]:
    """A mock tool source whose `open()` is an async context manager yielding `tools`."""
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=tools)
    context.__aexit__ = AsyncMock(return_value=None)
    source = MagicMock(spec=ToolSourceInterface)
    source.open.return_value = context
    return source, context


def make_tool_call_then_answer() -> FakeToolModel:
    return make_model(
        AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 2, "b": 1}, "id": "1"}]),
        AIMessage(content="TEST_ANSWER"),
    )


def test_ask_returns_the_conversation_with_the_tool_called() -> None:
    source, _ = make_tool_source([add])
    service = AgentService(make_tool_call_then_answer(), source)

    reply = asyncio.run(service.ask("TEST_PROMPT"))

    assert [turn.role for turn in reply.turns] == ["human", "ai", "tool", "ai"]
    assert reply.turns[0].text == "TEST_PROMPT"
    assert reply.turns[1].tool_names == ("add",)
    assert reply.turns[2].text == "3"
    assert reply.text == "TEST_ANSWER"


def test_ask_opens_and_closes_the_tool_source_once() -> None:
    source, context = make_tool_source([add])
    service = AgentService(make_tool_call_then_answer(), source)

    asyncio.run(service.ask("TEST_PROMPT"))

    source.open.assert_called_once_with()
    context.__aenter__.assert_awaited_once()
    context.__aexit__.assert_awaited_once()


def test_ask_answers_directly_when_the_model_calls_no_tool() -> None:
    source, _ = make_tool_source([add])
    service = AgentService(make_model(AIMessage(content="TEST_ANSWER")), source)

    reply = asyncio.run(service.ask("TEST_PROMPT"))

    assert [turn.role for turn in reply.turns] == ["human", "ai"]
    assert all(turn.tool_names == () for turn in reply.turns)
