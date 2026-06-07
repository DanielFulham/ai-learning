from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel

from application.agent import ToolCallingAgent
from application.container import initialise
from application.tools.arithmetic import add
from interfaces.chat_model_provider_interface import ChatModelProviderInterface


def _make_mock_provider() -> MagicMock:
    """Build a mock provider that returns a mock chat model."""
    provider = MagicMock(spec=ChatModelProviderInterface)
    provider.create.return_value = MagicMock(spec=BaseChatModel)
    return provider


def test_initialise_returns_tool_calling_agent():
    agent = initialise(tools=[add], provider=_make_mock_provider())
    assert isinstance(agent, ToolCallingAgent)


def test_initialise_calls_provider_create():
    provider = _make_mock_provider()
    initialise(tools=[add], provider=provider)
    provider.create.assert_called_once()


def test_initialise_is_stateless():
    """Each call returns a fresh agent — no shared state between invocations."""
    provider = _make_mock_provider()
    agent_1 = initialise(tools=[add], provider=provider)
    agent_2 = initialise(tools=[add], provider=provider)
    assert agent_1 is not agent_2