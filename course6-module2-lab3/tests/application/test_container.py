from typing import cast
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel

from application.container import AgentStrategy, initialise
from application.manual_loop_agent import ManualLoopAgent
from application.recursive_agent import RecursiveAgent
from application.two_step_chain_agent import TwoStepChainAgent
from interfaces.chat_model_provider_interface import ChatModelProviderInterface


def _make_provider() -> MagicMock:
    provider = MagicMock(spec=ChatModelProviderInterface)
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = MagicMock()
    provider.create.return_value = llm
    return provider


def test_default_strategy_returns_recursive_agent() -> None:
    agent = initialise(chat_model_provider=_make_provider())
    assert isinstance(agent, RecursiveAgent)


def test_manual_loop_strategy_returns_manual_loop_agent() -> None:
    agent = initialise(strategy=AgentStrategy.MANUAL_LOOP, chat_model_provider=_make_provider())
    assert isinstance(agent, ManualLoopAgent)


def test_two_step_strategy_returns_two_step_agent() -> None:
    agent = initialise(strategy=AgentStrategy.TWO_STEP_CHAIN, chat_model_provider=_make_provider())
    assert isinstance(agent, TwoStepChainAgent)


def test_recursive_strategy_returns_recursive_agent() -> None:
    agent = initialise(strategy=AgentStrategy.RECURSIVE, chat_model_provider=_make_provider())
    assert isinstance(agent, RecursiveAgent)


def test_provider_create_called_once() -> None:
    provider = _make_provider()
    initialise(chat_model_provider=provider)
    provider.create.assert_called_once()


def test_each_call_returns_fresh_agent_instance() -> None:
    provider = _make_provider()
    agent1 = initialise(chat_model_provider=provider)
    agent2 = initialise(chat_model_provider=provider)
    assert agent1 is not agent2


def test_recursive_agent_wires_six_tools() -> None:
    agent = cast(RecursiveAgent, initialise(chat_model_provider=_make_provider()))
    assert len(agent._tools) == 6
    assert set(agent._tools.keys()) == {
        "extract_video_id",
        "fetch_transcript",
        "search_youtube",
        "get_full_metadata",
        "get_thumbnails",
        "get_trending_videos",
    }