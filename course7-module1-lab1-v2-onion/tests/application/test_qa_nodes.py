from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.qa_nodes import context_provider_node, make_qa_node
from domain.qa_exchange import QAExchange
from domain.state_schemas import QAState


def test_context_provider_returns_state_unchanged_on_no_match() -> None:
    state: QAState = {"exchange": QAExchange(question="What is the weather today?")}
    result = context_provider_node(state)
    exchange = result.get("exchange")
    assert exchange is not None
    assert exchange.context is None


def test_context_provider_matches_langgraph_keyword() -> None:
    state: QAState = {"exchange": QAExchange(question="What is LangGraph?")}
    result = context_provider_node(state)
    exchange = result.get("exchange")
    assert exchange is not None
    assert exchange.context is not None
    assert "LangGraph" in exchange.context


def test_context_provider_matches_guided_project_keyword() -> None:
    """The V1 hallucination case — keyword-matched context returned for a
    question whose actual subject is project recommendations. V2 preserves
    the bug; the fix is an eval pipeline, not a smarter matcher."""
    state: QAState = {"exchange": QAExchange(question="What is the best guided project?")}
    result = context_provider_node(state)
    exchange = result.get("exchange")
    assert exchange is not None
    assert exchange.context is not None


def test_context_provider_raises_on_missing_exchange() -> None:
    with pytest.raises(ValueError, match="requires an exchange"):
        context_provider_node({})


def test_qa_node_returns_no_context_message_when_context_none() -> None:
    model = MagicMock(spec=BaseChatModel)
    node = make_qa_node(model)
    state: QAState = {"exchange": QAExchange(question="What is the weather today?")}
    result = node(state)
    exchange = result.get("exchange")
    assert exchange is not None
    assert exchange.answer is not None
    assert "don't have enough context" in exchange.answer
    model.invoke.assert_not_called()


def test_qa_node_invokes_model_with_prompt_when_context_present() -> None:
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = AIMessage(content="LangGraph is a state machine framework.")
    node = make_qa_node(model)
    state: QAState = {
        "exchange": QAExchange(
            question="What is LangGraph?",
            context="LangGraph is a Python library.",
        )
    }
    result = node(state)
    exchange = result.get("exchange")
    assert exchange is not None
    assert exchange.answer == "LangGraph is a state machine framework."
    model.invoke.assert_called_once()
    prompt = model.invoke.call_args[0][0]
    assert "Context: LangGraph is a Python library." in prompt
    assert "Question: What is LangGraph?" in prompt


def test_qa_node_catches_model_errors_into_answer() -> None:
    model = MagicMock(spec=BaseChatModel)
    model.invoke.side_effect = RuntimeError("connection refused")
    node = make_qa_node(model)
    state: QAState = {
        "exchange": QAExchange(
            question="What is LangGraph?",
            context="LangGraph is a Python library.",
        )
    }
    result = node(state)
    exchange = result.get("exchange")
    assert exchange is not None
    assert exchange.answer is not None
    assert "An error occurred" in exchange.answer
    assert "connection refused" in exchange.answer