import pytest
from unittest.mock import MagicMock

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage


from application.two_step_chain_agent import TwoStepChainAgent


def _make_tool(name: str, return_value: str):
    @tool
    def _t(arg: str) -> str:
        """Test tool."""
        return f"{name}({arg})={return_value}"

    _t.name = name  # type: ignore[misc]
    return _t


def _make_llm(*responses: AIMessage) -> MagicMock:
    bound = MagicMock()
    bound.invoke.side_effect = list(responses)
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound
    return llm


def test_handles_two_step_flow() -> None:
    """The agent's happy path — exactly two tool calls then a summary."""
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "step1", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "step2", "args": {"arg": "in"}, "id": "call_2", "type": "tool_call"}],
        ),
        AIMessage(content="summary"),
    )
    agent = TwoStepChainAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    assert agent.run("two-step") == "summary"

def test_raises_type_error_when_chain_returns_non_string() -> None:
    """Chain final summary must be a string, not a list of content blocks."""
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "step1", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "step2", "args": {"arg": "in"}, "id": "call_2", "type": "tool_call"}],
        ),
        AIMessage(content=[{"type": "text", "text": "block content"}]),  # type: ignore[arg-type]
    )
    agent = TwoStepChainAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    with pytest.raises(TypeError, match="Expected string"):
        agent.run("query")

def test_runs_three_llm_invocations_even_when_first_response_has_no_tool_calls() -> None:
    """
    The chain hardcodes three LLM invocations regardless of whether they're needed.
    A query that needs zero tools still triggers two pointless extra invokes.
    This test pins that behaviour so the limitation is documented in the suite.
    """
    bound = MagicMock()
    # First response has no tool calls (model decided no tools needed).
    # Chain still makes two more invokes anyway because shape is hardcoded.
    bound.invoke.side_effect = [
        AIMessage(content="no tools needed"),
        AIMessage(content="redundant invoke 2"),
        AIMessage(content="redundant invoke 3 (this is what gets returned)"),
    ]
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound

    agent = TwoStepChainAgent(llm, [_make_tool("unused", "x")])
    result = agent.run("trivial query")

    # Three invokes happened; final content from the third one is returned.
    assert bound.invoke.call_count == 3
    assert result == "redundant invoke 3 (this is what gets returned)"


def test_empty_tool_calls_produces_empty_tool_messages_list() -> None:
    """
    When first AIMessage has tool_calls=[], the tool_messages list is empty
    but the chain proceeds anyway. This is the underlying mechanism that
    makes the previous test work — the chain doesn't crash, it just does
    nothing useful with empty tool_calls.
    """
    bound = MagicMock()
    bound.invoke.side_effect = [
        AIMessage(content="", tool_calls=[]),  # explicit empty
        AIMessage(content="", tool_calls=[]),
        AIMessage(content="summary anyway"),
    ]
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound

    agent = TwoStepChainAgent(llm, [_make_tool("unused", "x")])
    result = agent.run("query")

    assert result == "summary anyway"

def test_raises_value_error_when_tool_call_has_no_id() -> None:
    """Malformed tool_call without id must raise a clear error, not a Pydantic stack trace."""
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "extract_video_id", "args": {"url": "x"}, "id": None, "type": "tool_call"}],  # type: ignore[typeddict-item]
        ),
    )
    agent = TwoStepChainAgent(llm, [_make_tool("extract_video_id", "id123")])

    with pytest.raises(ValueError, match="has no id"):
        agent.run("query")