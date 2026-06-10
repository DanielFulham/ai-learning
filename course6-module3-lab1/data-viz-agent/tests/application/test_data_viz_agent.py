"""Tests for application.data_viz_agent.

DataVizAgent wraps create_agent and exposes run() and run_with_trace().
Tests verify:
- The constructor calls create_agent with the right arguments
- run() returns the final answer string
- run_with_trace() returns a structured AgentTrace
- Tool calls and tool results are correctly paired by tool_call_id
- Multiple AIMessages with tool calls are all captured
- Error paths raise informative exceptions

The LLM and create_agent are both patched so no API calls happen.
"""

from unittest.mock import MagicMock, patch

from matplotlib.pylab import trace
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from application.data_viz_agent import DataVizAgent
from domain.agent_trace import AgentTrace


@pytest.fixture
def fake_llm() -> MagicMock:
    """A mock LLM. Never called directly — create_agent receives it and
    drives all interactions through the compiled graph."""
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def fake_tool() -> MagicMock:
    """A mock tool. Same reasoning: the agent passes it to create_agent
    but never invokes it directly."""
    return MagicMock(spec=BaseTool)


# --- Constructor ---


def test_constructor_calls_create_agent_once(fake_llm: MagicMock, fake_tool: MagicMock) -> None:
    with patch("application.data_viz_agent.create_agent") as mock_create:
        DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test prompt")
        mock_create.assert_called_once_with(
            model=fake_llm,
            tools=[fake_tool],
            system_prompt="test prompt",
        )


def test_constructor_stores_compiled_agent(fake_llm: MagicMock, fake_tool: MagicMock) -> None:
    """The compiled graph from create_agent is held as _agent for invocation."""
    compiled = MagicMock()
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        assert agent._agent is compiled


# --- run() ---


def test_run_returns_final_answer_string(fake_llm: MagicMock, fake_tool: MagicMock) -> None:
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [
            HumanMessage(content="how many rows?"),
            AIMessage(content="There are 395 rows."),
        ]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        answer = agent.run("how many rows?")

    assert answer == "There are 395 rows."


def test_run_passes_query_as_user_message(fake_llm: MagicMock, fake_tool: MagicMock) -> None:
    """The query becomes a user message in the invocation payload."""
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [HumanMessage(content="anything"), AIMessage(content="response")]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        agent.run("my specific question")

    compiled.invoke.assert_called_once_with(
        {"messages": [{"role": "user", "content": "my specific question"}]}
    )


# --- run_with_trace() ---


def test_run_with_trace_returns_agent_trace(fake_llm: MagicMock, fake_tool: MagicMock) -> None:
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [
            HumanMessage(content="query"),
            AIMessage(content="answer"),
        ]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        trace = agent.run_with_trace("query")

    assert isinstance(trace, AgentTrace)
    assert trace.query == "query"
    assert trace.final_answer == "answer"
    assert trace.tool_calls == ()
    

def test_run_with_trace_captures_single_tool_call(
    fake_llm: MagicMock, fake_tool: MagicMock
) -> None:
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [
            HumanMessage(content="how many rows?"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "python_repl", "args": {"code": "len(df)"}, "id": "call_1"}
                ],
            ),
            ToolMessage(content="395", tool_call_id="call_1"),
            AIMessage(content="There are 395 rows."),
        ]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        trace = agent.run_with_trace("how many rows?")

    assert len(trace.tool_calls) == 1
    call = trace.tool_calls[0]
    assert call.name == "python_repl"
    assert call.args == {"code": "len(df)"}
    assert call.result == "395"


def test_run_with_trace_pairs_calls_by_id(fake_llm: MagicMock, fake_tool: MagicMock) -> None:
    """Tool results can arrive in any order relative to their calls — pairing
    must be by tool_call_id, not by position."""
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [
            HumanMessage(content="query"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tool_a", "args": {"x": 1}, "id": "call_A"},
                    {"name": "tool_b", "args": {"y": 2}, "id": "call_B"},
                ],
            ),
            # Results returned in REVERSE order (B before A).
            ToolMessage(content="result_B", tool_call_id="call_B"),
            ToolMessage(content="result_A", tool_call_id="call_A"),
            AIMessage(content="done"),
        ]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        trace = agent.run_with_trace("query")

    assert len(trace.tool_calls) == 2
    call_a = next(c for c in trace.tool_calls if c.name == "tool_a")
    call_b = next(c for c in trace.tool_calls if c.name == "tool_b")
    assert call_a.result == "result_A"
    assert call_b.result == "result_B"


def test_run_with_trace_captures_calls_across_multiple_ai_messages(
    fake_llm: MagicMock, fake_tool: MagicMock
) -> None:
    """Multi-step agent runs produce multiple AIMessages, each with their own
    tool calls. The trace must include all of them — this is the regression
    guard for the early-return bug where `return records` was inside the loop."""
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [
            HumanMessage(content="multi-step query"),
            AIMessage(
                content="",
                tool_calls=[{"name": "step_1", "args": {}, "id": "id_1"}],
            ),
            ToolMessage(content="result_1", tool_call_id="id_1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "step_2", "args": {}, "id": "id_2"}],
            ),
            ToolMessage(content="result_2", tool_call_id="id_2"),
            AIMessage(content="final answer"),
        ]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        trace = agent.run_with_trace("multi-step query")

    assert len(trace.tool_calls) == 2
    assert [c.name for c in trace.tool_calls] == ["step_1", "step_2"]
    assert [c.result for c in trace.tool_calls] == ["result_1", "result_2"]


def test_run_uses_run_with_trace_internally(fake_llm: MagicMock, fake_tool: MagicMock) -> None:
    """run() is a thin wrapper around run_with_trace() that returns only the
    final answer. This pins the delegation so they can't drift apart."""
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [HumanMessage(content="q"), AIMessage(content="a")]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        run_answer = agent.run("q")
        trace_answer = agent.run_with_trace("q").final_answer

    assert run_answer == trace_answer


# --- Error paths ---


def test_raises_when_final_message_is_not_aimessage(
    fake_llm: MagicMock, fake_tool: MagicMock
) -> None:
    """Framework contract violation: every agent run ends with an AIMessage.
    If it doesn't, we raise loudly rather than silently returning garbage."""
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [
            HumanMessage(content="query"),
            ToolMessage(content="orphan", tool_call_id="call_1"),
        ]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        with pytest.raises(RuntimeError, match="Expected final message to be AIMessage"):
            agent.run("query")


def test_raises_when_final_content_is_not_str(
    fake_llm: MagicMock, fake_tool: MagicMock
) -> None:
    """AIMessage.content can be str | list (vision API). We don't support
    list content; if we see it, raise rather than coerce."""
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [
            HumanMessage(content="query"),
            AIMessage(content=[{"type": "text", "text": "hi"}]),
        ]
    }
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        with pytest.raises(RuntimeError, match="Expected final message content to be str"):
            agent.run("query")

def test_raises_on_empty_message_list(
    fake_llm: MagicMock, fake_tool: MagicMock
) -> None:
    """If the agent ever returns an empty message list (framework bug or
    aggressive timeout), surface a clear RuntimeError rather than the
    IndexError that messages[-1] would produce."""
    compiled = MagicMock()
    compiled.invoke.return_value = {"messages": []}
    with patch("application.data_viz_agent.create_agent", return_value=compiled):
        agent = DataVizAgent(llm=fake_llm, tools=[fake_tool], system_prompt="test")
        with pytest.raises(RuntimeError, match="no messages"):
            agent.run("query")