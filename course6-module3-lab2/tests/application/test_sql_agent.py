from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
import pytest

from application.sql_agent import SqlAgent
from domain.models import AgentTrace


def _make_agent_invoke_returning(final_content: str) -> MagicMock:
    compiled_agent = MagicMock()
    compiled_agent.invoke.return_value = {
        "messages": [AIMessage(content=final_content)]
    }
    return compiled_agent


def test_sql_agent_stores_trace_passed_to_constructor():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)
    tools = [MagicMock(spec=BaseTool)]

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = _make_agent_invoke_returning("ignored")
        agent = SqlAgent(llm=llm, tools=tools, dialect="sqlite", trace=trace)

    assert agent.trace is trace


def test_sql_agent_constructs_create_agent_with_provided_llm_and_tools():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)
    tool_a = MagicMock(spec=BaseTool)
    tool_b = MagicMock(spec=BaseTool)

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = _make_agent_invoke_returning("ignored")
        SqlAgent(llm=llm, tools=[tool_a, tool_b], dialect="sqlite", trace=trace)

    mock_create.assert_called_once()
    call_args, call_kwargs = mock_create.call_args
    assert call_args[0] is llm
    assert call_args[1] == [tool_a, tool_b]


def test_sql_agent_formats_system_prompt_with_dialect_and_default_top_k():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = _make_agent_invoke_returning("ignored")
        SqlAgent(llm=llm, tools=[], dialect="postgresql", trace=trace)

    system_prompt = mock_create.call_args.kwargs["system_prompt"]
    assert "postgresql" in system_prompt
    assert "at most 5 results" in system_prompt


def test_sql_agent_formats_system_prompt_with_custom_top_k():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = _make_agent_invoke_returning("ignored")
        SqlAgent(llm=llm, tools=[], dialect="sqlite", trace=trace, top_k=10)

    system_prompt = mock_create.call_args.kwargs["system_prompt"]
    assert "at most 10 results" in system_prompt


def test_sql_agent_wires_logging_middleware():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = _make_agent_invoke_returning("ignored")
        SqlAgent(llm=llm, tools=[], dialect="sqlite", trace=trace)

    middleware = mock_create.call_args.kwargs["middleware"]
    assert len(middleware) == 1


def test_ask_invokes_underlying_agent_with_question_as_user_message():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)
    compiled = _make_agent_invoke_returning("There are 347 albums.")

    with patch("application.sql_agent.create_agent", return_value=compiled):
        agent = SqlAgent(llm=llm, tools=[], dialect="sqlite", trace=trace)
        agent.ask("How many albums?")

    compiled.invoke.assert_called_once_with(
        {"messages": [{"role": "user", "content": "How many albums?"}]}
    )


def test_ask_returns_final_message_content_as_string():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)
    compiled = _make_agent_invoke_returning("There are 347 albums.")

    with patch("application.sql_agent.create_agent", return_value=compiled):
        agent = SqlAgent(llm=llm, tools=[], dialect="sqlite", trace=trace)
        result = agent.ask("How many albums?")

    assert result == "There are 347 albums."


def test_ask_raises_when_message_content_is_not_str():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)
    compiled = MagicMock()
    compiled.invoke.return_value = {
        "messages": [AIMessage(content=[{"type": "text", "text": "answer"}])]
    }

    with patch("application.sql_agent.create_agent", return_value=compiled):
        agent = SqlAgent(llm=llm, tools=[], dialect="sqlite", trace=trace)
        with pytest.raises(TypeError, match="Expected str content"):
            agent.ask("How many albums?")


def test_ask_can_be_called_multiple_times_against_same_compiled_agent():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)
    compiled = _make_agent_invoke_returning("answer")

    with patch("application.sql_agent.create_agent", return_value=compiled):
        agent = SqlAgent(llm=llm, tools=[], dialect="sqlite", trace=trace)
        agent.ask("Question 1")
        agent.ask("Question 2")

    assert compiled.invoke.call_count == 2


def test_sql_agent_does_not_compile_agent_per_ask_call():
    trace = AgentTrace()
    llm = MagicMock(spec=BaseChatModel)

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = _make_agent_invoke_returning("answer")
        agent = SqlAgent(llm=llm, tools=[], dialect="sqlite", trace=trace)
        agent.ask("Question 1")
        agent.ask("Question 2")

    assert mock_create.call_count == 1