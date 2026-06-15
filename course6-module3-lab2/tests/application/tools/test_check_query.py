from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from application.tools.check_query import make_check_query


def _make_llm_returning(content: str) -> MagicMock:
    llm = MagicMock(spec=BaseChatModel)
    llm.invoke.return_value = AIMessage(content=content)
    return llm


def test_make_check_query_returns_a_base_tool():
    llm = _make_llm_returning("SELECT 1")
    tool = make_check_query(llm)
    assert isinstance(tool, BaseTool)


def test_tool_is_named_sql_db_query_checker():
    llm = _make_llm_returning("SELECT 1")
    tool = make_check_query(llm)
    assert tool.name == "sql_db_query_checker"


def test_tool_description_matches_canonical_toolkit():
    llm = _make_llm_returning("SELECT 1")
    tool = make_check_query(llm)
    assert tool.description == (
        "Use this tool to double check if your query is correct before "
        "executing it. Always use this tool before executing a query with "
        "sql_db_query!"
    )


def test_invocation_returns_content_from_llm_response():
    llm = _make_llm_returning("SELECT * FROM Album LIMIT 5;")
    tool = make_check_query(llm)

    result = tool.invoke({"query": "SELECT * FROM Album LIMIT 5;"})

    assert result == "SELECT * FROM Album LIMIT 5;"


def test_invocation_sends_system_then_human_message_to_llm():
    llm = _make_llm_returning("SELECT 1")
    tool = make_check_query(llm)

    tool.invoke({"query": "SELECT * FROM Artist"})

    llm.invoke.assert_called_once()
    messages = llm.invoke.call_args.args[0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


def test_invocation_passes_query_as_human_message_content():
    llm = _make_llm_returning("SELECT 1")
    tool = make_check_query(llm)

    tool.invoke({"query": "SELECT * FROM Artist"})

    messages = llm.invoke.call_args.args[0]
    assert messages[1].content == "SELECT * FROM Artist"


def test_invocation_uses_sql_expert_system_prompt():
    llm = _make_llm_returning("SELECT 1")
    tool = make_check_query(llm)

    tool.invoke({"query": "SELECT 1"})

    system_message = llm.invoke.call_args.args[0][0]
    assert "SQL expert" in system_message.content
    assert "Double check" in system_message.content
    assert "NOT IN with NULL" in system_message.content


def test_invocation_handles_llm_returning_a_rewritten_query():
    llm = _make_llm_returning("SELECT * FROM Album LIMIT 5")
    tool = make_check_query(llm)

    result = tool.invoke({"query": "SELECT * FROM Album"})

    assert result == "SELECT * FROM Album LIMIT 5"


def test_invocation_calls_llm_once_per_invocation():
    llm = _make_llm_returning("SELECT 1")
    tool = make_check_query(llm)

    tool.invoke({"query": "SELECT 1"})
    tool.invoke({"query": "SELECT 2"})

    assert llm.invoke.call_count == 2