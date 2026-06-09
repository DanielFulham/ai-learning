import json

import pytest
from unittest.mock import MagicMock

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.recursive_agent import RecursiveAgent


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


def test_returns_final_content_when_no_tool_calls() -> None:
    llm = _make_llm(AIMessage(content="direct answer"))
    agent = RecursiveAgent(llm, [_make_tool("noop", "result")])

    assert agent.run("hello") == "direct answer"


def test_recurses_for_two_sequential_tool_calls() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "step1", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "step2", "args": {"arg": "in"}, "id": "call_2", "type": "tool_call"}],
        ),
        AIMessage(content="done"),
    )
    agent = RecursiveAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    assert agent.run("multi-step") == "done"


def test_recurses_for_n_sequential_tool_calls() -> None:
    llm = _make_llm(
        *[
            AIMessage(
                content="",
                tool_calls=[{"name": "step", "args": {"arg": "in"}, "id": f"call_{i}", "type": "tool_call"}],
            )
            for i in range(5)
        ],
        AIMessage(content="five-step done"),
    )
    agent = RecursiveAgent(llm, [_make_tool("step", "x")])

    assert agent.run("query") == "five-step done"


def test_parallel_tool_calls_in_one_turn() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "a", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"},
                {"name": "b", "args": {"arg": "in"}, "id": "call_2", "type": "tool_call"},
            ],
        ),
        AIMessage(content="parallel done"),
    )
    agent = RecursiveAgent(llm, [_make_tool("a", "x"), _make_tool("b", "y")])

    assert agent.run("do both") == "parallel done"


def test_raises_runtime_error_when_max_iterations_exceeded() -> None:
    """A model that never stops calling tools must trip the cap before recursion blows the stack."""
    infinite_tool_calls = [
        AIMessage(
            content="",
            tool_calls=[{"name": "loop", "args": {"arg": "x"}, "id": f"call_{i}", "type": "tool_call"}],
        )
        for i in range(50)
    ]
    llm = _make_llm(*infinite_tool_calls)
    agent = RecursiveAgent(llm, [_make_tool("loop", "x")], max_iterations=3)

    with pytest.raises(RuntimeError, match="max_iterations"):
        agent.run("loop forever")


def test_custom_max_iterations_respected() -> None:
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "step", "args": {"arg": "x"}, "id": f"call_{i}", "type": "tool_call"}],
        )
        for i in range(10)
    ]
    llm = _make_llm(*responses)
    agent = RecursiveAgent(llm, [_make_tool("step", "x")], max_iterations=2)

    with pytest.raises(RuntimeError, match="max_iterations \\(2\\)"):
        agent.run("query")

def test_unknown_tool_name_returns_structured_error_to_llm() -> None:
    """LLM hallucinated tool name should produce a clear error, not a KeyError."""
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="recovered"),
    )

    agent = RecursiveAgent(llm, [_make_tool("a", "x"), _make_tool("b", "y")])

    result = agent.run("query")

    assert result == "recovered"

def test_raises_type_error_when_final_content_is_not_string() -> None:
    """Multimodal-style content blocks must not silently leak through the str return type."""
    llm = _make_llm(
        AIMessage(content=[{"type": "text", "text": "block-formatted response"}]),  # type: ignore[arg-type]
    )
    agent = RecursiveAgent(llm, [_make_tool("a", "x")])

    with pytest.raises(TypeError, match="Expected string content"):
        agent.run("query")

def test_tool_returning_dict_is_json_serialised_for_llm() -> None:
    """Dict tool results must be JSON, not Python repr — booleans, None, and unicode must round-trip."""
    
    @tool
    def dict_returning_tool(arg: str) -> dict:
        """Test tool returning a dict with edge-case values."""
        return {"title": "naïve résumé", "views": None, "is_live": False, "count": 100}
    
    dict_returning_tool.name = "dict_tool"  # type: ignore[misc]
    
    captured_tool_messages = []
    
    bound = MagicMock()
    
    def fake_invoke(messages):
        # Capture the ToolMessage content the LLM would see
        for msg in messages:
            if hasattr(msg, "tool_call_id") and msg.tool_call_id == "call_1":
                captured_tool_messages.append(msg.content)
                # Stop the loop on the second invoke
                return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[{"name": "dict_tool", "args": {"arg": "x"}, "id": "call_1", "type": "tool_call"}],
        )
    
    bound.invoke.side_effect = fake_invoke
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound
    
    agent = RecursiveAgent(llm, [dict_returning_tool])
    agent.run("test")
    
    # The content the LLM saw must be parseable JSON, not Python repr
    assert len(captured_tool_messages) == 1
    parsed = json.loads(captured_tool_messages[0])
    assert parsed == {"title": "naïve résumé", "views": None, "is_live": False, "count": 100}

def test_raises_value_error_when_tool_call_has_no_id() -> None:
    """Malformed tool_call without id must raise a clear error, not a Pydantic stack trace."""
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "extract_video_id", "args": {"url": "x"}, "id": None, "type": "tool_call"}],  # type: ignore[typeddict-item]
        ),
    )
    agent = RecursiveAgent(llm, [_make_tool("extract_video_id", "id")])
    
    with pytest.raises(ValueError, match="has no id"):
        agent.run("query")

import logging


def test_logs_exception_when_tool_raises(caplog) -> None:
    """Tool exceptions must be logged with stack trace, not just swallowed into a string."""
    failing_tool = _make_tool("failing", "irrelevant")
    failing_tool.func = MagicMock(side_effect=RuntimeError("infra exploded"))  # type: ignore[attr-defined]

    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "failing", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="recovered"),
    )
    agent = RecursiveAgent(llm, [failing_tool])

    with caplog.at_level(logging.ERROR):
        agent.run("query")

    assert any("failing" in record.message for record in caplog.records)
    assert any("infra exploded" in str(record.exc_info[1]) for record in caplog.records if record.exc_info)