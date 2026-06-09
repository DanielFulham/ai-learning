import json
from unittest.mock import MagicMock

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
import pytest

from application.manual_loop_agent import ManualLoopAgent


def _make_tool(name: str, return_value: str):
    @tool
    def _t(arg: str) -> str:
        """Test tool."""
        return f"{name}({arg})={return_value}"

    _t.name = name  # type: ignore[misc]
    return _t


def _make_llm(*responses: AIMessage) -> MagicMock:
    """Build a mocked LLM whose bind_tools().invoke() returns the given AIMessages in order."""
    bound = MagicMock()
    bound.invoke.side_effect = list(responses)
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound
    return llm


def test_returns_final_content_when_no_tool_calls() -> None:
    llm = _make_llm(AIMessage(content="direct answer"))
    agent = ManualLoopAgent(llm, [_make_tool("noop", "result")])

    result = agent.run("hello")

    assert result == "direct answer"


def test_dispatches_single_tool_call_then_summarises() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "extract_id", "args": {"arg": "url"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="final summary"),
    )
    agent = ManualLoopAgent(llm, [_make_tool("extract_id", "abc123")])

    result = agent.run("get the id")

    assert result == "final summary"


def test_dispatches_two_sequential_tool_calls() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "step1", "args": {"arg": "input"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "step2", "args": {"arg": "input"}, "id": "call_2", "type": "tool_call"}],
        ),
        AIMessage(content="done"),
    )
    agent = ManualLoopAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    result = agent.run("multi-step query")

    assert result == "done"


def test_dispatches_parallel_tool_calls_in_one_turn() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "step1", "args": {"arg": "input"}, "id": "call_1", "type": "tool_call"},
                {"name": "step2", "args": {"arg": "input"}, "id": "call_2", "type": "tool_call"},
            ],
        ),
        AIMessage(content="parallel done"),
    )
    agent = ManualLoopAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    result = agent.run("do both")

    assert result == "parallel done"


def test_tool_exception_returns_error_message_to_llm() -> None:
    failing_tool = _make_tool("failing", "irrelevant")
    failing_tool.func = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[attr-defined]

    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "failing", "args": {"arg": "input"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="recovered"),
    )
    agent = ManualLoopAgent(llm, [failing_tool])

    result = agent.run("query")

    assert result == "recovered"

def test_raises_runtime_error_when_max_iterations_exceeded() -> None:
    """A model that never stops calling tools must trip the cap, not exhaust resources."""
    # Build N+1 responses, all with tool_calls — the cap must fire before any final answer.
    infinite_tool_calls = [
        AIMessage(
            content="",
            tool_calls=[{"name": "loop", "args": {"arg": "x"}, "id": f"call_{i}", "type": "tool_call"}],
        )
        for i in range(50)
    ]
    llm = _make_llm(*infinite_tool_calls)
    agent = ManualLoopAgent(llm, [_make_tool("loop", "x")], max_iterations=3)

    with pytest.raises(RuntimeError, match="max_iterations"):
        agent.run("loop forever")


def test_custom_max_iterations_respected() -> None:
    """Constructor param overrides default."""
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "step", "args": {"arg": "x"}, "id": f"call_{i}", "type": "tool_call"}],
        )
        for i in range(10)
    ]
    llm = _make_llm(*responses)
    agent = ManualLoopAgent(llm, [_make_tool("step", "x")], max_iterations=2)

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
    agent = ManualLoopAgent(llm, [_make_tool("real_tool", "x")])

    result = agent.run("query")

    assert result == "recovered"

def test_raises_type_error_when_final_content_is_not_string() -> None:
    """Multimodal-style content blocks must not silently leak through the str return type."""
    llm = _make_llm(
        AIMessage(content=[{"type": "text", "text": "block-formatted response"}]),  # type: ignore[arg-type]
    )
    agent = ManualLoopAgent(llm, [_make_tool("noop", "x")])

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
    
    agent = ManualLoopAgent(llm, [dict_returning_tool])
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
    agent = ManualLoopAgent(llm, [_make_tool("extract_video_id", "abc123")])
    
    with pytest.raises(ValueError, match="has no id"):
        agent.run("query")


def test_logs_exception_when_tool_raises(caplog) -> None:
    """Tool exceptions must be logged with stack trace, not just swallowed into a string."""
    import logging

    failing_tool = _make_tool("failing", "irrelevant")
    failing_tool.func = MagicMock(side_effect=RuntimeError("infra exploded"))  # type: ignore[attr-defined]

    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "failing", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="recovered"),
    )
    agent = ManualLoopAgent(llm, [failing_tool])

    with caplog.at_level(logging.ERROR):
        agent.run("query")

    assert any("failing" in record.message for record in caplog.records)
    assert any("infra exploded" in str(record.exc_info[1]) for record in caplog.records if record.exc_info)