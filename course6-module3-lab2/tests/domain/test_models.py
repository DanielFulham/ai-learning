from dataclasses import FrozenInstanceError

import pytest

from domain.models import AgentTrace, ToolCallRecord


def _make_record(
    tool_call_id: str = "call_abc",
    tool_name: str = "sql_db_query",
    args: dict | None = None,
    result: str = "[(347,)]",
    duration_ms: float = 1.5,
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        args=args if args is not None else {"query": "SELECT COUNT(*) FROM Album"},
        result=result,
        duration_ms=duration_ms,
    )


def test_tool_call_record_is_frozen():
    record = _make_record()
    with pytest.raises(FrozenInstanceError):
        setattr(record, "tool_name", "something_else")


def test_tool_call_record_equality_is_by_value():
    a = _make_record(tool_call_id="call_1")
    b = _make_record(tool_call_id="call_1")
    assert a == b


def test_tool_call_record_inequality_when_any_field_differs():
    a = _make_record(tool_call_id="call_1")
    b = _make_record(tool_call_id="call_2")
    assert a != b


def test_agent_trace_starts_empty():
    trace = AgentTrace()
    assert len(trace) == 0
    assert trace.tool_calls == ()


def test_agent_trace_append_records_in_order():
    trace = AgentTrace()
    first = _make_record(tool_call_id="call_1")
    second = _make_record(tool_call_id="call_2")

    trace.append(first)
    trace.append(second)

    assert trace.tool_calls == (first, second)


def test_agent_trace_len_matches_append_count():
    trace = AgentTrace()
    for i in range(3):
        trace.append(_make_record(tool_call_id=f"call_{i}"))
    assert len(trace) == 3


def test_agent_trace_tool_calls_returns_tuple_not_list():
    trace = AgentTrace()
    trace.append(_make_record())
    assert isinstance(trace.tool_calls, tuple)


def test_agent_trace_tool_calls_property_does_not_leak_mutable_internals():
    trace = AgentTrace()
    trace.append(_make_record(tool_call_id="call_1"))

    snapshot = trace.tool_calls
    trace.append(_make_record(tool_call_id="call_2"))

    assert len(snapshot) == 1
    assert snapshot[0].tool_call_id == "call_1"