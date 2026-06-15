from unittest.mock import MagicMock

from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages.tool import ToolCall
from langgraph.prebuilt.tool_node import ToolRuntime

from application.middleware import make_log_tool_call_middleware
from domain.models import AgentTrace


def _make_request(
    tool_call_id: str = "call_abc",
    tool_name: str = "sql_db_query",
    args: dict | None = None,
) -> ToolCallRequest:
    tool_call: ToolCall = {
        "id": tool_call_id,
        "name": tool_name,
        "args": args if args is not None else {"query": "SELECT 1"},
        "type": "tool_call",
    }
    return ToolCallRequest(
        tool_call=tool_call,
        tool=None,
        state={},
        runtime=MagicMock(spec=ToolRuntime),
    )

def _make_response(content: str | list = "[(1,)]"):
    response = MagicMock()
    response.content = content
    return response

def test_make_log_tool_call_middleware_returns_object_with_wrap_tool_call():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)
    assert hasattr(middleware, "wrap_tool_call")


def test_middleware_appends_one_record_per_invocation():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    handler = MagicMock(return_value=_make_response("[(1,)]"))
    middleware.wrap_tool_call(_make_request(), handler)

    assert len(trace) == 1


def test_middleware_records_tool_call_id():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    handler = MagicMock(return_value=_make_response())
    middleware.wrap_tool_call(_make_request(tool_call_id="call_xyz"), handler)

    assert trace.tool_calls[0].tool_call_id == "call_xyz"


def test_middleware_records_tool_name():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    handler = MagicMock(return_value=_make_response())
    middleware.wrap_tool_call(_make_request(tool_name="sql_db_schema"), handler)

    assert trace.tool_calls[0].tool_name == "sql_db_schema"


def test_middleware_records_args():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    handler = MagicMock(return_value=_make_response())
    middleware.wrap_tool_call(_make_request(args={"table_names": "Album, Artist"}), handler)

    assert trace.tool_calls[0].args == {"table_names": "Album, Artist"}


def test_middleware_records_response_content_as_string_result():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    handler = MagicMock(return_value=_make_response("[(347,)]"))
    middleware.wrap_tool_call(_make_request(), handler)

    assert trace.tool_calls[0].result == "[(347,)]"


def test_middleware_records_duration_as_positive_float():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    handler = MagicMock(return_value=_make_response())
    middleware.wrap_tool_call(_make_request(), handler)

    assert trace.tool_calls[0].duration_ms > 0


def test_middleware_calls_handler_with_unchanged_request():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    request = _make_request()
    handler = MagicMock(return_value=_make_response())
    middleware.wrap_tool_call(request, handler)

    handler.assert_called_once_with(request)


def test_middleware_returns_handler_response_unchanged():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    expected_response = _make_response("[(42,)]")
    handler = MagicMock(return_value=expected_response)

    result = middleware.wrap_tool_call(_make_request(), handler)

    assert result is expected_response


def test_middleware_records_in_order_across_multiple_invocations():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)
    handler = MagicMock(return_value=_make_response())

    middleware.wrap_tool_call(_make_request(tool_call_id="call_1"), handler)
    middleware.wrap_tool_call(_make_request(tool_call_id="call_2"), handler)
    middleware.wrap_tool_call(_make_request(tool_call_id="call_3"), handler)

    ids = [record.tool_call_id for record in trace.tool_calls]
    assert ids == ["call_1", "call_2", "call_3"]


def test_middleware_stringifies_non_string_response_content():
    trace = AgentTrace()
    middleware = make_log_tool_call_middleware(trace)

    handler = MagicMock(return_value=_make_response(content=[{"type": "text", "text": "hello"}]))
    middleware.wrap_tool_call(_make_request(), handler)

    assert isinstance(trace.tool_calls[0].result, str)