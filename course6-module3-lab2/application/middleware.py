import time

from langchain.agents.middleware import wrap_tool_call

from domain.models import AgentTrace, ToolCallRecord


def make_log_tool_call_middleware(trace: AgentTrace):
    @wrap_tool_call
    def log_tool_call(request, handler):
        start = time.perf_counter()
        response = handler(request)
        duration_ms = (time.perf_counter() - start) * 1000

        tool_call = request.tool_call
        trace.append(ToolCallRecord(
            tool_call_id=tool_call["id"],
            tool_name=tool_call["name"],
            args=tool_call["args"],
            result=str(response.content),
            duration_ms=duration_ms,
        ))
        return response

    return log_tool_call