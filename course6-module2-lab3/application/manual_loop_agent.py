from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.messages.tool import ToolCall

from application.interfaces.video_agent_interface import VideoAgentInterface


class ManualLoopAgent(VideoAgentInterface):
    """Agent that runs the tool-calling dispatch loop imperatively.

    Uses a while-loop that runs until the model emits an AIMessage with no
    tool_calls. The simplest of the three orchestrations — no LCEL, no
    recursion, just a list and a loop. This is the shape lab23 used.
    """

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]) -> None:
        self._tools = {t.name: t for t in tools}
        self._llm_with_tools = llm.bind_tools(tools)

    def run(self, query: str) -> str:
        messages: list[AnyMessage] = [HumanMessage(content=query)]
        response = self._llm_with_tools.invoke(messages)
        messages.append(response)

        while getattr(messages[-1], "tool_calls", None):
            last = messages[-1]
            if not isinstance(last, AIMessage):
                raise TypeError(f"Expected AIMessage at top of loop, got {type(last).__name__}")

            for tool_call in last.tool_calls:
                tool_message = self._execute_tool(tool_call)
                messages.append(tool_message)

            response = self._llm_with_tools.invoke(messages)
            messages.append(response)

        final = messages[-1].content
        if not isinstance(final, str):
            raise TypeError(f"Expected string content from final AIMessage, got {type(final).__name__}")
        return final

    def _execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        try:
            result = self._tools[tool_call["name"]].invoke(tool_call["args"])
            content = result if isinstance(result, str) else str(result)
        except Exception as e:
            content = f"Error: {str(e)}"
        return ToolMessage(content=content, tool_call_id=tool_call["id"])