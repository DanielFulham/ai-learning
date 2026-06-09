import json
import logging

from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.messages.tool import ToolCall

from application.interfaces.video_agent_interface import VideoAgentInterface

logger = logging.getLogger(__name__)

class ManualLoopAgent(VideoAgentInterface):
    """Agent that runs the tool-calling dispatch loop imperatively.

    Uses a while-loop that runs until the model emits an AIMessage with no
    tool_calls. The simplest of the three orchestrations — no LCEL, no
    recursion, just a list and a loop. This is the shape lab23 used.

    Capped at max_iterations rounds. A misbehaving model that never stops
    calling tools will hit the cap and raise rather than exhausting the
    LLM provider's context window or rate limits.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        max_iterations: int = 25,
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._llm_with_tools = llm.bind_tools(tools)
        self._max_iterations = max_iterations

    def run(self, query: str) -> str:
        messages: list[AnyMessage] = [HumanMessage(content=query)]
        response = self._llm_with_tools.invoke(messages)
        messages.append(response)

        iteration = 0
        while getattr(messages[-1], "tool_calls", None):
            if iteration >= self._max_iterations:
                raise RuntimeError(
                    f"Agent exceeded max_iterations ({self._max_iterations}) without producing a final answer"
                )

            last = messages[-1]
            if not isinstance(last, AIMessage):
                raise TypeError(f"Expected AIMessage at top of loop, got {type(last).__name__}")

            for tool_call in last.tool_calls:
                tool_message = self._execute_tool(tool_call)
                messages.append(tool_message)

            response = self._llm_with_tools.invoke(messages)
            messages.append(response)
            iteration += 1

        final = messages[-1].content
        if not isinstance(final, str):
            raise TypeError(f"Expected string content from final AIMessage, got {type(final).__name__}")
        return final

    def _execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        tool_call_id = tool_call.get("id")
        if tool_call_id is None:
            raise ValueError(
                f"Tool call for '{tool_call.get('name')}' has no id; cannot construct ToolMessage. "
                "This indicates an LLM provider returning a malformed tool_call."
            )

        name = tool_call["name"]
        if name not in self._tools:
            return ToolMessage(
                content=f"Error: Unknown tool '{name}'. Available tools: {sorted(self._tools.keys())}",
                tool_call_id=tool_call_id,
            )
        try:
            result = self._tools[name].invoke(tool_call["args"])
            content = result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as e:
            logger.exception(
                "Tool '%s' raised during invocation (tool_call_id=%s, args=%s)",
                name,
                tool_call_id,
                tool_call.get("args"),
            )
            content = f"Error: {str(e)}"
        return ToolMessage(content=content, tool_call_id=tool_call_id)