import json
import logging

from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableLambda

from application.interfaces.video_agent_interface import VideoAgentInterface

logger = logging.getLogger(__name__)

class RecursiveAgent(VideoAgentInterface):
    """Agent that loops the tool-calling dispatch until the model is done.

    The model decides how many tool calls are needed. The loop terminates when
    the model emits an AIMessage with no tool_calls. This is the shape
    production agent frameworks (create_agent, LangGraph) use internally,
    with extra niceties around retries and observability.

    Capped at max_iterations rounds. The underlying implementation uses
    Python recursion, so the cap also protects against blowing the call
    stack — a misbehaving model that loops forever would otherwise raise
    RecursionError after ~1000 frames.
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
        self._chain = self._build_chain()

    def run(self, query: str) -> str:
        result = self._chain.invoke({"query": query})
        final = result[-1].content
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
    
    def _process_tool_calls(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        last = messages[-1]
        tool_messages = [
            self._execute_tool(tc) for tc in getattr(last, "tool_calls", [])
        ]
        updated = messages + tool_messages
        next_response = self._llm_with_tools.invoke(updated)
        return updated + [next_response]

    def _should_continue(self, messages: list[AnyMessage]) -> bool:
        return bool(getattr(messages[-1], "tool_calls", None))

    def _recurse(self, messages: list[AnyMessage], iteration: int = 0) -> list[AnyMessage]:
        if self._should_continue(messages):
            if iteration >= self._max_iterations:
                raise RuntimeError(
                    f"Agent exceeded max_iterations ({self._max_iterations}) without producing a final answer"
                )
            return self._recurse(self._process_tool_calls(messages), iteration + 1)
        return messages

    def _build_chain(self):
        def initial_messages(state: dict) -> list[AnyMessage]:
            return [HumanMessage(content=state["query"])]

        def first_invoke(messages: list[AnyMessage]) -> list[AnyMessage]:
            return messages + [self._llm_with_tools.invoke(messages)]

        return (
            RunnableLambda(initial_messages)
            | RunnableLambda(first_invoke)
            | RunnableLambda(self._recurse)
        )